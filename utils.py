from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from PIL import Image
import pandas as pd
import numpy as np
import torch
import clip
import os

from dataset import ImageNet
from dataset import Flickr30k

######## Change this to your own paths ########
#FLICKR_PATH = "/cs_storage/public_datasets/flickr30k-labels/results_new.csv"
FLICKR_PATH = "flickr30k/Images/results.csv"

def load_dataset(dataset, image_processsor, batch_size=6):
    match dataset:
        case 'flickr30k':
            dataframe = pd.read_csv(FLICKR_PATH, sep="|", skipinitialspace=True)
            dataframe = dataframe.rename(columns={"image_name": "image", "comment": "caption"})
            dataset = Flickr30k(
                dataframe["image"].values,
                dataframe["caption"].values,
                transform=image_processsor,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=True
            )
        case _:
            raise Exception('No such dataset {dataset}')
    return dataloader

def load_model(model_name):
    match model_name:
        case 'vit-gpt2':
            loc = "ydshieh/vit-gpt2-coco-en"
            processor = ViTImageProcessor.from_pretrained(loc)
            mean = processor.image_mean
            std = processor.image_std
            tokenizer = AutoTokenizer.from_pretrained(loc)
            model = VisionEncoderDecoderModel.from_pretrained(loc).eval().cuda()
            model.eval()
            encoder = model.encoder
        case 'blip':
            loc = "Salesforce/blip-image-captioning-large"
            processor = BlipProcessor.from_pretrained(loc)
            mean = processor.image_processor.image_mean
            std = processor.image_processor.image_std
            model = BlipForConditionalGeneration.from_pretrained(loc).eval().cuda()
            tokenizer = None
            encoder = model.vision_model
        case 'blip2':
            loc = "Salesforce/blip2-opt-2.7b"
            processor = Blip2Processor.from_pretrained(loc, use_fast=False)
            model = Blip2ForConditionalGeneration.from_pretrained(loc, torch_dtype=torch.float16).cuda()
            model.eval()
            mean = processor.image_processor.image_mean
            std = processor.image_processor.image_std
            tokenizer = None
            encoder = model.vision_model
        case _:
            raise Exception('No such model {model_name}')
    return processor, tokenizer, model, encoder, mean, std
    

def preprocess_and_emb(imgs, feature_extractor, encoder):
    p_imgs = []
    for img in imgs:
        p_img = feature_extractor(images=img, return_tensors="pt").pixel_values
        p_imgs.append(p_img)
    p_imgs = torch.vstack(p_imgs).cuda()
    embs = encoder(p_imgs)[0]
    return p_imgs, embs

def save_img_and_text(img, text, image_mean, image_std, eps, i, target_img=False, targeted=False, adv=False):
    inv_normalize = transforms.Normalize(
        mean=[-image_mean[0]/image_std[0], -image_mean[1]/image_std[1], -image_mean[2]/image_std[2]],
        std=[1/image_std[0], 1/image_std[1], 1/image_std[2]]
    )
    inv_img = inv_normalize(img)
    # if targeted:
    #     folder_path = f"/home/razla/Text2Image-Attack/targeted/{eps}/{i}"
    # else:
    #     folder_path = f"/home/razla/Text2Image-Attack/results/{eps}/{i}"
    folder_path = f"/home/razla/Text2Image-Attack/specific/{eps}/{i}"
    os.makedirs(folder_path, exist_ok=True)
    if not target_img and adv:
        save_image(inv_img, f'{folder_path}/pert_{i}.png')
        f = open(f'{folder_path}/captions.txt', "a")
        f.write(f'Adversarial pred: {text}\n')
        f.close()
    elif target_img:
        save_image(inv_img, f'{folder_path}/target_{i}.png')
        f = open(f'{folder_path}/captions.txt', "a")
        f.write(f'Target pred: {text}\n')
        f.close()
    else:
        save_image(inv_img, f'{folder_path}/orig_{i}.png')    
        f = open(f'{folder_path}/captions.txt', "a")
        f.write(f'Original pred: {text}\n')
        f.close()
    
def predict(model_name, model, tokenizer, processor, image):
    
    # pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    # pixel_values = pixel_values.cuda()
    
    match model_name:
        case 'vit-gpt2':
            with torch.no_grad():
                output_ids = model.generate(image, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
        case 'blip':
            with torch.no_grad():
                if image.shape[0] == 2:
                    output_ids1 = model.generate(image[0].unsqueeze(0), max_length=16, num_beams=4).sequences
                    output_ids2 = model.generate(image[1].unsqueeze(0), max_length=16, num_beams=4).sequences
                    preds1 = processor.tokenizer.decode(output_ids1[0], skip_special_tokens=True)
                    preds2 = processor.tokenizer.decode(output_ids2[0], skip_special_tokens=True)
                    preds = [preds1, preds2] 
                else:
                    output_ids1 = model.generate(image[0].unsqueeze(0), max_length=16, num_beams=4).sequences
                    preds = processor.tokenizer.decode(output_ids1[0], skip_special_tokens=True)
        case 'blip2':
            # BLIP-2 expects PIL input

            # Convert to list of PIL images
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # Single image
                    image = [to_pil_image(image.cpu())]
                else:
                    image = [to_pil_image(im.cpu()) for im in image]

            inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=30)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            preds = [p.strip() for p in preds]

    return preds

def make_df(labels):
    dataframe = pd.read_csv(labels)
    train_dataframe = dataframe
    return train_dataframe

def build_loaders(captions):
    dataframe = pd.read_csv(captions)
    trans = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    dataset = Flickr30k(
        dataframe["image"].values,
        dataframe["caption"].values,
        transforms=trans,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=1,
        shuffle=False
    )
    return dataloader

def clip_score(clip_model, clip_preprocess, x, y):
    transform = transforms.ToPILImage()
    img = transform(x.squeeze(0))
    image = clip_preprocess(img).unsqueeze(0).cuda()
    text = clip.tokenize(y).cuda()

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)
    return probs

if __name__ == '__main__':
    dataloader = build_loaders("/cs_storage/public_datasets/flickr30k-labels/results_new.csv")
    imgs, labels = next(iter(dataloader))
    print(imgs.shape)
    print(labels)