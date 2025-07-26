
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

import argparse
import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resize_224 = transforms.Resize((224, 224), antialias=True)

def pgd(model, model_name, encoder, tokenizer, image_processor, image_mean, image_std, clip_model, loader, nb_epoch, eps, c = 0.1, targeted=True, lr=0.05, nb_imgs=1000, projection=None):
    total_losses = []
    clip_losses = []
    encoder = encoder.to(device)

    if projection is not None:
        projection = projection.to(device)

    imgs_counter = 0
    for i, batch in enumerate(loader):
        batch = {k: v.cuda() if k!="caption" else v for k, v in batch.items()}
        x = batch['image']
        y = batch['caption']
        x = torch.stack([x[0], x[5]])
        y = [y[0], y[5]]

        with torch.no_grad():
            model_orig_pred = predict(model_name, model, tokenizer, image_processor, x)
            pred_texts = clip.tokenize(model_orig_pred).cuda()
            true_texts = clip.tokenize(y).cuda()

            pred_texts_features = clip_model.encode_text(pred_texts)
            true_texts_features = clip_model.encode_text(true_texts)
            cos_sim = F.cosine_similarity(pred_texts_features, true_texts_features)

        if any(len(t) > 200 for t in y):
            continue

        with torch.no_grad():
            y_true_emb = clip_model.encode_text(true_texts)

            x_resized = torch.stack([resize_224(img) for img in x])
            x_true_emb = clip_model.encode_image(x_resized)

        clip_score_before = F.cosine_similarity(x_true_emb, y_true_emb).mean()

        imgs_counter += 1
        if imgs_counter > nb_imgs:
            break

        noise = torch.zeros_like(x[0:1], device=device, dtype=x.dtype , requires_grad=True)
        optimizer = AdamW([noise], lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=nb_epoch)

        with torch.no_grad():
            x_pil = [to_pil_image(img.cpu()) for img in x]
            x_enc = image_processor(images=x_pil, return_tensors="pt", do_rescale=False).pixel_values.to(device, dtype=torch.float16)
            x_emb = encoder(x_enc).pooler_output[0].detach()
            x_emb_proj = projection(x_emb.float()) if projection else x_emb.float()

        save_img_and_text(resize_224(x[0]), model_orig_pred[0], image_mean, image_std, eps, i, target_img=True, targeted=True, adv=False)
        save_img_and_text(resize_224(x[1]), model_orig_pred[1], image_mean, image_std, eps, i, target_img=False, targeted=True, adv=False)
        print(f'Cos sim: {cos_sim.mean().item():.4f}')
        print(f'Pred: {model_orig_pred}')
        print(f'Orig: {y}')

        cur_losses = []

        for epoch in range(nb_epoch):
            optimizer.zero_grad()
            x_adv = (x[1:2] + noise).clamp(0, 1)
            x_adv_resized =  F.interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)

            if isinstance(image_mean, list):
                image_mean = torch.tensor(image_mean, device=x_adv_resized.device)
            if isinstance(image_std, list):
                image_std = torch.tensor(image_std, device=x_adv_resized.device)

            mean = image_mean.view(1, 3, 1, 1)
            std = image_std.view(1, 3, 1, 1)
            x_adv_normalized = (x_adv_resized - mean) / std
            x_adv_normalized = x_adv_normalized.to(dtype=torch.float16)

            #x_adv_enc = image_processor(images=[to_pil_image(img.cpu()) for img in x_adv_resized], return_tensors="pt", do_rescale=False).pixel_values.to(device, dtype=torch.float16)
            x_adv_emb = encoder(x_adv_normalized).pooler_output
            x_adv_proj = projection(x_adv_emb.float()) if projection else x_adv_emb.float()

            l2_dist = torch.norm(noise.view(noise.shape[0], -1), p=2, dim=1)

            if not targeted:
                untargeted_loss = F.cosine_similarity(x_adv_proj, x_emb_proj.unsqueeze(0)).mean()
                loss = untargeted_loss + c * l2_dist
            else:
                targeted_loss = 1 - F.cosine_similarity(x_adv_proj, x_emb_proj.unsqueeze(0)).mean()
                loss = targeted_loss + c * l2_dist

            loss.backward()
            torch.nn.utils.clip_grad_norm_([noise], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            noise.data.clamp_(-eps, eps)
            cur_losses.append(loss.item())

            if (epoch + 1) % 100 == 0:
                #print(f"   Epoch {epoch+1}/{nb_epoch}, loss={loss.item():.4f}")
                max_abs_noise = noise.data.abs().max().item()
                print(f"   Epoch {epoch+1}/{nb_epoch}, loss={loss.item():.4f}, target_loss={targeted_loss.item():.4f}, l2={l2_dist.item():.4f}, max_noise={max_abs_noise:.4f}")

        with torch.no_grad():
            adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
            print(f'After attack:\n\t{adv_pred}')

            adv_tokenized = clip.tokenize(adv_pred).cuda()
            y_adv_emb = clip_model.encode_text(adv_tokenized)
            x_adv_resized = torch.stack([resize_224(img) for img in x_adv])
            x_adv_emb_clip = clip_model.encode_image(x_adv_resized)

        clip_score_after = F.cosine_similarity(x_adv_emb_clip, y_adv_emb).mean()
        save_img_and_text(resize_224(x_adv[0]), adv_pred, image_mean, image_std, eps, i, target_img=False, targeted=targeted, adv=True)
        total_losses.append(cur_losses)
        clip_losses.append((clip_score_before.item(), clip_score_after.item()))
        print(f'CLIP loss before {clip_score_before:.4f} and after {clip_score_after:.4f}\n')

        torch.cuda.empty_cache()

    return total_losses, clip_losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='blip2')
    parser.add_argument("--dataset", type=str, default='flickr30k')
    parser.add_argument("--eps", type=float, default=50/255)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_imgs", type=int, default=1000)
    args = parser.parse_args()

    model_name = args.model
    dataset = args.dataset
    eps = args.eps
    nb_epoch = args.n_epochs
    n_imgs = args.n_imgs

    clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name=model_name)
    image_mean = torch.tensor(image_mean).view(3, 1, 1).to(device)
    image_std = torch.tensor(image_std).view(3, 1, 1).to(device)

    dataloader = load_dataset(dataset, image_processor, batch_size=6)

    if model_name == "blip2":
        projection = nn.Linear(1408, 512).to(device)
    else:
        projection = None

    total_losses, clip_losses = pgd(model, model_name, encoder, tokenizer, image_processor, image_mean, image_std,
                                        clip_model, dataloader, nb_epoch, eps, c=0.1, targeted=True, lr=0.01,
                                        nb_imgs=n_imgs, projection=projection)

    mean_loss = sum(loss[-1] for loss in total_losses) / len(total_losses)
    print(f'Mean last loss: {mean_loss}')

    mean_before = sum(x for x, _ in clip_losses) / len(clip_losses)
    mean_after = sum(y for _, y in clip_losses) / len(clip_losses)
    print(f'Mean CLIP loss before: {mean_before}')
    print(f'Mean CLIP loss after: {mean_after}')

    std_before = (sum((x - mean_before)**2 for x, _ in clip_losses) / len(clip_losses))**0.5
    std_after = (sum((y - mean_after)**2 for _, y in clip_losses) / len(clip_losses))**0.5
    print(f'STD CLIP loss before: {std_before}')
    print(f'STD CLIP loss after: {std_after}')
