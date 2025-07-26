# I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models

## This paper was accepted to European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2023 (ECML-PKDD 2023)

Code accompanying the paper:
[I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models", ECML-PKDD Machine Learning and Cybersecurity Workshop, 2023](https://arxiv.org/abs/2306.07591).

### Abstract:
Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

## Prerequisites
    conda create -n isdp python=3.10.9
    pip install -r requirements.txt

## Download the Flickr30k dataset
1. Download dataset from this link: [dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k)
2. Update the FLICKR_PATH variable in utils.py accordingly

## Run
    python attack_targeted.py/attack_untargeted.py --model=<model_name> --dataset=<dataset_name> --eps=<epsilon> --n_epochs=<num_epochs> --n_imgs=<n_images>

    python attack_targeted.py --model=blip2 --dataset=flickr30k --eps=100 --n_epochs=1000 --n_imgs=1000

- The values used in the paper are:
  - model=vit-gpt2
  - dataset=flickr30k
  - n_epochs=1000
  - images=1000

If you wish to cite this paper:
```
@article{lapid2023see,
  title={I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models},
  author={Lapid, Raz and Sipper, Moshe},
  journal={arXiv preprint arXiv:2306.07591},
  year={2023}
}
```
![alt text](https://github.com/razla/I-See-Dead-People-Gray-Box-Adversarial-Attack-on-Image-To-Text-Models/blob/main/figures/examples.png)
