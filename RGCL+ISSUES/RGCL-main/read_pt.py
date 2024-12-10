import torch
import clip
import json
from tqdm import tqdm
from transformers import pipeline
import os

from textualInversion import TextualInversion, LinearProjection

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Running on GPU")
else:
    print("Running on CPU")

data_path = "./data/CLIP_Embedding/HarMeme/train_openai_clip-vit-large-patch14-336_HF_original.pt"
data = torch.load(data_path)
file_path = './data/gt/HarMeme/train.jsonl'

def get_text_by_id(file_path, target_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if entry['id'] == target_id:
                return entry['text']
    return None

def summarize_text(text, max_words):
    max_words = min(len(text.split()), max_words)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summary = summarizer(text, max_new_tokens=max_words, min_length=1, do_sample=False)[0]['summary_text']
    return summary

num_entries = data['img_feats'].size(0)
img_map = LinearProjection(1024, 1024, 1, [0.2, 0.4, 0.1]).to(device)
clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
clip_model.visual.proj = None
clip_model.float()
for _, p in clip_model.named_parameters():
    p.requires_grad_(False)
text_inv = TextualInversion(clip_model, 1024, True, True, False, [0.2, 0.4, 0.1], True, True, 1024, 1).to(device)

torch.cuda.empty_cache()

checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_file = os.path.join(checkpoint_dir, 'progress.pth')

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    start_idx = checkpoint['start_idx']
    data['img_feats'][:start_idx] = checkpoint['img_feats']
    data['text_feats'][:start_idx] = checkpoint['text_feats']
    print(f"Resuming from index {start_idx}")
else:
    start_idx = 0

for i in tqdm(range(start_idx, num_entries), desc="Processing entries"):
    try:
        img_feats = data['img_feats'][i].unsqueeze(0).to(device)
        target_id = data['ids'][i // 4][i % 4]
        text = get_text_by_id(file_path, target_id).replace("\n", " ")
        text = summarize_text(text, 30)
        prompt = clip.tokenize(f'{"a photo of $ "}, {text}').to(device)
        data['img_feats'][i] = img_map(img_feats).detach().cpu()
        data['text_feats'][i] = text_inv(prompt, img_feats).detach().cpu()

        del img_feats
        del prompt
        del text
        torch.cuda.empty_cache()

        if i % 100 == 0 or i == num_entries - 1:
            checkpoint_data = {
                'start_idx': i + 1,
                'img_feats': data['img_feats'].clone(),
                'text_feats': data['text_feats'].clone()
            }
            torch.save(checkpoint_data, checkpoint_file)

    except Exception as e:
        print(f"Error at index {i}: {e}")

torch.save(data, "./data/CLIP_Embedding/HarMeme/train_openai_clip-vit-large-patch14-336_HF.pt")
print(data['img_feats'].shape, data['text_feats'].shape)
