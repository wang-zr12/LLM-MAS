# Google Colab Model Evaluation Script

from google.colab import drive

drive.mount('/content/drive')

import sys
import torch
import json
import tqdm
import os, gc

# Modify the path to load from Google Drive
reward_model_path = '/content/drive/MyDrive/BaichuanCharRM/'

# Add the model directory to Python path
sys.path.append('/content/drive/MyDrive/')

from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

# Configuration
max_seq_length = 4096

# Load character profiles
with open("character_profiles.json", "r") as f:
    character_profile = json.load(f)

# Load generation records
with open("generation_trans_cot.jsonl", 'r', encoding='utf-8') as f:
    records = json.load(f)


def format_input(example):
    """Format input for the reward model"""
    input_text = "<RoleInfo>\n\n" \
                 + str(character_profile[example['role']]) + "\n\n<Context>\n\n" + example[
                     'context'] + "\n\n<Response>\n\n" + example['model_output'] + "\n\n<Dimension>\n\n" + example[
                     "metric_zh"]
    return input_text


# === Avoid reloading model if already loaded ===
if 'base_model' not in globals():
    print("Model not loaded yet. Loading tokenizer and model to GPU...")
    tokenizer = BaichuanTokenizer.from_pretrained(reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = BaichuanCharRM.from_pretrained(reward_model_path, torch_dtype=torch.bfloat16).cuda()
    print("Model loaded.")
else:
    print("Model already loaded in GPU memory.")

# Evaluate records
for record in tqdm.tqdm(records):
    input_text = format_input(record)
    input_ids = tokenizer.encode(text=input_text, add_special_tokens=False) + [tokenizer.eos_token_id]

    # Truncate input if too long
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]

    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

    # Compute score
    with torch.no_grad():
        output = base_model(input_ids=input_ids)
        score = output[1].item() * 4 + 1

        record[record['metric_en']] = score

    # Free GPU
    del input_ids, output
    torch.cuda.empty_cache()
    gc.collect()

# Ensure the results directory exists
os.makedirs('/content/drive/MyDrive/results', exist_ok=True)

# Save results
with open('/content/drive/MyDrive/results/evaluatio_cot.jsonl', 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print("Evaluation completed. Results saved to /content/drive/MyDrive/results/evaluation.jsonl")
