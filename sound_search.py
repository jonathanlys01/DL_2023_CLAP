"""
Retrieves a music / sound using a text query.
"""

import time
import argparse
from dataset import ESC_50, UrbanSound8k, FMA, Audioset
import os
import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset, one of ESC-50, FMA, UrbanSound8K or all (short names: e, f, u, a)")
parser.add_argument("--query", "-q", type=str, required=True, help="Text query")
parser.add_argument("--limit", "-l", type=int, default=-1, help="Limit the number of samples to search")
parser.add_argument("--model", "-m", type=str, choices=["music", "general",], default="general", help="Model type, default is general")
parser.add_argument("--number", "-n", type=int, default=1, help="Number of results to return")
args = parser.parse_args()

root = "downloads/"

ds_type = args.dataset

if ds_type.lower() in ["esc-50", "esc50", "esc","e"]:
    ds_type = "ESC-50"
    path_to_audio = os.path.join(root, "ESC-50-master", "audio")
    path_to_annotation = os.path.join(root, "ESC-50-master", "meta", "esc50.csv")
    
elif ds_type.lower() in ["fma", "freemusicarchive","f"]:
    ds_type = "FMA"
    path_to_audio = os.path.join(root, "fma_small")
    path_to_annotation = os.path.join(root, "fma_metadata", "tracks.csv")
    
elif ds_type.lower() in ["urbansound8k", "urbansound", "urban","u"]:
    ds_type = "UrbanSound8K"
    path_to_audio = os.path.join(root, "UrbanSound8K", "audio")
    path_to_annotation = os.path.join(root, "UrbanSound8K", "metadata", "UrbanSound8K.csv")

elif ds_type.lower() in ["audioset", "audio", "as"]:
    ds_type = "AudioSet"
    path = os.path.join(root, "audioset")

elif ds_type.lower() in ["all", "a"]:
    ds_type = "all"
    paths_to_audio = {
        "ESC-50": os.path.join(root, "ESC-50-master", "audio"),
        "FMA": os.path.join(root, "fma_small"),
        "UrbanSound8K": os.path.join(root, "UrbanSound8K", "audio"),
    }
    paths_to_annotation = {
        "ESC-50": os.path.join(root, "ESC-50-master", "meta", "esc50.csv"),
        "FMA": os.path.join(root, "fma_metadata", "tracks.csv"),
        "UrbanSound8K": os.path.join(root, "UrbanSound8K", "metadata", "UrbanSound8K.csv"),
    }
else:
    raise ValueError(f"Invalid dataset type, {ds_type}")

start = time.time()

print("Loading model:", end=" ")
if args.model == "general":
    cpt = "laion/clap-htsat-unfused"
elif args.model == "music":
    cpt = "laion/larger_clap_music"
print(args.model)

from transformers import ClapModel, ClapProcessor

processor = ClapProcessor.from_pretrained(cpt)
model = ClapModel.from_pretrained(cpt).to("cuda")

print(f"Model {args.model} loaded in {time.time() - start:.2f} seconds")
print()

if ds_type == "all":
    datasets = {
        "ESC-50": ESC_50(paths_to_audio["ESC-50"], paths_to_annotation["ESC-50"]),
        "FMA": FMA(paths_to_audio["FMA"], paths_to_annotation["FMA"]),
        "UrbanSound8K": UrbanSound8k(paths_to_audio["UrbanSound8K"], paths_to_annotation["UrbanSound8K"]),
    }
elif ds_type == "ESC-50":
    datasets = {
        "ESC-50": ESC_50(path_to_audio, path_to_annotation),
    }
elif ds_type == "FMA":
    datasets = {
        "FMA": FMA(path_to_audio, path_to_annotation),
    }
elif ds_type == "UrbanSound8K":
    datasets = {
        "UrbanSound8K": UrbanSound8k(path_to_audio, path_to_annotation),
    }
elif ds_type == "AudioSet":
    datasets = {
        "AudioSet": Audioset(path),
    }

if ds_type == "all":
    indexes = []
    for ds in datasets:
        indexes += [(ds, i) for i in range(len(datasets[ds]))]
    # [("esc", 0), ("esc", 1), ..., ("fma", 0), ("fma", 1), ..., ("urban", 0), ("urban", 1), ...]

else:
    indexes = [(ds_type, i) for i in range(len(datasets[list(datasets.keys())[0]]))]
    # [("esc", 0), ("esc", 1), ...]

import random as rd
if args.limit > 0:
    
    indexes = rd.sample(indexes, args.limit)
# else, all indexes

# Generating text embeddings
print()
print("Generating text embeddings")
texts = [args.query]

inputs_text = processor(text=texts, return_tensors="pt", padding=True)

for key, value in inputs_text.items():
        inputs_text[key] = value.to("cuda")

with torch.inference_mode():
    outputs_text = model.get_text_features(**inputs_text)


cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
cosine_sim = cosine_sim.to("cuda")

print("Searching for sounds")
pbar = tqdm(indexes)
best_sim = -1
best_filename = "None"
best_label = "None"
best_ds = "None"

results = []

for ds, ind in pbar:
    
    filename = datasets[ds].audios[ind]
    label = datasets[ds].annotations[ind]

    try:
        audio, sr = datasets[ds].open_audio(filename, sr=48000)
    except:
        print(f"Error loading {filename}, skipping")
        continue

    inputs_audio = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)

    for key, value in inputs_audio.items():
        inputs_audio[key] = value.to("cuda")

    with torch.inference_mode():
        outputs_audio = model.get_audio_features(**inputs_audio)

    sim = cosine_sim(outputs_text, outputs_audio).cpu().item()

    results.append({
        "filename": filename,
        "label": label,
        "sim": sim,
        "ds": ds,
    })

    if sim > best_sim:
        best_sim = sim
        best_filename = filename
        best_label = label
        best_ds = ds

    pbar.set_description(f"Best sim: {best_sim:.2f} (label: {best_label})")


print()

print("Top results:")
results = sorted(results, key=lambda x: x["sim"], reverse=True)
for i in range(args.number):
    s = f"#{i+1} ({results[i]['sim']:.3f}): {results[i]['filename']} ({results[i]['label']}) from {results[i]['ds']}"
    print(s)
