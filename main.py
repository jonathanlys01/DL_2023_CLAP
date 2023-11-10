import time
import argparse
from dataset import ESC_50
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from augmentations import augmentations # dict with augmented text labels

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_audio", "-a", type=str, required=True, help="Path to audio files")
parser.add_argument("--path_to_annotation", "-t", type=str, required=True, help="Path to annotation files")
parser.add_argument("--plot", "-p", action=argparse.BooleanOptionalAction, default=False, help="Plot all audios")

args = parser.parse_args()

print()

print("Loading dataset...")
esc_50 = ESC_50(args.path_to_audio, args.path_to_annotation)

start = time.time()

from transformers import ClapModel, ClapProcessor

print("Loading model...")

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

print(f"Model loaded in {time.time() - start:.2f} seconds")

preds = []
limit = len(esc_50.audios) # no limit
labels = esc_50.classes
assert labels == list(augmentations.keys()), "Labels do not match"
texts = [augmentations[label] for label in labels]
inputs_text = processor(text=texts, return_tensors="pt", padding=True)

for key, value in inputs_text.items():
        inputs_text[key] = value.to("cuda")

with torch.inference_mode():
    outputs_text = model.get_text_features(**inputs_text)

pbar = tqdm(range(min(limit, len(esc_50.audios))))

for ind in pbar:
    filename = esc_50.audios[ind]
    label = esc_50.annotations[ind]

    audio, sr = esc_50.open_audio(filename, sr=48000)

    plt.plot(audio)
    plot_name = filename.split("/")[-1].split(".")[0]
    plt.title(f"Audio: {plot_name} / Label: {label}")
    if args.plot:
        plt.savefig(f"temp/audio_{plot_name}.png")
        plt.savefig(f"last_audio.png")
    plt.close()

    inputs_audio = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

    for key, value in inputs_audio.items():
        inputs_audio[key] = value.to("cuda")

    with torch.inference_mode():
        outputs_audio = model.get_audio_features(**inputs_audio)

    cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    sim = cosine_sim(outputs_text, outputs_audio)

    pred_index = torch.argmax(sim).item()
    
    preds.append(esc_50.classes[pred_index])
    
    acc = 100 * np.mean(np.array(preds) == np.array(esc_50.annotations[:len(preds)]))
    
    pbar.set_description(f"Accuracy: {acc:.2f}%")
    

print(f"Accuracy: {acc:.2f}")

classes_pred = np.unique(preds)
classes_true = np.unique(esc_50.annotations)

all_classes = np.unique(np.concatenate((classes_pred, classes_true)))

confusion_matrix = np.zeros((len(all_classes), len(all_classes)))

for i in range(len(preds)):
    row = np.where(all_classes == preds[i])[0][0]
    col = np.where(all_classes == esc_50.annotations[i])[0][0]
    confusion_matrix[row, col] += 1
    
plt.figure(figsize=(12, 12))
plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")
plt.xticks(np.arange(len(all_classes)), all_classes, rotation=90)
plt.yticks(np.arange(len(all_classes)), all_classes)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.colorbar()
plt.tight_layout()
plt.title(f"Confusion matrix (Accuracy: {acc:.2f}%)")

plt.savefig("confusion_matrix.png")


end = time.time()
import datetime

elapsed = str(datetime.timedelta(seconds=(end - start))) # convert time to hh:mm:ss.nnnnnn format
elapsed = elapsed.split(".")[0] # remove microseconds
if elapsed.startswith("0:"):
    elapsed = elapsed[2:] # remove leading 0:

print(f"Elapsed time: {elapsed}")


