import time
import argparse
from dataset import ESC_50, UrbanSound8k, FMA
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from augmentations import augmentations

parser = argparse.ArgumentParser()

help = "Dataset type, one of ESC-50, FMA, UrbanSound8K (short names: e, f, u)"
parser.add_argument("--dataset", "-d", type=str, required=True, help=help)
parser.add_argument("--limit", "-l", type=int, default=-1, help="Limit number of samples")
parser.add_argument("--plot", "-p", choices=["no", "cm", "audio", "all"], 
                    default="no", help="Plot confusion matrix, audio, or both, default no")
parser.add_argument("--model", "-m", type=str, 
                    choices=["music", "general", "default"], default="default", 
                    help="Model type, default will choose the best model for the dataset")
parser.add_argument("--topk", "-k", type=int, default=1, help="Top k predictions")
parser.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=False, 
                    help="Verbose mode")
args = parser.parse_args()

ds_type = args.dataset
root = "downloads/"

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
    
else:
    raise ValueError(f"Invalid dataset type, {ds_type}")

start = time.time()

print("Loading model:", end=" ")
if args.model == "default":
    if ds_type in ["ESC-50", "UrbanSound8K"]:
        args.model = "general"
    elif ds_type == "FMA":
        args.model = "music"
print(args.model)
        
if args.model == "general":
    cpt = "laion/clap-htsat-unfused"
elif args.model == "music":
    cpt = "laion/larger_clap_music"
    
from transformers import ClapModel, ClapProcessor

model = ClapModel.from_pretrained(cpt).to("cuda")
processor = ClapProcessor.from_pretrained(cpt)  
print(f"Model {args.model} loaded in {time.time() - start:.2f} seconds")
print()
print(f"Loading dataset {ds_type}")
if ds_type == "ESC-50":
    dataset = ESC_50(path_to_audio, path_to_annotation)
elif ds_type == "FMA":
    dataset = FMA(path_to_audio, path_to_annotation)
elif ds_type == "UrbanSound8K":
    dataset = UrbanSound8k(path_to_audio, path_to_annotation)
print()

preds = []
limit = len(dataset.audios) if args.limit == -1 else args.limit
labels = dataset.classes

augmentations = augmentations[ds_type]

assert labels == list(augmentations.keys()), "Labels do not match augmentations"

texts = [augmentations[label] for label in labels]
inputs_text = processor(text=texts, return_tensors="pt", padding=True)

for key, value in inputs_text.items():
        inputs_text[key] = value.to("cuda")

with torch.inference_mode():
    outputs_text = model.get_text_features(**inputs_text)

if args.verbose:
    pbar = range(min(limit, len(dataset.audios))) # no tqdm if verbose
else:
    pbar = tqdm(range(min(limit, len(dataset.audios))))

cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for ind in pbar:
    filename = dataset.audios[ind]
    label = dataset.annotations[ind]

    try:
        audio, sr = dataset.open_audio(filename, sr=48000)
    except:
        print(f"Error opening {filename}")
        continue

    if args.plot in ["all", "audio"]:
        plt.plot(audio)
        plot_name = filename.split("/")[-1].split(".")[0]
        plt.title(f"Audio: {plot_name} / Label: {label}")
        plt.savefig(f"temp/audio_{plot_name}.png")
        plt.savefig(f"figs/last_audio.png")
        plt.close()

    inputs_audio = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)

    for key, value in inputs_audio.items():
        inputs_audio[key] = value.to("cuda")

    with torch.inference_mode():
        outputs_audio = model.get_audio_features(**inputs_audio)


    sim = cosine_sim(outputs_text, outputs_audio)

    pred_index_list = torch.argsort(sim, descending=True)[:args.topk].cpu().numpy()

    if any([dataset.classes[pred_index] == label for pred_index in pred_index_list]):
        pred_index = dataset.classes.index(label)
    else:
        pred_index = pred_index_list[0] # argmax if no match

    preds.append(dataset.classes[pred_index])
    
    acc = 100 * np.mean(np.array(preds) == np.array(dataset.annotations[:len(preds)]))
    
    if args.verbose:
        print(f"True: {label}, Predicted: {dataset.classes[pred_index]}")
        print(f"Top {args.topk} Accuracy: {acc:.2f}%")
    else:
        pbar.set_description(f"Top {args.topk} accuracy: {acc:.2f}%")
    
print(f"Final top {args.topk} accuracy: {acc:.2f}")

if args.plot in ["all", "cm"]:
    all_classes = np.array(dataset.classes)

    confusion_matrix = np.zeros((len(all_classes), len(all_classes)))

    for i in range(len(preds)):
        row = np.where(all_classes == preds[i])[0][0]
        col = np.where(all_classes == dataset.annotations[i])[0][0]
        confusion_matrix[row, col] += 1
        
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion_matrix, cmap="Blues", interpolation="nearest")
    middle = (np.min(confusion_matrix) + np.max(confusion_matrix)) / 2

    ###
    if len(all_classes) < 15: # don't show text if there are too many classes
        for i in range(len(all_classes)):
            for j in range(len(all_classes)):
                if confusion_matrix[i, j] == 0:
                    continue
                else:
                    text = f"{confusion_matrix[i, j]:.1e}"
                    color = "white" if confusion_matrix[i, j] > middle else "black"
                    # The cmap is Blues so the text will be white if the background is dark
                    plt.text(j, i, text , ha="center", va="center", color=color,)
    ###

    plt.xticks(np.arange(len(all_classes)), all_classes, rotation=90)
    plt.yticks(np.arange(len(all_classes)), all_classes)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.colorbar()
    plt.tight_layout()
    plt.title(f"Confusion matrix (top{args.topk} accuracy: {acc:.2f}%) (ds: {ds_type})")
    
    plt.savefig(f"figs/last_confusion_matrix_{ds_type.lower()}_{args.topk}.png")
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"temp/confusion_matrix_{date}.png")

end = time.time()
import datetime

elapsed = str(datetime.timedelta(seconds=(end - start))) # convert time to hh:mm:ss.nnnnnn format
elapsed = elapsed.split(".")[0] # remove microseconds
if elapsed.startswith("0:"):
    elapsed = elapsed[2:] # remove leading 0:

print(f"Total elapsed time: {elapsed}")


