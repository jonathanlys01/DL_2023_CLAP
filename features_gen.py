"""
Generates features for a given dataset and saves them in the cached_features folder.

Similar arguments to the main.py file.
"""

import time
import argparse
from dataset import ESC_50, UrbanSound8k, FMA, Audioset
import os
import numpy as np
import torch
from tqdm import tqdm
from augmentations import augmentations

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    help = "Dataset type, one of ESC-50, FMA, UrbanSound8K, AudioSet (short names: e, f, u, a)"
    parser.add_argument("--dataset", "-d", type=str, required=True, help=help)

    parser.add_argument("--model", "-m", type=str, 
                        choices=["music", "general", "default"], default="default", 
                        help="Model type, default will choose the best model for the dataset")
    
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
    elif ds_type.lower() in ["audioset", "audio", "a"]:
        ds_type = "Audioset"
        path = os.path.join(root, "audioset")
        
    else:
        raise ValueError(f"Invalid dataset type, {ds_type}")

    start = time.time()

    print("Loading model:", end=" ")
    if args.model == "default":
        if ds_type in ["ESC-50", "UrbanSound8K", "Audioset"]:
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
    elif ds_type == "Audioset":
        dataset = Audioset(path)
    print()
    
    path_to_features = os.path.join("cached_features", ds_type)

    if not os.path.exists(path_to_features):
        os.makedirs(path_to_features)
        
    print("Generating audio features")
    
    audio_labels = []
    audio_features = []
    
    for i in tqdm(range(len(dataset))):
        
        filename = dataset.audios[i]
        label = dataset.annotations[i] # won't be encoded, but used for classification
        
        try:
            audio, sr = dataset.open_audio(filename, sr=48000)
        except:
            print(f"Error opening {filename}")
            continue # next iteration
        
        inputs_audio = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)
        
        for key, value in inputs_audio.items():
            inputs_audio[key] = value.to("cuda")

        with torch.inference_mode():
            outputs_audio = model.get_audio_features(**inputs_audio)
            
        audio_features.append(outputs_audio.cpu().numpy().reshape(-1)) # add audio features as a 1D numpy array
        audio_labels.append(dataset.classes.index(label)) # add numerical label
        
    audio_features = np.array(audio_features) # shape (n_samples, n_features)
    audio_labels = np.array(audio_labels) # shape (n_samples)
    
    np.savez_compressed(os.path.join(path_to_features, "audio_features.npz"), 
                        audio_features=audio_features, audio_labels=audio_labels)
    
    print(f"Audio features saved in {path_to_features}")
    
    print("Generating text features")
    
    raw_texts = dataset.classes
    augmented_texts = [augmentations[ds_type][label] for label in raw_texts]
    
    inputs_raw = processor(text=raw_texts, return_tensors="pt", padding=True)
    inputs_augmented = processor(text=augmented_texts, return_tensors="pt", padding=True)
    
    for key, value in inputs_raw.items():
        inputs_raw[key] = value.to("cuda")
    for key, value in inputs_augmented.items():
        inputs_augmented[key] = value.to("cuda")
    
    with torch.inference_mode():
        outputs_raw = model.get_text_features(**inputs_raw)
        outputs_augmented = model.get_text_features(**inputs_augmented)
        
    raw_features = outputs_raw.cpu().numpy()
    augmented_features = outputs_augmented.cpu().numpy()
    
    np.savez_compressed(os.path.join(path_to_features, "text_features.npz"), 
                        raw_features=raw_features, augmented_features=augmented_features)
    
    print(f"Text features saved in {path_to_features}")
    
    print(f"Total time: {time.time() - start:.2f} seconds")    
    
        
        
        
        
        
        
        
        
        
        
        