import gradio as gr
import plotly.graph_objects as go
import argparse
from transformers import ClapModel, ClapProcessor
import numpy as np
import torch
import os
from sklearn.metrics import pairwise_distances  # cosine similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from augmentations import augmentations 
import threading
import time
import librosa
import soundfile as sf

# Chose the model

parser = argparse.ArgumentParser()
parser.add_argument("--share", "-s", type=argparse.BooleanOptionalAction, default=False, help="Share the app")

args = parser.parse_args()

share = args.share


# Load the models

start = time.time()

cpt1 = "laion/clap-htsat-unfused"
cpt2 = "laion/larger_clap_music"

processor_general = ClapProcessor.from_pretrained(cpt1)
model_general = ClapModel.from_pretrained(cpt1).to("cuda")

processor_music = ClapProcessor.from_pretrained(cpt2)
model_music = ClapModel.from_pretrained(cpt2).to("cuda")

model_general.eval()
model_music.eval()

print(f"Models loaded in {time.time() - start:.2f} seconds")

def embed(text, model):
    if model == "general":
        processor = processor_general
        model = model_general
    elif model == "music":
        processor = processor_music
        model = model_music
    else:
        raise ValueError(f"Invalid model, {model}")
    texts = [text]
    inputs_text = processor(text=texts, return_tensors="pt", padding=True)

    for key, value in inputs_text.items():
        inputs_text[key] = value.to("cuda")

    with torch.inference_mode():
        outputs_text = model.get_text_features(**inputs_text)
    
    return outputs_text.cpu().numpy() # shape (1, 768)

def project(X, method="pca"):
    if method == "tsne":
        distances = pairwise_distances(X, metric="cosine", n_jobs=-1)
        X_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(distances)
    elif method == "pca":
        X_embedded = PCA(n_components=2).fit_transform(X)
    else:
        raise ValueError(f"Invalid method, {method}")

    return X_embedded

def plot(audio_features_embedded,
        projected_aug_text_features,
        projected_query_embedding,
        audio_labels, classes, title=""):

    fig = go.Figure()

    for i, txt in enumerate(classes):
        fig.add_trace(go.Scattergl(x=audio_features_embedded[audio_labels == i][:, 0],
                                y=audio_features_embedded[audio_labels == i][:, 1],
                                mode='markers', 
                                name=txt,
                                    marker=dict(
                                        symbol='circle',
                                        opacity=0.5,
                                        size=5,   
                                        line_width=0.5)))
        
    fig.add_trace(go.Scattergl(x=projected_aug_text_features[:, 0],
                            y=projected_aug_text_features[:, 1],
                            mode='markers', 
                            name="Text",
                                marker=dict(
                                    symbol='square',
                                    color="green",
                                    opacity=0.5,
                                    size=5,   
                                    line_width=0.5)))
    fig.add_trace(go.Scattergl(x=[projected_query_embedding[0]],
                            y=[projected_query_embedding[1]],
                            mode='markers', 
                            name="Query",
                                marker=dict(
                                    symbol='star',
                                    color="red",
                                    opacity=1,
                                    size=30,   
                                    line_width=0.5)))

    fig.update_layout(title=title)
    fig.show()

def background_task(audio_features, raw_text_features, aug_text_features, query_embedding, audio_labels, classes, viz, title):
    all_features = np.concatenate([audio_features, raw_text_features, aug_text_features, query_embedding], axis=0)
    all_features_projected = project(all_features, method=viz)

    projected_audio_features = all_features_projected[:len(audio_features)]
    _ = all_features_projected[len(audio_features):len(audio_features)+len(raw_text_features)]
    projected_aug_text_features = all_features_projected[len(audio_features)+len(raw_text_features):-1]
    projected_query_embedding = all_features_projected[-1]

    plot(projected_audio_features, 
        projected_aug_text_features, 
        projected_query_embedding, 
        audio_labels, classes, title)

# Define the search function

root = "cached_features"

def audio_retrieval(dataset, 
                    model,
                    query, viz):

    classes = list(augmentations[dataset].keys())

    # Load the features

    audio_file = np.load(os.path.join(root, dataset, 
                                      model,
                                      "audio_features.npz"), allow_pickle=True)
    audio_features = audio_file.get("audio_features")
    audio_labels = audio_file.get("audio_labels")
    audio_paths = audio_file.get("audio_paths")

    text_file = np.load(os.path.join(root, dataset,
                                     model,
                                     "text_features.npz"), allow_pickle=True)
    raw_text_features = text_file.get("raw_features")
    aug_text_features = text_file.get("augmented_features")

    # Embed the query
    query_embedding = embed(query, model)

    # Compute the scores (cosine similarity)

    distances = pairwise_distances(query_embedding, audio_features, metric="cosine", n_jobs=-1)[0]

    scores = 1 - distances # distance to similarity


    ranks = np.argsort(scores)[::-1] # high is good

    # Get the top results

    temp = [audio_labels[i] for i in ranks[:5]]
    temp = [classes[i] for i in temp]
    temp = [f"#{i+1}: {j}" for i, j in enumerate(temp)]
    s1,s2,s3, s4, s5 = temp

    displayed_scores = {
        s1: scores[ranks[0]],
        s2: scores[ranks[1]],
        s3: scores[ranks[2]],
        s4: scores[ranks[3]],
        s5: scores[ranks[4]],
    }

    paths = [audio_paths[i] for i in ranks[:5]]

    L = []

    for path in paths:
        print(path)
        info = sf.info(path)
        y, sr = librosa.load(path, sr=info.samplerate)
        if info.subtype in ["FLOAT32", "MPEG_LAYER_III"]:
            y = np.clip(y, a_min=-1., a_max=1.)
            y = (y* 32767.).astype(np.int16)
        L.append(
            (sr, y) # (sample rate, audio), required by gradio
        )
    print()

    a1, a2, a3, a4, a5 = L



    if viz != "None":
        t = threading.Thread(target=background_task, args=(audio_features, 
                                                           raw_text_features, 
                                                           aug_text_features, 
                                                           query_embedding, 
                                                           audio_labels, 
                                                           classes, viz, f"Q: {query}, DS: {dataset}"))
        t.start()


    return displayed_scores, a1, a2, a3, a4, a5


# Launch the interface

demo = gr.Interface(
    audio_retrieval,
    [
        gr.Radio(["ESC-50", "UrbanSound8K", "FMA", "Audioset"], label="Dataset", info="Choose the dataset"),
        gr.Radio(["general", "music"], label="Model", info="Choose the model"),
        gr.Textbox(label="Text"),
        gr.Radio(["None","pca", "tsne"], label="Visualization", info="Visualization method (background task)",value="None"),
    ],
    [   gr.Label(num_top_classes=5, label="Cosine similarity ranking (top 3)"),
        gr.Audio(label="Audio 1"),
        gr.Audio(label="Audio 2"),
        gr.Audio(label="Audio 3"),
        gr.Audio(label="Audio 4"),
        gr.Audio(label="Audio 5"),
    ],
    
)
            
demo.launch(share = share)
    



    

