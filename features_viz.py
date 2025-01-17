"""
Allows the visualization of the features of the dataset in a 2D space through t-SNE or PCA.
"""

import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from augmentations import augmentations
from time import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()

help = "Dataset type, one of ESC-50, FMA, UrbanSound8K, AudioSet (short names: e, f, u, a)"
parser.add_argument("--dataset", "-d", type=str, required=True, help=help)
parser.add_argument("--type", "-t", type=str, default="tsne", help="Method to use for visualization, one of tsne, pca")
parser.add_argument("--model", "-m", type=str, choices=["music", "general"], default="general", help="Model type, default is general")
args = parser.parse_args()

method = args.type

if method in ["tsne", "t-sne", "t"]:
    method = "tsne"
elif method in ["pca", "p"]:
    method = "pca"
else:
    raise ValueError(f"Invalid method, {method}")
print("Using", method)
root = "cached_features"

ds_type = args.dataset

if ds_type.lower() in ["esc-50", "esc50", "esc","e"]:
    ds_type = "ESC-50"
elif ds_type.lower() in ["fma", "freemusicarchive","f"]:
    ds_type = "FMA"
elif ds_type.lower() in ["urbansound8k", "urbansound", "urban","u"]:
    ds_type = "UrbanSound8K"
elif ds_type.lower() in ["audioset", "audio", "a"]:
    ds_type = "Audioset"
else:
    raise ValueError(f"Invalid dataset type, {ds_type}")


classes = list(augmentations[ds_type].keys())

path_to_features = os.path.join(root, ds_type, args.model)
assert os.path.exists(path_to_features), f"Path does not exist: {path_to_features}, please generate features first."

print("Loading from:", path_to_features)  

audio_file = np.load(os.path.join(path_to_features, "audio_features.npz"), allow_pickle=True)

audio_features = audio_file.get("audio_features")
audio_labels = audio_file.get("audio_labels")


text_file = np.load(os.path.join(path_to_features, "text_features.npz"), allow_pickle=True)

raw_text_features = text_file.get("raw_features")
aug_text_features = text_file.get("augmented_features")

all_features = np.concatenate([audio_features, raw_text_features, aug_text_features], axis=0)

start = time()

if method == "tsne":
    cosine_dist = pairwise_distances(all_features, all_features, metric="cosine", n_jobs=-1) 

    tsne = TSNE(n_components=2, metric="precomputed", n_jobs=-1)

    tsne_results = tsne.fit_transform(cosine_dist)

    audio_features_tsne = tsne_results[:audio_features.shape[0]]
    raw_text_features_tsne = tsne_results[audio_features.shape[0]:audio_features.shape[0]+raw_text_features.shape[0]]
    aug_text_features_tsne = tsne_results[audio_features.shape[0]+raw_text_features.shape[0]:]

elif method == "pca":
    pca = PCA(n_components=2)

    pca_results = pca.fit_transform(all_features)

    audio_features_tsne = pca_results[:audio_features.shape[0]]
    raw_text_features_tsne = pca_results[audio_features.shape[0]:audio_features.shape[0]+raw_text_features.shape[0]]
    aug_text_features_tsne = pca_results[audio_features.shape[0]+raw_text_features.shape[0]:]

print(f"Time taken: {time() - start:.2f}s")

plt.figure(figsize=(16, 16))

colors = cm.rainbow(np.linspace(0, 1, len(classes)))

for i, txt in enumerate(classes):
    plt.scatter(audio_features_tsne[:, 0][audio_labels == i], 
                audio_features_tsne[:, 1][audio_labels == i],
                marker="o", color=colors[i], label=txt, alpha=0.7,) 

plt.scatter(raw_text_features_tsne[:, 0], raw_text_features_tsne[:, 1],
            marker="x", c="r", s=100, label="Raw Text Features",)

plt.scatter(aug_text_features_tsne[:, 0], 
            aug_text_features_tsne[:, 1],
            marker="*",c="b", s=100, label="Augmented Text Features", )

def get_norm(features, classes):
    # heuristic to get the norm of the features
    n = len(classes)

    return np.sqrt(np.sum(features**2, axis=1).mean() / n)
    

def u(norm):
    theta = np.random.uniform(0, 2*np.pi)
    return norm*np.cos(theta), norm*np.sin(theta)

fontsize = 8

for i, txt in enumerate(classes):
    x, y = u(get_norm(audio_features_tsne[audio_labels == i], classes))
    plt.annotate(txt, 
                 xy = (raw_text_features_tsne[i, 0], raw_text_features_tsne[i, 1]),
                 xytext = (raw_text_features_tsne[i, 0] + x, raw_text_features_tsne[i, 1] + y),
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                fontsize=fontsize)
    x, y = u(get_norm(audio_features_tsne[audio_labels == i], classes))
    plt.annotate(txt,
                    xy = (aug_text_features_tsne[i, 0], aug_text_features_tsne[i, 1]),
                    xytext = (aug_text_features_tsne[i, 0] + x, aug_text_features_tsne[i, 1] + y),
                    arrowprops=dict(facecolor='black', arrowstyle="->"),
                    fontsize=fontsize)
    

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(f"t-SNE visualization of {ds_type} features")
plt.tight_layout()
plt.savefig(f"figs/viz_{ds_type}_features_{args.model}_{method}.png", dpi=300)










