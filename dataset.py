import argparse
import os
import librosa
from torch.utils.data import Dataset 
import pandas as pd
import random as rd

def check_audio(path: str):
    if any([path.endswith(ext) for ext in ['.wav', '.mp3', '.flac']]):
        return True
    else:
        print(f"File {path} is not an audio file")
        return False
    

class ESC_50(Dataset):
    def __init__(self, path_to_audio, path_to_annotation):

      
        temp = pd.read_csv(path_to_annotation)


        audio_col = temp["filename"].values
        annotation_col = temp["category"].values

        assert all([audio in [
            path for path in os.listdir(path_to_audio) if check_audio(path)
            ] for audio in audio_col]), "Not all audios have annotations"

        self.audios = [os.path.join(path_to_audio, audio) for audio in audio_col]
        self.annotations = annotation_col

        self.classes = sorted(list(set(self.annotations)))

        assert len(self.audios) == len(self.annotations), "Number of audios and annotations do not match"

        indexes = [i for i in range(len(self.audios))]
        rd.shuffle(indexes)
        self.audios = [self.audios[i] for i in indexes]
        self.annotations = [self.annotations[i] for i in indexes]

        print(f"Dataset loaded with {len(self.audios)} samples (ESC-50)")
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        return {"text": self.annotations[idx], "audio": self.audios[idx]}
    
    def open_audio(self,path, sr=None):

        y, sr = librosa.load(path, sr=sr)
        return y, sr
    

class FMA(Dataset):
    def __init__(self, path_to_audio, path_to_annotation):
        # nested paths
        list_folders = [os.path.join(path_to_audio, f)\
            for f in os.listdir(path_to_audio)\
            if os.path.isdir(os.path.join(path_to_audio, f))]
        # [path_to_audio/000, path_to_audio/001, ...]
        # there's a README.md file in the root of the dataset, so we need to remove it
        
        audios = []
        for folder in list_folders:
            audios += [os.path.join(folder, audio) for audio in os.listdir(folder)]
        # [path_to_audio/000/000002.mp3, path_to_audio/000/000005.mp3, ...]
        
        self.audios = [audio for audio in audios if check_audio(audio)]
        rd.shuffle(self.audios) # in-place shuffle
        
        df_fma = pd.read_csv(path_to_annotation, header=[0, 1], index_col=0)
        indexes_small = df_fma.index[df_fma["set", "subset"] == "small"]
        df_fma_small = df_fma.loc[indexes_small]
        del df_fma
        annotations = {index:genre for index, genre in zip(df_fma_small.index, 
                                                           df_fma_small["track", "genre_top"])}
        # {track_id: genre, ...}
        
        temp_audio_ids = [int(audio.split("/")[-1].split(".")[0]) for audio in self.audios] # [track_id, ...]
        self.annotations = [annotations[audio_id] for audio_id in temp_audio_ids] 
        # reordering annotations to match audios
        del annotations
        del df_fma_small
        
        self.classes = sorted(list(set(self.annotations)))
        
        assert len(self.audios) == len(self.annotations), "Number of audios and annotations do not match"
        
        print(f"Dataset loaded with {len(self.audios)} samples (FMA)")
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        return {"text": self.annotations[idx], "audio": self.audios[idx]}
    
    def open_audio(self,path, sr=None):
        y, sr = librosa.load(path, sr=sr)
        return y, sr


class UrbanSound8k(Dataset):
    def __init__(self, path_to_audio, path_to_annotation):
        df = pd.read_csv(path_to_annotation)
        
        self.audios = [os.path.join((path_to_audio), 
                       f"fold{x['fold']}", 
                       x["slice_file_name"]) for _, x in df.iterrows()]
        assert all([os.path.exists(audio) for audio in self.audios]), "Not all audios exist"
        
        self.audios = [audio for audio in self.audios if check_audio(audio)]
        
        self.annotations = df["class"].values
        
        assert len(self.audios) == len(self.annotations), "Number of audios and annotations do not match"
        
        indexes = [i for i in range(len(self.audios))]
        rd.shuffle(indexes)
        self.audios = [self.audios[i] for i in indexes]
        self.annotations = [self.annotations[i] for i in indexes]
        
        self.classes = sorted(list(set(self.annotations)))
        print(f"Dataset loaded with {len(self.audios)} samples (UrbanSound8K)")
    def __len__(self):
        return len(self.audios)
    def __getitem__(self, idx):
        return {"text": self.annotations[idx], "audio": self.audios[idx]}
    
    def open_audio(self,path, sr=None):
        y, sr = librosa.load(path, sr=sr)
        return y, sr


class Audioset(Dataset):
    def __init__(self, path):
        audios = [os.path.join(path ,os.path.join(class_folder, file)) for class_folder in  os.listdir(path) for file in os.listdir(os.path.join(path, class_folder))]
        annotations = list(map(
            lambda x : os.path.basename(os.path.dirname(x)), 
            audios
        ))

        indices = [i for i in range(len(audios))]
        rd.shuffle(indices)

        self.audios = [audios[i] for i in indices]
        self.annotations = [annotations[i] for i in indices]

        self.classes = sorted(os.listdir(path))

    def __len__(self):
        return len(self.audios)
    def __getitem__(self, idx):
        return {"text": self.annotations[idx], "audio": self.audios[idx]}
    
    def open_audio(self,path, sr=None):
        y, sr = librosa.load(path, sr=sr)
        return y, sr
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help = "Dataset type, one of ESC-50, FMA, UrbanSound8K (short names: e, f, u)"
    parser.add_argument("--dataset_type", "-d", type=str, required=True, help=help)
    root = "downloads/"
    
    args = parser.parse_args()
    ds_type = args.dataset_type
    
    if ds_type.lower() in ["esc-50", "esc50", "esc","e"]:
        ds_type = "ESC-50"
    elif ds_type.lower() in ["fma", "freemusicarchive","f"]:
        ds_type = "FMA"
    elif ds_type.lower() in ["urbansound8k", "urbansound", "urban","u"]:
        ds_type = "UrbanSound8K"
    elif ds_type.lower() in ["audioset", "audio", "a"]:
        ds_type = "AudioSet"
    
    if ds_type == "ESC-50":
        print("Loading ESC-50 dataset")
        path_to_audio = os.path.join(root, "ESC-50-master/audio/")
        path_to_annotation = os.path.join(root, "ESC-50-master/meta/esc50.csv")
        dataset = ESC_50(path_to_audio, path_to_annotation)
        print("classes",dataset.classes)
        
    elif ds_type == "FMA":
        print("Loading FMA dataset")
        path_to_audio = os.path.join(root, "fma_small/")
        path_to_annotation = os.path.join(root, "fma_metadata/tracks.csv")
        dataset = FMA(path_to_audio, path_to_annotation)
        print("classes",dataset.classes)
    
    elif ds_type == "UrbanSound8K":  
        print("Loading UrbanSound8K dataset")
        path_to_audio = os.path.join(root, "UrbanSound8K/audio/")
        path_to_annotation = os.path.join(root, "UrbanSound8K/metadata/UrbanSound8K.csv")
        dataset = UrbanSound8k(path_to_audio, path_to_annotation)
        print("classes", dataset.classes)

    elif ds_type == "AudioSet":
        print("Loading Audioset dataset")
        path = os.path.join(root, "audioset")
        dataset = Audioset(path)
        print("classes", dataset.classes)

        print("samples", dataset.audios[:3], dataset.annotations[:3])
    

