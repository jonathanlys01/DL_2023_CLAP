import argparse
import os
import librosa
from torch.utils.data import Dataset 
import pandas as pd

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

        print(f"Dataset loaded with {len(self.audios)} samples")
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        return {"text": self.annotations[idx], "audio": self.audios[idx]}
    
    def open_audio(self,path, sr=None):

        y, sr = librosa.load(path, sr=sr)
        return y, sr
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_audio", "-a", type=str, required=True)
    parser.add_argument("--path_to_annotation", "-t", type=str, required=True)

    args = parser.parse_args()

    dataset = ESC_50(args.path_to_audio, args.path_to_annotation)
    
    print(dataset.classes)
