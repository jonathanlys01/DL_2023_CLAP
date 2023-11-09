import time
import argparse
from dataset import ESC_50
import numpy as np
import matplotlib.pyplot as plt
import uuid


parser = argparse.ArgumentParser()

parser.add_argument("--path_to_audio", "-a", type=str, required=True)
parser.add_argument("--path_to_annotation", "-t", type=str, required=True)

args = parser.parse_args()

esc_50 = ESC_50(args.path_to_audio, args.path_to_annotation)

start = time.time()

from transformers import ClapModel, ClapProcessor


import_time = time.time()
print(f"Import time: {import_time - start:.2f}")

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")


#ind = np.random.randint(0, len(esc_50))
ind = 1
filename = esc_50.audios[ind]
label = esc_50.annotations[ind]


audio, sr = esc_50.open_audio(filename, sr=48000)

plt.plot(audio)
plot_name = filename.split("/")[-1].split(".")[0]
plt.title(f"Audio: {plot_name} / Label: {label}") 
plt.savefig(f"temp/audio_{uuid.uuid4()}.png")

texts = esc_50.classes

inputs_text = processor(text=texts, return_tensors="pt", padding=True)
print(inputs_text)

inputs_audio = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

outputs = model(**inputs_text, **inputs_audio)
end = time.time()
print(f"Total time: {end - start:.2f}")


