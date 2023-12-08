import numpy as np
import gradio as gr

def audio_retrieval(dataset, text):
    return image, audio1, audio2, audio3

demo = gr.Interface(
    audio_retrieval,
    [
        gr.Radio(["ESC-50", "UrbanSound8k", "FMA", "Audioset"], label="Dataset", info="Choose the dataset"),
        gr.Textbox(label="Text"),
    ],
    [gr.Image(label="t-snee"), gr.Audio(label="top1"), gr.Audio(label="top2"), gr.Audio(label="top3")]
    
)
if __name__ == "__main__":
    demo.launch()
