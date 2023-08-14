from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline

import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize

"""
# Welcome to text to video!
"""

def call_model():
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16", device_map = 'auto')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

def main(pipe = call_model()):

    prompt_text = st.text_input('Prompt', 'Write Prompt Here')
    num_inference_steps = st.number_input('num_inference_steps: default 25')
    num_frames = st.number_input('num_frames: default 20')

    if st.button('Generate and play'):
        video_frames = pipe(prompt_text, num_inference_steps=int(num_inference_steps),num_frames=int(num_frames)).frames
        video_path = export_to_video(video_frames)
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    
    if st.button('Download'):
        st.download_button(label="Download Video", data=video_file, file_name= f'{prompt_text}_video.mp4')

if __name__ == "__main__":
    main(pipe = call_model())
