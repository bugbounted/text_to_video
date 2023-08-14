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
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


with st.echo(code_location='below'):

    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt_text = st.text_input('Prompt', 'Write Prompt Here')
    num_inference_steps = st.number_input('num_inference_steps: default 25')
    num_frames = st.number_input('num_frames: default 20')

    if st.button('Generate'):
        video_frames = pipe(prompt_text, num_inference_steps=num_inference_steps,num_frames=num_frames).frames
        video_path = export_to_video(video_frames)

    if st.button('Play'):
        st.video(video_path)
    
    if st.button('Download'):
        st.download_button(label="Download Video", data=video_path, file_name= f'{prompt_text}_video.mp4')
