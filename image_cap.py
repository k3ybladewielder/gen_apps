import os
import io
import IPython.display
from PIL import Image
import base64
from transformers import pipeline
import gradio as gr

import warnings
warnings.filterwarnings("ignore")

task = "image-to-text"
model = "Salesforce/blip-image-captioning-base"
image_captioner = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

def captioner(image):
    result = image_captioner(image)
    return result[0]['generated_text']

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                   )

demo.launch()
