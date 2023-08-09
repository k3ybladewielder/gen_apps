import os
import io
import IPython.display
from PIL import Image
import base64
from transformers import pipeline
import gradio as gr

import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']
    
gr.close_all()
demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )
demo.launch()
gr.close_all()
