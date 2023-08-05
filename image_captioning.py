import os
import io
import IPython.display
from PIL import Image
import base64
from transformers import pipeline

task = "image-to-text"
model = "Salesforce/blip-image-captioning-base"
image_captioner = pipeline("image-to-text", "Salesforce/blip-image-captioning-base")


#img_url = 'https://static.wikia.nocookie.net/onepiece/images/6/6d/Monkey_D._Luffy_Anime_Post_Timeskip_Infobox.png/revision/latest?cb=20190303115209&path-prefix=pt'
img_url = "https://static.wikia.nocookie.net/onepiece/images/a/af/Tony_Tony_Chopper_Anime_Post_Timeskip_Infobox.png/revision/latest?cb=20190113140816&path-prefix=pt"
display(IPython.display.Image(url=img_url))
image_captioner(img_url)   
    
    
