# Hugging Face Tutorial Reference
from transformers import pipeline
from PIL import Image
import os
import requests

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TO_SAVE = os.path.join(SCRIPT_DIR, '..', 'src', 'imgs')

    os.makedirs(TO_SAVE, exist_ok=True)

    # load pipe
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    # load image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image.save(os.path.join(TO_SAVE, "cat.png"))
    print(f"Image saved to {os.path.join(TO_SAVE, 'cat.png')}")

    # inference
    depth = pipe(image)["depth"]

    depth.save(os.path.join(TO_SAVE, "cat_relative_depth.png"))
    print(f"Depth map saved to {os.path.join(TO_SAVE, 'cat_relative_depth.png')}")  

    # # visualize depth map
    # depth.show()