"""Model predict."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#

# https://github.com/avinassh/pytorch-flask-api

import base64
import io
import os
import pdb
import re
import torch
import torchvision.transforms as transforms
from flask import Flask, request
from PIL import Image

from model import enable_amp, get_model, model_load

# Global variables
app = Flask(__name__)

model = get_model()
model_load(model, "output/ImageClean.pth")
# CPU or GPU ?
device = torch.device(os.environ["DEVICE"])
model.to(device)
model.eval()
enable_amp(model)


def get_prediction(image_bytes):
    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    # image = Image.open(filename).convert("RGB")
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = totensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        noise_level_est, output_tensor = model(input_tensor)

    output_tensor.clamp_(0, 1.0)
    output = output_tensor.squeeze(0).cpu()

    del noise_level_est, input_tensor, output_tensor
    torch.cuda.empty_cache()

    # Image
    return toimage(output)


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    s64 = "data:image/png;base64," + \
        str(base64.b64encode(img_bytes), encoding="utf-8")
    return s64


def base64_to_image(s64):
    base64_data = re.sub('^data:image/.+;base64,', '', s64)
    img_bytes = base64.b64decode(base64_data)
    image_data = io.BytesIO(img_bytes)
    image = Image.open(image_data)
    return image


@app.route('/')
def hello_world():
    return 'Hello, Image Clean !'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        img_bytes = file.read()
        output = get_prediction(image_bytes=img_bytes)

        # Transform image to base64
        s64 = image_to_base64(output)

        # Test
        # image = base64_to_image(s64)
        # image.save("result.png")

        return s64


if __name__ == "__main__":
    """Deploy."""
    # curl -X POST -F file=@/tmp/test.jpg http://127.0.0.1:5000/predict
    # app.run(debug=True, host="0.0.0.0")
    app.run(host="0.0.0.0")
