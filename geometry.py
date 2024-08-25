from PIL import Image, ImageDraw
import math
import io
import base64
import numpy as np
from fasthtml.common import *


def draw_arc(draw, R, P, angle_s, angle_e):
    if R > 0:
        box = [(P[0] - 2 * R, P[1] - R), (P[0], P[1] + R)]
    else:
        box = [(P[0], P[1] + R), (P[0] - 2 * R, P[1] - R)]

    draw.arc(box, angle_s, angle_e, (255, 0, 0))

def draw_lens(draw, R, T, P):
    angle = math.degrees(np.arccos((R - T / 2) / R))
    draw_arc(draw, R, (P[0] + T / 2, P[1]), -angle, angle)
    draw_arc(draw, - R, (P[0] - T / 2, P[1]), 180-angle, 180 + angle)

def generate_canvas():
    image = Image.new('RGB', (1024 + 512, 512), (225, 255, 255))
    draw = ImageDraw.Draw(image)

    draw_lens(draw, 1500, 10, (256, 256))
    draw_lens(draw, 1500, 10, (296, 256))

    my_stringIOBytes = io.BytesIO()
    image.save(my_stringIOBytes, format='JPEG')
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')
