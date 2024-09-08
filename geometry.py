from PIL import Image, ImageDraw
import math
import io
import base64
import numpy as np
from fasthtml.common import *
from cavity import CavityData

# def draw_arc(draw, R, P, angle_s, angle_e):
#     if R > 0:
#         box = [(P[0] - 2 * R, P[1] - R), (P[0], P[1] + R)]
#     else:
#         box = [(P[0], P[1] + R), (P[0] - 2 * R, P[1] - R)]

#     draw.arc(box, angle_s, angle_e, (255, 0, 0))

# def draw_lens(draw, R, T, P):
#     angle = math.degrees(np.arccos((R - T / 2) / R))
#     draw_arc(draw, R, (P[0] + T / 2, P[1]), -angle, angle)
#     draw_arc(draw, - R, (P[0] - T / 2, P[1]), 180-angle, 180 + angle)

def generate_canvas(data_obj, tab, offset = 0):

    images = []
    if data_obj:
        sim = data_obj['cavityData']
        match tab:
            case 1:
                images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw)
            case 2:
                images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw, aligned = True, keep_aspect = False)
            case 3:
                images.append(Image.new('RGB', (1024, 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                powerChart = sim.get_state()[0]
                mxArg = np.argmax(powerChart.y) + offset
                powerChart.draw(draw, markX = [mxArg])
                images.append(Image.new('RGB', (1024, 512), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw, aligned = True, keep_aspect = False)


    my_base64_jpgData = []
    for image in images:
        my_stringIOBytes = io.BytesIO()
        image.save(my_stringIOBytes, format='JPEG')
        my_stringIOBytes.seek(0)
        my_base64_jpgData.append(base64.b64encode(my_stringIOBytes.read()))

    return Div(Div(Div("Real view",  hx_post="/tabgeo/1", hx_target="#geometry", cls=f"tab {'tabselected' if tab == 1 else ''}", hx_vals='js:{localId: getLocalId()}'),
        Div("Align & stretch", hx_post="tabgeo/2", hx_target="#geometry", cls=f"tab {'tabselected' if tab == 2 else ''}", hx_vals='js:{localId: getLocalId()}'),
        Div("Peak beam", hx_post="tabgeo/3", hx_target="#geometry", cls=f"tab {'tabselected' if tab == 3 else ''}", hx_vals='js:{localId: getLocalId()}')),
        Div(
            Button(">", hx_post=f"/moveonchart/{offset + 1}", hx_trigger="click, keyup[key=='ArrowRight'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
            Button("<", hx_post=f"/moveonchart/{offset - 1}", hx_trigger="click, keyup[key=='ArrowLeft'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
            ),
        *[Div(Img(src=f'data:image/jpg;base64,{str(jpgData, "utf-8")}') for jpgData in my_base64_jpgData)],
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


