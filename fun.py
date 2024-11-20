from PIL import Image, ImageDraw
import math
import io
import base64
import numpy as np
from fasthtml.common import *

def ver_func(l):
    vf = []
    z = l / 2
    for i in range(l):
        vf.append(math.exp(-(i - z) * (i - z) / 200))

    return vf

def draw_single_front(draw: ImageDraw, px, py, w, h, n, vec):
    for i in range(n):
        c = math.floor(vec[i] * 255)
        draw.rectangle([(px, py + h * i), (px + w, py + h * (i + 1))], fill = (c, c, c, 255))

def draw_multimode(draw: ImageDraw):
    draw_single_front(draw, 30, 30, 10, 2, 256, ver_func(256))
 
def generate_fun(data_obj, tab, offset = 0):

    images = []
    added = Div()
    match tab:
        case 1:
            images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
            draw = ImageDraw.Draw(images[- 1])
            draw_multimode(draw)
            added = Div(Select(Option("Gaussian Beam"), 
                               Option("Two Slit"), 
                               Option("Delta"), 
                               Option("Zero"),                               
                               id="incomingFront"),
                     Button("Init", onclick="initMultiMode()"),
                     Button("Propogate", onclick="propogateMultiMode()"),
                     Button("Lens", onclick="lensMultiMode()"),
                     Button("Switch view", onclick="switchViewMultiMode()"),
)
        # case 2:
        #     images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
        #     draw = ImageDraw.Draw(images[- 1])
        #     sim.draw_cavity(draw, aligned = True, keep_aspect = False)
        #     added = generate_beam_params(sim, tab)
        # case 3:
        #     images.append(Image.new('RGB', (1024, 256), (225, 255, 255)))
        #     draw = ImageDraw.Draw(images[- 1])
        #     powerChart = sim.get_state()[0]
        #     mxArg = np.argmax(powerChart.y) + offset
        #     powerChart.draw(draw, markX = [mxArg])
        #     images.append(Image.new('RGB', (1024, 512), (225, 255, 255)))
        #     draw = ImageDraw.Draw(images[- 1])
        #     sim.draw_cavity(draw, aligned = True, keep_aspect = False)
        #     added = Div(
        #             Button(">", hx_post=f"/moveonchart/{offset + 1}", hx_trigger="click, keyup[key=='ArrowRight'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
        #             Button("<", hx_post=f"/moveonchart/{offset - 1}", hx_trigger="click, keyup[key=='ArrowLeft'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
        #             style="padding:0px 3px;")


    my_base64_jpgData = []
    for image in images:
        my_stringIOBytes = io.BytesIO()
        image.save(my_stringIOBytes, format='JPEG')
        my_stringIOBytes.seek(0)
        my_base64_jpgData.append(base64.b64encode(my_stringIOBytes.read()))
    
    print(len(images))

    return Div(Div(Div("Multimode",  hx_post="/tabfun/1", hx_target="#fun", cls=f"tab {'tabselected' if tab == 1 else ''}", hx_vals='js:{localId: getLocalId()}'),
        Div("Align & stretch", hx_post="tabfun/2", hx_target="#fun", cls=f"tab {'tabselected' if tab == 2 else ''}", hx_vals='js:{localId: getLocalId()}'),
        Div("Peak beam", hx_post="tabfun/3", hx_target="#fun", cls=f"tab {'tabselected' if tab == 3 else ''}", hx_vals='js:{localId: getLocalId()}')),
        added,
        Canvas(id="funCanvas", width=1000, height = 800)
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


