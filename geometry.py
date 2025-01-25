from PIL import Image, ImageDraw
import math
import io
import base64
import numpy as np
from fasthtml.common import *
from cavity import CavityData

def generate_beam_params(sim, tab):
    return  Div(Form(
        Span("X"), Input(type="text", name="beam_x", value=str(sim.str_beam_x), 
            hx_post=f"/beamParams/{tab}", hx_vals='js:{localId: getLocalId()}', hx_trigger="keyup changed delay:0.1s",
            hx_target="#geometryCanvas", hx_swap="outerHTML",
            style=f"width:60px;{'background: #f7e2e2; color: red;' if sim.beam_x_error else ''}"),
        Span(NotStr("&#952;"), style="margin-left: 10px;"), Input(type="text", name="beam_theta", value=str(sim.str_beam_theta), 
            hx_post=f"/beamParams/{tab}", hx_vals='js:{localId: getLocalId()}', hx_trigger="keyup changed delay:0.1s",
            hx_target="#geometryCanvas", hx_swap="outerHTML",
            style=f"width:60px;{'background: #f7e2e2; color: red;' if sim.beam_theta_error else ''}"))
        , id="beamParams", style="padding:3px;"
    )

def geometry_tab(label, index, tab):
    cl = f"tab {'tabselected' if tab == index else ''}"
    return Div(label, hx_post=f"tabgeo/{index}", hx_target="#geometry", cls=cl, hx_vals='js:{localId: getLocalId()}')

def generate_geometry(data_obj, tab, offset = 0):

    images = []
    added = Div()
    if data_obj:
        sim = data_obj['cavityData']
        match tab:
            case 1:
                images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw)
                added = generate_beam_params(sim, tab)
            case 2:
                images.append(Image.new('RGB', (1024, 512 + 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw, aligned = True, keep_aspect = False)
                added = generate_beam_params(sim, tab)
            case 3:
                images.append(Image.new('RGB', (1024, 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                powerChart = sim.get_state()[0]
                mxArg = np.argmax(powerChart.y) + offset
                powerChart.draw(draw, markX = [mxArg])
                images.append(Image.new('RGB', (1024, 512), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                sim.draw_cavity(draw, aligned = True, keep_aspect = False)
                added = Div(
                        Button(">", hx_post=f"/moveonchart/{offset + 1}/3", hx_trigger="click, keyup[key=='ArrowRight'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                        Button("<", hx_post=f"/moveonchart/{offset - 1}/3", hx_trigger="click, keyup[key=='ArrowLeft'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                        style="padding:0px 3px;")
            case 4:
                images.append(Image.new('RGB', (1024, 256), (225, 255, 255)))
                draw = ImageDraw.Draw(images[- 1])
                powerChart = sim.get_state()[0]
                mxArg = np.argmax(powerChart.y) + offset
                powerChart.draw(draw, markX = [mxArg])
                table_data = sim.get_recorded_data()

                added = Div(
                            Div(
                                Button(">", hx_post=f"/moveonchart/{offset + 1}/4", hx_trigger="click, keyup[key=='ArrowRight'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                                Button("<", hx_post=f"/moveonchart/{offset - 1}/4", hx_trigger="click, keyup[key=='ArrowLeft'] from:body", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                                Button("Step", hx_post=f"/recordstep/{offset}", hx_trigger="click", hx_target="#geometry", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                                style="padding:0px 3px;"
                            ),
                            Table( 
                                Thead(Tr(*[Th(h, scope="col") for h in table_data[0]])),  
                                Tbody(*[Tr(*[Td(d) for d in r]) for r in table_data[1:]]),
                                border=1,
                            )
                        )

    my_base64_jpgData = []
    for image in images:
        my_stringIOBytes = io.BytesIO()
        image.save(my_stringIOBytes, format='JPEG')
        my_stringIOBytes.seek(0)
        my_base64_jpgData.append(base64.b64encode(my_stringIOBytes.read()))

    return Div(
        Div(
            geometry_tab("Real view", 1, tab),
            geometry_tab("Align & stretch", 2, tab),
            geometry_tab("Peak beam", 3, tab),
            geometry_tab("Test step", 4, tab),
        ),
        added,
        *[Div(Img(src=f'data:image/jpg;base64,{str(jpgData, "utf-8")}') for jpgData in my_base64_jpgData)], id="geometryCanvas"
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


