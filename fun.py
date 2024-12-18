from PIL import Image, ImageDraw
import math
import io
import base64
from fasthtml.common import *

elements = [
    [{"t": "L", "par":[0.3, 0.075]}, {"t": "L", "par":[0.45, 0.075]}, {"t": "X", "par":[0.85]}, ],
    [{"t": "L", "par":[0.3, 0.075]}, {"t": "C", "par":[0.3735, 0.003]}, {"t": "L", "par":[0.45, 0.075]}, {"t": "X", "par":[0.85]}, ],
]

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

def Element(el, s, tab):
    match (el["t"]):
        case "X": # cavity length
            pos = el["par"][0]
            return Div(
                Span(f'X P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:50px;", value=f'{pos}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}/{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                style="border: 1px solid red; display: inline-block; padding:2px;"
            )
            
        case "L": # lens
            par = el["par"]              
            return Div(
                Span(f'L{s + 1} P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:70px;", value=f'{par[0]}'),
                Span("f:"),
                Input(type="number", id=f'el{s}focal', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}/{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 

                style="border: 1px solid red; display: inline-block; padding:2px;"
            )
        case "C": # crystal
            par = el["par"]              
            return Div(
                Span(f'C{s + 1} P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:70px;", value=f'{par[0]}'),
                Span("L:"),
                Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 

                style="border: 1px solid red; display: inline-block; padding:2px;"
            )

def funCanvas(idd, width=1000, height=800 ):
    return Div(
        Div(
            Div(
                Button(
                    Img(src="static/zoomin.png", alt="Zoom in", width="24", height="24"),
                    cls="imgButton",
                    onclick="zoomMultiMode(1);"
                ),
                Button(
                    Img(src="static/zoomout.png", alt="Zoom out", width="24", height="24"),
                    cls="imgButton",
                    onclick="zoomMultiMode(-1);"
                ),
            ),
            Canvas(id=f"funCanvas{idd}", width=width, height=height, **{'onmousemove':f"mainCanvasMouseMove(event, {idd});"}),
        )
    )   
def graphCanvas():
    return Div(
        Div(
            Select(Option("A"), 
                        Option("B"), 
                        Option("C"), 
                        Option("D"), 
                        Option("AbsE(x)"), 
                        Option("ArgE(x)"), 
                        Option("M(x)"), 
                        Option("Width(x)"),                               
                        Option("Waist(x)"),                               
                        Option("QWaist(x)"),                               
                        id="displayOption",
                        **{'onchange':"drawGraph();"},),
                                Canvas(id="graphCanvas", width=1000, height = 200,
            ),
            Canvas(id="graphCanvas", width=1000, height = 200,
                **{'onmousemove':"graphCanvasMouseMove(event);",
                'onmousedown':"graphCanvasMouseDown(event);",
                'onmouseup':"graphCanvasMouseUp(event);",},
            )
        )
    )    

def initBeamType(beamParamInit = 0.0005):
    return Div(
        Select(Option("Gaussian Beam"), Option("Two Slit"), Option("Mode He5"), 
            Option("Gaussian shift"), Option("Delta"), Option("Zero"),                               
            id="incomingFront"
        ),
        Input(type="number", id=f'beamParam', placeholder="beam", step="0.0001", style="width:80px;", value=f'{beamParamInit}'),
        style="display:inline-block;"
    )

def generate_fun(data_obj, tab, offset = 0):

    images = []
    added = Div()
    added2 = Div()
    match tab:
        case 1:
            added = Div(
                Div(initBeamType(), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiMode(1);"),
                    Button("Full", onclick="fullCavityMultiMode()"),
                    Button("Roundtrip", onclick="roundtripMultiMode()"),
                    Button("Delta graph", onclick="deltaGraphMultiMode()"),
                    Button("Switch view", onclick="switchViewMultiMode()"),
                    Input(type="number", id=f'initialRange', placeholder="range(m)", step="0.001", style="width:80px;", value=f'0.005'),
                    Button("Auto range", onclick="initElementsMultiMode(); autoRangeMultiMode();"),
                    Input(type="number", id=f'nSamplesPower', placeholder="power", step="1", style="width:80px;", value=f'8'),

                ),
                Div(
                    *[Element(el, i, tab) for i, el in enumerate(elements[tab - 1])],
                    Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/1", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                ),
                funCanvas(1),
                graphCanvas()
            )
        case 2:
            added = Div(
                Div(initBeamType(beamParamInit = 0.00003), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiMode(2);"),
                    Button("Full", onclick="fullCavityCrystal()"),
                    Button("Switch view", onclick="switchViewMultiMode()"),
                    Input(type="number", id=f'initialRange', placeholder="range(m)", step="0.0001", style="width:120px;", value=f'0.00034613292'),
                    Input(type="number", id=f'power', placeholder="power", step="1000000", style="width:120px;", value=f'10000000'),
                ),
                Div(
                    *[Element(el, i, tab) for i, el in enumerate(elements[tab - 1])],
                    Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/2", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                ),
                funCanvas(1, width=500, height=400),
                funCanvas(2, width=500, height=400),
                graphCanvas()
            )
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
        Div("Crystal", hx_post="tabfun/2", hx_target="#fun", cls=f"tab {'tabselected' if tab == 2 else ''}", hx_vals='js:{localId: getLocalId()}'),
        Div("Peak beam", hx_post="tabfun/3", hx_target="#fun", cls=f"tab {'tabselected' if tab == 3 else ''}", hx_vals='js:{localId: getLocalId()}')),

        added,
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


