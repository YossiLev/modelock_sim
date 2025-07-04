from PIL import Image, ImageDraw
import math
import io
import base64
from fasthtml.common import *
from controls import *

elements = [
    #[{"t": "L", "par":[0.3, 0.075]}, {"t": "L", "par":[0.45, 0.075, 1.0], "del": 1.0}, {"t": "X", "par":[0.85]}, ],
    [{"t": "L", "par":[0.07, 0.05]}, {"t": "L", "par":[0.392, 0.008, 1.0], "del": 1.0}, 
     {"t": "L", "par":[0.408, 0.008, 1.0], "del": 1.0}, {"t": "L", "par":[0.73, 0.05]}, {"t": "X", "par":[0.80]}], 
    [{"t": "L", "par":[0.9, 0.075]}, {"t": "C", "par":[0.9735, 0.003]}, {"t": "L", "par":[1.05, 0.075, 1.0]}, {"t": "X", "par":[1.55]}, ],
    [{"t": "L", "par":[0.9, 0.075]}, {"t": "L", "par":[0.982318181, 10.000, 0.5], "del": 0.5}, 
        {"t": "L", "par":[1.057818181, 0.075, 1.0], "del": 1.0}, {"t": "X", "par":[1.557818181]}, ],   
]

current_cavity_name = "Yehuda"
cavities = [
    { "name": "Empty", "elements": []},
    { "name": "Eden", "elements": 
        [ 
            "S",
            "P 22.12cm",
            "L 7.5cm",
            "P 9.51cm",
            "L 0.8cm",
            "P 0.9cm",
            ">D ",
            "P 0.83cm",
            "L 0.8cm",
            "P 10.0cm",
            "L 7.5cm",
            "P 21.5cm",
            "E",
        ]
    },
    { "name": "Yehuda", "elements": 
        [
            "S",
            "P 7cm",
            "L 5cm",
            "P 33cm",
            "L 0.8cm",
            "P 0.83cm",
            ">D ",
            "P 0.83cm",
            "L 0.8cm",
            "P 23.4cm",
            "L 5cm",
            "P 7cm",
            "E"
        ]
    },
    { "name": "Mallachi 1", "elements": 
        [
            "S",
            "P 100mm",
            "L 50mm",
            "P 107.5mm",
            "L 7.5mm",
            "P 7.5mm",
            "D 4mm",
            "P 4mm",
            "P 7.5mm",
            "L 7.5mm",
            "P 87.5mm",
            "L 80mm ",
            "P 240mm",
            "E",
        ]
    },
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
    par = el["par"]         
    delta = el["del"] if "del" in el else 0.0
    match (el["t"]):
        case "X": # cavity length
            pos = par[0]
            return Div(
                Span(f'X P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:50px;", value=f'{pos}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}/{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'),
                Div(NotStr("Delta:"), Input(type="number", id=f'el{s}delta', placeholder="0", style="width:50px;", value='0.0')),
                style="border: 1px solid red; display: inline-block; padding:2px;"
            )
            
        case "L": # lens
            return Div(
                Span(f'L{s + 1} P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:70px;", value=f'{par[0]}'),
                Span("f:"),
                Input(type="number", id=f'el{s}focal', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}/{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                Div(NotStr("Delta factor:"), Input(type="number", id=f'el{s}delta', placeholder="0", style="width:50px;", value=f'{delta}')),
                style="border: 1px solid red; display: inline-block; padding:2px;"
            )
        case "C": # crystal
            return Div(
                Span(f'C{s + 1} P:', id=f'type{s}'),
                Input(type="number", id=f'el{s}dist', placeholder="0", step="0.01", style="width:70px;", value=f'{par[0]}'),
                Span("L:"),
                Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                Button(NotStr("X"), escapse=False, hx_post=f'/removeElements/{tab}{s}', hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                Div(NotStr("Delta factor:"), Input(type="number", id=f'el{s}delta', placeholder="0", style="width:50px;", value=f'{delta}')),
                style="border: 1px solid red; display: inline-block; padding:2px;"
            )

def funCanvas(idd, width=1000, height=800, useZoom = True, 
    style="background-color: #f5f5f9; background-image: radial-gradient(circle at center center, #dcf68e, #f5f5f9), repeating-radial-gradient(circle at center center, #dcf68e, #dcf68e, 10px, transparent 20px, transparent 10px);background-blend-mode: multiply;",
 ):
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
                Button(
                    Img(src="static/ziw.png", alt="Zoom out", title="Stretch horizontally", width="24", height="24"),
                    cls="imgButton",
                    onclick="zoomMultiMode(2);"
                ),
                Button(
                    Img(src="static/zow.png", alt="Zoom out", title="Reduce horizontal stretch", width="24", height="24"),
                    cls="imgButton",
                    onclick="zoomMultiMode(-2);"
                ),
                Button(
                    Img(src="static/z0w.png", alt="Zoom out", title="Reset horizontal stretch", width="24", height="24"),
                    cls="imgButton",
                    onclick="zoomMultiMode(20);"
                ),
            ) if useZoom else Div(),
            Canvas(id=f"funCanvas{idd}", width=width, height=height, style=style,
                   **{'onmousemove':f"mainCanvasMouseMove(event);",
                      'onmousedown':f"mainCanvasMouseDown(event);",
                      'onmouseup':f"mainCanvasMouseUp(event);",
                      }),
                      

        )
    )   
def graphCanvas(id="graphCanvas", width=1000, height = 200, options=True):
    return Div(
        Div(
            Div(Select(Option("A"), Option("B"), Option("C"), Option("D"), 
                        Option("AbsE(x)"), Option("ArgE(x)"), Option("M(x)"), 
                        Option("Width(x)"), Option("Waist(x)"), Option("QWaist(x)"),
                        id="displayOption",
                        **{'onchange':"drawGraph();"},),
            Label(Input(id="cbxPrevCompare", type='checkbox', name='Compare', checked=False), "Compare"))
            if options else "",

            Div(
                Button(
                    Img(src="static/zoomin.png", alt="Zoom in", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"zoomGraph('{id}', 1);"
                ),
                Button(
                    Img(src="static/zoomout.png", alt="Zoom out", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"zoomGraph('{id}', -1);"
                ),
                Button(
                    Img(src="static/c12.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"selectGraph('{id}');"
                ),
                Button(
                    Img(src="static/degaussian.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"degaussGraph('{id}');"
                ),
                Button(
                    Img(src="static/delorentzian.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"delorentzGraph('{id}');"
                ),
                Button(
                    Img(src="static/desech.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"desechGraph('{id}');"
                ),                
                Span("1.0", id=f"{id}-zoomVal", style="display: none;"),
                Span("0", id=f"{id}-selectVal", style="display: none;"),
                Span("0", id=f"{id}-degaussVal", style="display: none;"),
                Span("0", id=f"{id}-delorentzVal", style="display: none;"),
                Span("0", id=f"{id}-desechVal", style="display: none;"),
                style="position: absolute; float: left; z-index: 10;"
            ),
            Div(Span("", id=f"{id}-message", style=""),
                style=f"position: absolute; z-index: 5; width: {width}px; text-align: right; font-size:10px"
            ),
            Canvas(
                id=id, width=width, height = height, 
                   style="background-color: #e5e5f7; background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #444cf755, #444cf7 );",
                **{'onmousemove':"graphCanvasMouseMove(event);",
                'onmousedown':"graphCanvasMouseDown(event);",
                'onmouseup':"graphCanvasMouseUp(event);",},
                
            )
        )
    )    

def initBeamType(beamParamInit = 0.0005, beamDistInit = 0.0):
    return Div(
        Select(Option("Gaussian Beam"), Option("Gaussian Noise"), Option("Two Slit"), Option("Mode He5"), 
            Option("Gaussian shift"), Option("Delta"), Option("Zero"),                               
            id="incomingFront"
        ),
        Input(type="number", id=f'beamParam', placeholder="beam", step="0.0001", style="width:90px;", value=f'{beamParamInit}'),
        Input(type="number", id=f'beamDist', placeholder="dist", step="0.0001", style="width:90px;", value=f'{beamDistInit}'),
        style="display:inline-block;"
    )

def collectData(data_obj, delay=0, more=False):
    if data_obj is None:
        return Div("mmData")
    data_obj.assure("mmData")

    mmData = data_obj.mmData
    mmDataSer = mmData.serialize_mm_data(delay, more)
    return Div(mmDataSer, style="height:1px; overflow:hidden;")

def ViewButton(label, title, part, highlight):
    titles = ["Before entering the crystal", 
              "After Kerr non-linear phase shift (lensing)", 
              "After power preserving soft aperture", 
              "After frequency gain and dispaersion", 
              "After active gain modulation",
              "After gain saturaion and diffreaction",
              "After linear propoagation in cavity arm"]
    titles_side =[" propogating to right", " propogating to left"]
    return Button(label, id=f"view_button-{part}-{label}", title=title, 
                  hx_post=f"/mmView/{part}/{label}", hx_vals='js:{localId: getLocalId()}', 
                  hx_swap="innerHTML", hx_target="#numData", cls=("buttonH" if highlight else ""))

def num_but_title(label):
    titles = ["Before entering the crystal", 
              "After Kerr non-linear phase shift (lensing)", 
              "After power preserving soft aperture", 
              "After frequency gain and dispersion", 
              "After active gain modulation",
              "After gain saturaion and diffreaction",
              "After linear propoagation in cavity arm"]
    titles_side =[" (propogating to right)", " (propogating to left)"]
    return titles[(label - 1) % 7] + titles_side[(label - 1) // 7]

def ViewButtons(data_obj, part):
    if (data_obj is not None):
        mmData = data_obj.mmData
    return Div(
        *[ViewButton(f"{x}", num_but_title(x), part, data_obj and mmData.view_on_stage[part] == f"{x}") for x in range(1, 15)],
        Div("", style="width:20px; display:inline-block;"),
        ViewButton("Frq", "Frequency analysis of the pixel along time", part, data_obj and mmData.view_on_amp_freq[part] == "Frq"),
        ViewButton("Amp", "electric field at this pixel", part, data_obj and mmData.view_on_amp_freq[part] == "Amp"),
        Div("", style="width:20px; display:inline-block;"),
        ViewButton("Phs", "Phase of the complex data", part, data_obj and mmData.view_on_abs_phase[part] == "Phs"),
        ViewButton("Abs", "Absolute value of the complex data", part, data_obj and mmData.view_on_abs_phase[part] == "Abs"),
        ViewButton("Pow", "Power value of the complex data", part, data_obj and mmData.view_on_abs_phase[part] == "Pow"),
        style="display:inline-block;"
    )

def multimode_charts(data_obj):
    if data_obj is None:
        print("No data")
        return Div()
    data_obj.assure("mmData")
    mmData = data_obj.mmData
    return Div(
        Div(
            Div(
                Div(cls="handle", draggable="true"),
                ViewButtons(data_obj, 0),
                Div(
                    funCanvas("Sample1", width=1024, height=mmData.n_samples, useZoom=False, style="position: absolute; top: 0; left: 0;"),
                    funCanvas("Sample1top", width=1024, height=mmData.n_samples, useZoom=False, style="z-index:10; position: absolute; top: 0; left: 0;"),
                    style="position: relative; width: 1024px; height: 256px;"
                ), cls="container"
            ),
            Div(
                Div(cls="handle", draggable="true"),
                ViewButtons(data_obj, 1),
                Div(
                    funCanvas("Sample2", width=1024, height=mmData.n_samples, useZoom=False, style="position: absolute; top: 0; left: 0;"),
                    funCanvas("Sample2top", width=1024, height=mmData.n_samples, useZoom=False, style="z-index:10; position: absolute; top: 0; left: 0;"),
                    style="position: relative; width: 1024px; height: 256px;"
                ), cls="container"
            ),
            Div(
                Div(cls="handle", draggable="true"),
                    FlexN([graphCanvas(id="gr1", width=mmData.n_samples, height = 200, options=False), 
                            graphCanvas(id="gr2", width=mmData.n_samples, height = 200, options=False),
                            graphCanvas(id="gr3", width=mmData.n_samples, height = 200, options=False),
                            graphCanvas(id="gr4", width=mmData.n_samples, height = 200, options=False),
                    ]), cls="container"
            ),
            Div(
                Div(cls="handle", draggable="true"),
                    graphCanvas(id="gr5", width=1024, height = 200, options=False), cls="container"
            ),
            Div(
                Div(cls="handle", draggable="true"),
                    Div(id="plotData1", style="height:500px; width:1000px"), cls="container"
            ),
            id="containerList"
        ),
        # Button("Progress left", onclick="progressMultiTime(1)"),
        # Button("Progress right", onclick="progressMultiTime(2)"),
        # funCanvas("Test", width=1400, height=mmData.n_samples, useZoom=True),
        # graphCanvas(width=1400),
        Div(collectData(data_obj, 500), id="numData"),
    )

def InputS(id, title, value, step=0.01, width = 50):
    return Input(type="number", id=id, title=title, 
                 value=value, step=f"{step}", 
                 hx_trigger="input changed delay:1s", hx_post="/mmUpdate", hx_include="#multiTimeOptionsForm *", 
                 hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;"),

def generate_multi_on_server(data_obj):
    if data_obj is None:
        params = {
            "beamType": 0,
            "initialRange": 0.001, #0.00024475293,
            "seed": 0,
            "aperture": 0.000056,
            "epsilon": 1.8,
            "gainFactor": 0.5,
            "dispersionFactor": 0.45,
            "lensingFactor": 1.0,
            "modulationGainFactor": 0.0,
            "isFactor": 15000,
            "crystalShift": 0.0001,
            "nRounds": 1

        }
    else:
        data_obj.assure("mmData")
        mmData = data_obj.mmData
        params = {
            "beamType": mmData.beam_type,
            "initialRange": mmData.initial_range,
            "seed": mmData.seed,
            "aperture": mmData.aperture,
            "diffractionWaist": mmData.diffraction_waist,
            "epsilon": mmData.epsilon,
            "gainFactor": mmData.gain_factor,
            "dispersionFactor": mmData.dispersion_factor,
            "lensingFactor": mmData.lensing_factor,
            "modulationGainFactor": mmData.modulation_gain_factor,
            "isFactor": mmData.is_factor,
            "crystalShift": mmData.crystal_shift,
            "reportEveryStep": mmData.report_every_step,
            "nRounds": mmData.n_rounds_per_full,
        }
    n_rounds_options = [1, 10, 100, 300, 1000, 2000, 5000, 10000]
    selected_rounds_option = params["nRounds"]

    return Div(
        Div(
            initBeamType(beamParamInit = 0.00003, beamDistInit = 0.0), 
            Button("Init", hx_post=f"/mmInit", hx_include="#nRounds, #multiTimeOptionsForm *", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML", hx_target="#multiModeServer"),
            Button("Full", hx_ext="ws", ws_connect="/mmRun", ws_send=True, hx_include="#nRounds, #multiTimeOptionsForm", hx_vals='js:{localId: getLocalId()}'),
            Select(*[Option(x, selected="1") if (x == selected_rounds_option) else Option(x) for x in n_rounds_options], id="nRounds", **{'onchange':"nMaxMatricesChanged();"},),
            Button("Update", hx_put=f"/mmUpdate", hx_include="#multiTimeOptionsForm *", hx_vals='js:{localId: getLocalId()}'),
            Button("Stop", hx_put=f"/mmStop", hx_include="#multiTimeOptionsForm *", hx_vals='js:{localId: getLocalId()}', hx_swap="none"),
            Button("Clear 3D", onclick="ClearPlot3D();"),
            Button("Add 3D", onclick="AddPlot3D();"),
            Label(Input(id="cbxAutoRecord", type='checkbox', checked=False), "Auto 3D"),
            Button("Center", hx_post=f"/mmCenter", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML", hx_target="#numData"),

        ),
        Div(
            Div(
                Select(Option("Radial", selected="1") if params["beamType"] == 1 else Option("Radial"),
                        Option("1-Dimensional", selected="1") if params["beamType"] == 0 else Option("1-Dimensional"), id="beamType",),
                InputS('initialRange', "The range of the wave front (meters)", f'{params["initialRange"]}', step=0.0001, width = 60),
                InputS('seed', "Random seed", f'{params["seed"]}', step="", width = 50),
                InputS('aperture', "Width of a Gaussian aperture (meters)", f'{params["aperture"]}', step=0.00001, width = 60),
                InputS('diffractionWaist', "Width of a diffraction (meters)", f'{params["diffractionWaist"]}', step=0.00001, width = 60),
                InputS('epsilon', "Gain Epsilon", f'{params["epsilon"]}', step=0.01, width = 50),
                InputS("gainFactor", "Gain factor", f'{params["gainFactor"]}', step=0.01),
                InputS("dispersionFactor", "Dispersion factor", f'{params["dispersionFactor"]}', step=0.01),
                InputS("lensingFactor", "Kerr lensing factor", f'{params["lensingFactor"]}', step=0.01, width = 40),
                InputS("modulationGainFactor", "Modulation gain factor", f'{params["modulationGainFactor"]}', step=0.01, width = 40),   
                InputS("isFactor", "Intensity saturation factor", f'{params["isFactor"]}', step=0.01, width = 60),
                InputS("crystalShift", "Crystal position shift (mm)", f'{params["crystalShift"]}', step=0.00001, width = 60),
                InputS("reportEveryStep", "Report poeriod", f'{params["reportEveryStep"]}', step=0.00001, width = 50),
                #Select(Option("256"), Option("512"), Option("1024"), Option("2048"), Option("4096"), id="nSamples",),
                Input(type="text", id=f'stepsCounter', title="Number of roundtrips made", hx_trigger="input changed delay:1s", hx_post="/mmUpdate", hx_include="#multiTimeOptionsForm *", 
                    style="width:60px; text-align: right", value=f'0', **{'readonly':"readonly"},),
                id="multiTimeOptionsForm"
            ),
            # Button(">", onclick="shiftFronts(- 5);"),
            # Button("<", onclick="shiftFronts(5);"),
            # Button(">>", onclick="shiftFronts(- 50);"),
            # Button("<<", onclick="shiftFronts(50);"),
        ),
        Div(
            Button("Save Beam", hx_post=f"/mmSaveState", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML", hx_target="#numData"),
            Button("Restore Beam", hx_post=f"/mmRestoreState", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML", hx_target="#numData"),
            Button("Save Parameters", onclick="saveMultiTimeParametersProcess()"),
            Button("Restore Parameters", onclick="restoreMultiTimeParametersProcess()"),
            Div(Div("Give a name to saved parameters"),
                Div(Input(type="text", id=f'parametersName', placeholder="Descibe", style="width:450px;", value="")),
                Button("Save", onclick="saveMultiTimeParameters(1)"),
                Button("Cancel", onclick="saveMultiTimeParameters(0)"),
                id="saveParametersDialog", cls="pophelp", style="position: absolute; visibility: hidden"),
            Div(Div("Select the parameters set"),
                Div("", id="restoreParametersList"),
                Div("", id="copyParametersList"),
                Button("Cancel", onclick="restoreMultiTimeParameters(-1)"),
                Button("Export", onclick="exportMultiTimeParameters()"),
                Button("Import", onclick="importMultiTimeParameters()"),
                id="restoreParametersDialog", cls="pophelp", style="position: absolute; visibility: hidden"),
        ),
        # Div(
        #     *[Element(el, i, 5) for i, el in enumerate(elements[2])],
        #     Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/2", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
        # ),
        multimode_charts(data_obj),
        
        id="multiModeServer",
    )
def collect_cavity(name):
    cav = [c for c in cavities if c["name"] == name]
    if len(cav) == 0:
        return ""
    return "\n".join(cav[0]["elements"])

def generate_multimode(data_obj, tab):

    images = []
    added = Div()
    match tab:
        case 1:
            added = Div(
                Header("See light lateral shape as a 1D front progressing in the cavity", help="This is a help text"),
                Div(
                    initBeamType(), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiMode(1);"),
                    Select(Option("All"), Option("1"), Option("2"), Option("3"), Option("4"), Option("5"), id="nMaxMatrices", **{'onchange':"nMaxMatricesChanged();"},),
                    Button("Full", onclick="initElementsMultiMode(); initMultiMode(1); fullCavityMultiMode()"),
                    Button("Full Range", onclick="initElementsMultiMode(); initMultiMode(1); fullCavityMultiMode(mode=2)"),
                    Button("Gaussian", onclick="initElementsMultiMode(); fullCavityGaussian()"),
                    Button("Roundtrip", onclick="initElementsMultiMode(); initMultiMode(1); roundtripMultiMode()"),
                    Button("Delta graph", onclick="initElementsMultiMode(); initMultiMode(1);deltaGraphMultiMode()"),
                    Button("Switch view", onclick="switchViewMultiMode()"),
                    Input(type="number", id=f'initialRange', placeholder="range(m)", step="0.001", style="width:80px;", value=f'0.005'),
                    Button("Auto range", onclick="initElementsMultiMode(); autoRangeMultiMode();"),
                    Select(Option("256"), Option("512"), Option("1024"), Option("2048"), Option("4096"), id="nSamples", **{'onchange':"nSamplesChanged();"},),
                ),
                FlexN([
                    PickerDivs("pickEl", 15, 0, style="width: 13px; height: 13px; margin:2px; cursor: pointer;"),
                    Textarea(collect_cavity(data_obj.current_cavity_name), id="pickEl_text", spellcheck="false", style="width:200px; height: 300px; border: none; font-size: 13px; font-family: Arial;"),
                    Div(
                        Div(Input(type="number", id=f'pickEl_edit_inc', step="0.001", style="width:100px; height: 20px;", value=f'0.001')),
                        *[Div(Button(x["name"], hx_post=f"/setcavity/1/{x['name']}", hx_target="#genMultiMode", hx_vals='js:{localId: getLocalId()}')) for x in cavities],)
                ]),
                funCanvas(1),
                graphCanvas()
            )
        case 2: # Crystal
            added = Div(
                Div(initBeamType(beamParamInit = 0.00003, beamDistInit = 0.0), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiMode(2);"),
                    Button("Full", onclick="fullCavityCrystal()"),
                    Button("Full(prev)", onclick="fullCavityCrystal(2)"),
                    Button("Switch view", onclick="switchViewMultiMode()"),
                    Input(type="number", id=f'initialRange', placeholder="range(m)", step="0.0001", style="width:100px;", value=f'0.00024475293'),
                    Input(type="number", id=f'power', placeholder="power", step="1000000", style="width:80px;", value=f'30000000'),
                    Input(type="number", id=f'apreture', placeholder="apreture", step="0.00001", style="width:70px;", value=f'0.000056', **{'onchange':"apertureChanged();"},),
                    Select(Option("256"), Option("512"), Option("1024"), Option("2048"), Option("4096"), id="nSamples", **{'onchange':"nSamplesChanged();"},),
                    Select(Option("5"), Option("4"), Option("3"), Option("2"), Option("1"), Option("0"), id="nLenses", **{'onchange':"nLensesChanged();"},),
                ),
                Div(
                    *[Element(el, i, tab) for i, el in enumerate(elements[tab - 1])],
                    Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/2", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                ),
                FlexN([funCanvas(1, width=500, height=400), funCanvas(2, width=500, height=400)]),

                graphCanvas()
            )
        case 3: # MultiTime
            added = Div(
                Div(initBeamType(beamParamInit = 0.00003, beamDistInit = 0.0), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiTime();", ),
                    Button("Phase", onclick="timeCavityStep(1, true)"),
                    Button("Full", onclick="timeCavityStep(5, false)"),
                    Select(Option("0"), Option("1"), Option("2"), Option("3"), Option("4"), id="nRounds", **{'onchange':"nRoundsChanged();"}),
                    Div(collectData(data_obj), id="numData"),
                    Button("New Cavity", onclick="refreshCavityMatrices()"),
                    Button("CavSim Orig", onclick="calcOriginalSimMatrices()", title="Initialize the linear cavity as the original simulation for single mode"),
                    Button("This Cavity", onclick="calcCurrentCavityMatrices()", title="Initialize the linear cavity according to the stucture define below"),

                ),
                Div(
                    Input(type="number", id=f'initialRange', title="The range of the wave front (meters)", step="0.0001", style="width:100px;", value=f'0.00024475293'),
                    #Input(type="number", id=f'power', placeholder="power", step="1000000", style="width:80px;", value=f'30000000'),
                    Input(type="number", id=f'aperture', title="Width of a Gaussian aperture (meters)", 
                           style="width:80px;", value=f'0.000056', **{'onchange':"multiTimeApertureChanged();"},),
                    Input(type="number", id=f'epsilon', title="Gain Epsilon", 
                           style="width:50px;", value=f'0.2', **{'onchange':"epsilonChanged();"},),
                    Input(type="number", id=f'gainFactor', title="Gain factor", 
                           style="width:50px;", value=f'0.5', **{'onchange':"gainFactorChanged();"},),
                    Input(type="number", id=f'dispersionFactor', title="Dispersion factor", 
                           style="width:50px;", value=f'1.0', **{'onchange':"dispersionFactorChanged();"},),
                    Input(type="number", id=f'lensingFactor', title="Kerr lensing factor", 
                           style="width:50px;", value=f'1.0', **{'onchange':"lensingFactorChanged();"},),
                    Input(type="number", id=f'modulationGainFactor', title="Modulation gain factor", 
                           style="width:50px;", value=f'0.1', **{'onchange':"modulationGainFactorChanged();"},),
                    Input(type="number", id=f'isFactor', title="Intensity saturation factor", 
                           style="width:120px;", value=f'{200 * 352000}', **{'onchange':"isFactorChanged();"},),
                    Select(Option("256"), Option("512"), Option("1024"), Option("2048"), Option("4096"), id="nSamples", **{'onchange':"nSamplesChanged();"},),
                    Input(type="text", id=f'stepsCounter', title="Number of roundtrips made", 
                           style="width:60px; text-align: right", value=f'0', **{'readonly':"readonly"},),
                    Button(">", onclick="shiftFronts(- 5);"),
                    Button("<", onclick="shiftFronts(5);"),
                    Button(">>", onclick="shiftFronts(- 50);"),
                    Button("<<", onclick="shiftFronts(50);"),
                ),
                Div(
                    *[Element(el, i, tab) for i, el in enumerate(elements[2])],
                    Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/2", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                ),
                FlexN([funCanvas("Time", width=1024, height=256, useZoom=False), Div(id="TimeCanvasOptions"), Div(id="TimeCanvasViews")]),
                FlexN([funCanvas("Frequency", width=1024, height=256, useZoom=False), Div(id="FrequencyCanvasOptions"), Div(id="FrequencyCanvasViews")]),
                FlexN([graphCanvas(id="gainSat", width=256, height = 200, options=False), 
                       graphCanvas(id="meanPower", width=256, height = 200, options=False),
                       graphCanvas(id="sampleX", width=256, height = 200, options=False),
                       graphCanvas(id="kerrPhase", width=256, height = 200, options=False),
                ]),
                graphCanvas(id="sampleY", width=1024, height = 200, options=False),
                Button("Progress left", onclick="progressMultiTime(1)"),
                Button("Progress right", onclick="progressMultiTime(2)"),
                funCanvas("Test", width=1400, height=256, useZoom=True),
                graphCanvas(width=1400)
            )
        case 4: # tester
            added = Div(
                Div(initBeamType(), 
                    Button("Init", onclick="initElementsMultiMode(); initMultiMode(4);"),
                    Select(Option("All"), Option("1"), Option("2"), Option("3"), Option("4"), Option("5"), id="nMaxMatrices", **{'onchange':"nMaxMatricesChanged();"},),
                    Button("Full", onclick="initElementsMultiMode(); initMultiMode(4); fullCavityMultiMode()"),
                    Button("Roundtrip", onclick="initElementsMultiMode(); initMultiMode(4); roundtripMultiMode()"),
                    Button("Delta graph", onclick="initElementsMultiMode(); initMultiMode(4); deltaGraphMultiMode()"),
                    Button("Stability", onclick="initElementsMultiMode(); initMultiMode(4); calculateStability()"),
                    Button("Switch view", onclick="switchViewMultiMode()"),
                    Input(type="number", id=f'initialRange', placeholder="range(m)", step="0.001", style="width:80px;", value=f'0.005'),
                    Button("Auto range", onclick="initElementsMultiMode(); autoRangeMultiMode();"),
                    Select(Option("256"), Option("512"), Option("1024"), Option("2048"), Option("4096"), id="nSamples", **{'onchange':"nSamplesChanged();"},),
                ),
                Div(
                    *[Element(el, i, tab) for i, el in enumerate(elements[2])],
                    Button(NotStr("&#43;"), escapse=False, hx_post="/addElement/3", hx_target="#fun", hx_vals='js:{localId: getLocalId()}'), 
                ),
                funCanvas(1),
                graphCanvas()
            )
        case 5: # MultiTime on server
            added = generate_multi_on_server(data_obj)

    my_base64_jpgData = []
    for image in images:
        my_stringIOBytes = io.BytesIO()
        image.save(my_stringIOBytes, format='JPEG')
        my_stringIOBytes.seek(0)
        my_base64_jpgData.append(base64.b64encode(my_stringIOBytes.read()))
    
    print(len(images))

    return Div(Div(
        TabMaker("Multimode", "/tabfun/1", tab == 1),
        TabMaker("Crystal", "/tabfun/2", tab == 2),
        TabMaker("Multitime", "/tabfun/3", tab == 3),
        TabMaker("Tester", "/tabfun/4", tab == 4),
        TabMaker("MultitimeS", "/tabfun/5", tab == 5),

        ),
        # Div("Multimode",  hx_post="/tabfun/1", hx_target="#fun", cls=f"tab {'tabselected' if tab == 1 else ''}", hx_vals='js:{localId: getLocalId()}'),
        # Div("Crystal", hx_post="tabfun/2", hx_target="#fun", cls=f"tab {'tabselected' if tab == 2 else ''}", hx_vals='js:{localId: getLocalId()}'),
        # Div("Multitime", hx_post="tabfun/3", hx_target="#fun", cls=f"tab {'tabselected' if tab == 3 else ''}", hx_vals='js:{localId: getLocalId()}'),
        # Div("Tester", hx_post="tabfun/4", hx_target="#fun", cls=f"tab {'tabselected' if tab == 4 else ''}", hx_vals='js:{localId: getLocalId()}')),
        added,

        id="genMultiMode"
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


