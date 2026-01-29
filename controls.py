from fasthtml.common import *
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def cget(x):
    return x.get() if hasattr(x, "get") else x

def FlexN(v, style = ""):
    return Div(*v, style=f"display: flex; gap: 3px; {style}")

def TabMaker(label, group, sel, target="#fun", inc=""):
    return Div(label,  hx_post=group, hx_target=target, hx_include=inc, cls=f"tab {'tabselected' if sel else ''}", hx_vals='js:{localId: getLocalId()}'),

def Header(title, help = None):
    return Div(
        FlexN(
            [Div(title, cls="header"),
            Button("?", cls="helpbutton", onclick="toggleVisibility(this.parentElement.nextElementSibling);") if help else "",
            ]
        ),
        Div(NotStr(help), cls="pophelp", style="position: absolute; visibility: hidden") if help else "",
        cls="headercontainer",
    )

def PickerDivs(id, count, sel, style=""):
    sOn = f"background-color: #4CAF50; color: white; {style}"
    sOff = f"background-color: #f1a1a1; color: black; {style}"
    return Div(
        Input(type="hidden", id=f"{id}_val", value=f"{sel}"),
        *[Div(id=f'{id}_{i}', style=sOn if i == sel else sOff, onclick=f"pickerDivsSelect('{id}', {i})") for i in range(count)],
        cls="pickerdivs",
        tabindex="0",
        id = id,
        **{'onkeydown':"handlePickerKeyDown(event);"}
   )
def Frame_chart_buttons(title):
    return Div(Div("O"), Div("C"), Div("W"), style="display: flex; flex-direction: column;")

def Frame_chart(title, *args, **kwargs):
    #return Div(Frame_chart_buttons(title), generate_chart(*args, **kwargs), style="display: flex; flex-direction: row;")
    return generate_chart(*args, **kwargs)

def generate_chart(x, y, l, t, w=11, h=2, color= ["#0000ff", "#ff0000", "#ff8800", "#aaaa00","#008800", "#ff00ff", "#110011"], marker=None, twinx=False, lw=1):
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    marker = marker if isinstance(marker, list) else [marker] * len(y)
    lw = lw if isinstance(lw, list) else [lw] * len(y)
    if twinx and len(y) == 2:
        plt.close(fig)
        fig, ax1 = plt.subplots(figsize=(w, h))
        ax1.plot(x[0], y[0], label=l[0], marker=marker[0], color=color[0] if isinstance(color, list) else color, linewidth=lw[0])
        ax2 = ax1.twinx()
        ax2.plot(x[1 if len(x) == 2 else 0], y[1], label=l[0], marker= marker[1], color=color[1] if isinstance(color, list) else color, linewidth=lw[1])
    else:
        for i in range(len(y)):
            if len(x) == len(y):
                xi = i
            else:
                xi = 0
            plt.plot(x[xi], y[i], label=l[0], marker=marker[i], color=color[i] if isinstance(color, list) else color, linewidth=lw[i])

    if (len(x) > 0):
        fig.axes[0].set_title(t)
        if (len([n for n in l if len(n) > 0])):
            plt.legend()
    my_stringIOBytes = io.BytesIO()
    plt.savefig(my_stringIOBytes, format='jpg')
    plt.close(fig)
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')

# y is complex and its intensity and phase will be resented together
def generate_chart_complex(x, y, t, w=11, h=2, marker=None, lw=1):
    generate_chart([x], [cget(np.angle(y)).tolist(), cget(np.absolute(y)).tolist()], [""], t, w=w, h=h, color=["green", "red"], marker=marker, twinx=True, lw=[1, 3]),


def InputCalcM(id, title, value, step=0.01, width = 150):
    return Div(
            Div(title, cls="floatRight", style="font-size: 10px; top:-3px; right:10px;background: #e7edb8;"),
            Input(type="number", id=id, title=title,
                value=value, step=f"{step}", 
#                    hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
#                    hx_vals='js:{localId: getLocalId()}',
                style=f"width:{width}px; margin:2px;",
                **{'onkeyup':f"validateMat(event);",
                    'onpaste':f"validateMat(event);",}),
            style="display: inline-block; position: relative;"
    )
           
def InputCalcS(id, title, value, step=0.01, width = 150):
    return Div(
            Div(title, cls="floatRight", style="font-size: 10px; top:-1px; right:10px; padding: 0px 4px; background: #e7f0f0;"),
            Input(type="number", id=id, title=title,
                value=value, step=f"{step}", 
                # hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
                # hx_vals='js:{localId: getLocalId()}',
                style=f"width:{width}px; margin:2px;"),
            style="display: inline-block; position: relative;"
    )

def SelectCalcS(tab, id, title, options, selected, width = 150):
    return Select(*[Option(o) if o != selected else Option(o, selected="1") for o in options], id=id,
                hx_trigger="input changed", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", hx_include="#calcForm *", 
                hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;")

def ABCDMatControl(name, M, cavity=""):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    msg = f'&#9888; det={det}'
    return Div(
        Div(
            Div(
                Img(src="/static/cavity.png", title="Open as cavity", width=20, height=20, onclick=f"ToggleCavity('{name}');"),
                Img(src="/static/eigen.png", title="Copy", width=20, height=20, onclick=f"AbcdMatEigenValuesCalc('{name}');"),
                Img(src="/static/copy.png", title="Copy", width=20, height=20, onclick=f"AbcdMatCopy('{name}');"),
                Img(src="/static/paste.png", title="Paste", width=20, height=20, onclick=f"AbcdMatPaste('{name}');"),
                cls="floatRight"
            ),
            Span(name), 
            Span(NotStr(msg), id=f"{name}_msg", 
                    style=f"visibility: {'hidden' if abs(det - 1.0) < 0.000001 else 'visible'}; color: yellow; background-color: red; padding: 1px; border-radius: 4px; margin-left: 30px; ") if len(msg) > 0 else "",
        ),
        Div(
            InputCalcM(f'{name}_A', "A", f'{M[0][0]}', width = 150),
            InputCalcM(f'{name}_B', "B", f'{M[0][1]}', width = 150),
        ),
        Div(
            InputCalcM(f'{name}_C', "C", f'{M[1][0]}', width = 150),
            InputCalcM(f'{name}_D', "D", f'{M[1][1]}', width = 150),
        ),
        Div(
            Textarea(cavity, id=f'{name}_cavity', style="min-height: 400px;", spellcheck="false"),
            Div(Img(src="/static/matABCD.png", title="Calc matrix", width=20, height=20, onclick=f"CavityToABCD('{name}');ToggleCavity('{name}');"),
                Img(src="/static/flip.png", title="Flip cavity", width=20, height=20, onclick=f"FlipCavity('{name}');")),
            id=f'{name}_cavity_frame',
            style="visibility: hidden; position: absolute; right: 15px; top: 25px; z-index: 1000;"
        ),

        Div("", id=f"{name}_eigen", style="visibility: hidden;"),
        cls="ABCDMatControl"
    )


def QvecControl(name, Q):
    return Div(
        Div(
            Span(name), 
        ),
        Div(
            InputCalcM(f'{name}_QR', "R", f'{Q[0]}', width = 180),
            InputCalcM(f'{name}_QI', "I", f'{Q[1]}', width = 180),
        ),
        cls="QvecControl"
    )

random_seed = 12345
def random_lcg():
    global random_seed
    a = 1664525
    c = 1013904223
    m = 2**32
    random_seed = (a * random_seed + c) % m
    return random_seed / m  # Returns a float between 0 and 1
def random_lcg_set_seed(seed):
    global random_seed
    random_seed = seed

def graphCanvas(id="graphCanvas", width=1000, height = 200, options=True, mode=1):
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
                    onclick=f"zoomGraph('{id}', 1, {mode});"
                ),
                Button(
                    Img(src="static/zoomout.png", alt="Zoom out", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"zoomGraph('{id}', -1, {mode});"
                ),
                Button(
                    Img(src="static/c12.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"selectGraph('{id}', {mode});"
                ),
                Button(
                    Img(src="static/degaussian.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"degaussGraph('{id}', {mode});"
                ),
                Button(
                    Img(src="static/delorentzian.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"delorentzGraph('{id}', {mode});"
                ),
                Button(
                    Img(src="static/desech.png", alt="Select", width="16", height="16"),
                    cls="imgButton",
                    onclick=f"desechGraph('{id}', {mode});"
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
