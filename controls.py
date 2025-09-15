from fasthtml.common import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

def generate_chart(x, y, l, t, w=11, h=2, color="blue", marker=None, twinx=False):
    fig = plt.figure(figsize=(w, h))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    if twinx and len(y) == 2:
        plt.close(fig)
        fig, ax1 = plt.subplots(figsize=(w, h))
        ax1.plot(x[0], y[0], label=l[0], marker=marker, color=color[0] if isinstance(color, list) else color)
        ax2 = ax1.twinx()
        ax2.plot(x[1 if len(x) == 2 else 0], y[1], label=l[0], marker= marker, color=color[1] if isinstance(color, list) else color)
    else:
        for i in range(len(y)):
            if len(x) == len(y):
                xi = i
            else:
                xi = 0
            plt.plot(x[xi], y[i], label=l[0], marker=marker, color=color[i] if isinstance(color, list) else color)

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

def ABCDMatControl(name, M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    msg = f'&#9888; det={det}'
    return Div(
        Div(
            Div(
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
            InputCalcM(f'{name}_A', "A", f'{M[0][0]}', width = 180),
            InputCalcM(f'{name}_B', "B", f'{M[0][1]}', width = 180),
        ),
        Div(
            InputCalcM(f'{name}_C', "C", f'{M[1][0]}', width = 180),
            InputCalcM(f'{name}_D', "D", f'{M[1][1]}', width = 180),
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
