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
        ax2.plot(x[0], y[1], label=l[0], marker= marker, color=color[1] if isinstance(color, list) else color)
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
