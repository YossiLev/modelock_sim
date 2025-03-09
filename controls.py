from fasthtml.common import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def FlexN(v):
    return Div(*v, style="display: flex; gap: 3px;")

def TabMaker(label, group, sel):
    return Div(label,  hx_post=group, hx_target="#fun", cls=f"tab {'tabselected' if sel else ''}", hx_vals='js:{localId: getLocalId()}'),

def generate_chart(x, y, l, t, w=11, h=2):
    fig = plt.figure(figsize=(w, h))
    for i in range(len(x)):
        plt.plot(x[i], y[i], label=l[i])
    
    if (len(x) > 0):
        fig.axes[0].set_title(t)
        plt.legend()
    my_stringIOBytes = io.BytesIO()
    plt.savefig(my_stringIOBytes, format='jpg')
    plt.close(fig)
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')