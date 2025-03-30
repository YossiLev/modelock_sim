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
