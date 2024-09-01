import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fasthtml.common import *

def generate_chart(x, y, t):
    fig = plt.figure(figsize=(11, 2))
    plt.plot(x, y)
    fig.axes[0].set_title(t) #t + " {:.4e} -  {:.4e}".format(np.min(y), np.max(y))
    my_stringIOBytes = io.BytesIO()
    plt.savefig(my_stringIOBytes, format='jpg')
    plt.close(fig)
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')

def generate_all_charts(dataObj):
    # try:
        if dataObj is None:
            return  "No data"
        count = dataObj['count']
        seed = dataObj['seed']
        charts = dataObj['cavityData'].get_state()

        return Div(
            Div(f"Seed {seed} - Step {count}", id="count"),
            Div(
                Div(*[p.render() for p in dataObj['cavityData'].getPinnedParameters()], cls="row"), 
                *[Div(Div(generate_chart(chart.x, chart.y, chart.name), cls="box", style="background-color: #008080;")) for chart in charts],
                
                    cls="row"
                )

            , cls="column"
        )
    # except:
    #     return  "Error in data"