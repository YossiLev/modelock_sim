import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fasthtml.common import *

def generate_chart(data, t):
    fig = plt.figure(figsize=(11, 2))
    plt.plot(range(len(data)), data)
    fig.axes[0].set_title(t) #t + " {:.4e} -  {:.4e}".format(np.min(y), np.max(y))
    my_stringIOBytes = io.BytesIO()
    plt.savefig(my_stringIOBytes, format='jpg')
    plt.close(fig)
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')



def generate_all_charts(dataObj):
    try:
        id = dataObj['id']
        count = dataObj['count']
        seed = dataObj['seed']
        data1 = dataObj['data1']
        data2 = dataObj['data2']
        data3 = dataObj['data3']
        data4 = dataObj['data4']

        return Div(
            Div(f"Seed {seed} - Step {count}", id="count"),
            Div(
                    Div(Div(generate_chart(data1, "Power"), cls="box", style="background-color: #008080;", id="chart1")),
                    Div(Div(generate_chart(data2, "Spectrum"), cls="box", style="background-color: #008080;", id="chart2")),
                    Div(Div(generate_chart(data3, "Waist"), cls="box", style="background-color: #008080;", id="chart3")),
                    Div(Div(generate_chart(data4, "Phase"), cls="box", style="background-color: #008080;", id="chart4")),
                    cls="row"
                )
            , cls="column"
        )
    except:
        return  "No data"