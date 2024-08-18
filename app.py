from fasthtml import FastHTML
from fasthtml.common import *
import numpy as np
import asyncio
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from kerr import kerrInit, kerrStep

app = FastHTML(ws_hdr=True, hdrs=(
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css"),
        Link(rel="stylesheet", href="static/main.css", type="text/css"),

))
app.mount("/static", StaticFiles(directory="static"), name="static")


count = 0
run_state = False
data1 = np.zeros(2049)
data2 = np.zeros(2049)
data3 = np.zeros(2049)
data4 = np.zeros(2049)

def generate_chart(data, t):
    fig = plt.figure(figsize=(16, 2))
    plt.plot(range(len(data)), data)
    fig.axes[0].set_title(t) #t + " {:.4e} -  {:.4e}".format(np.min(y), np.max(y))
    my_stringIOBytes = io.BytesIO()
    plt.savefig(my_stringIOBytes, format='jpg')
    plt.close(fig)
    my_stringIOBytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIOBytes.read())

    return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')

def generate_all_charts():
    global count, data1, data2, run_state, count

    return Div(
        Div(f"Count is set to {count}", id="count"),
        Div(
                Div(Div(generate_chart(data1, "Power"), cls="box", style="background-color: #008080;", id="chart1")),
                Div(Div(generate_chart(data2, "Spectrum"), cls="box", style="background-color: #008080;", id="chart2")),
                Div(Div(generate_chart(data3, "Waist"), cls="box", style="background-color: #008080;", id="chart3")),
                Div(Div(generate_chart(data4, "Phase"), cls="box", style="background-color: #008080;", id="chart4")),
                cls="row"
            )
        ,cls="column"
    )
        
       

@app.get("/")
def home():

    return Body(
            Div(H1('Kerr Mode Locking Simulation')), 
            Button("Restart", hx_post="/init", hx_target="#charts", hx_swap="innerHTML"),
            Button("Step", hx_post="/inc", hx_target="#charts", hx_swap="innerHTML"),
            Button("Run", ws_send="1", hx_ext="ws", ws_connect="/run", hx_target="#charts", hx_swap="innerHTML"),
            Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML"),

            Div(generate_all_charts(), id="charts")
    )       


@app.post("/init")
def init():
    global count, data1, data2, data3, data4, run_state

    count = 0
    data1, data2, data3, data4 = kerrInit()
    return generate_all_charts()

@app.post("/stop")
def stop():
    global count, data1, data2, data3, data4, run_state

    run_state = False
    print("stop ", run_state)
    return generate_all_charts()

@app.post("/inc")
def increment():
    global count, data1, data2, data3, data4, run_state

    count = count + 1
    data1, data2, data3, data4 = kerrStep(count)

    return generate_all_charts()

async def on_connect(send):
    print('Connected!')
    #await send(Div('Hello, you have connected', id="notifications"))

async def on_disconnect(ws):
    print('Disconnected!')

@app.ws('/run', conn=on_connect, disconn=on_disconnect)
async def run(send):
    global count, data1, data2, data3, data4, run_state

    run_state = True

    while run_state and count < 999:
        count = count + 1
        data1, data2, data3, data4 = kerrStep(count)
        await send(Div(generate_all_charts(), id="charts", cls="row"))
        await asyncio.sleep(0.001)


    # print("send------")
    # count = count + 1
    # data1, data2, data3, data4 = kerrStep(count)

    # return generate_all_charts()


