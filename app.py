from fasthtml import FastHTML
from fasthtml.common import *
import numpy as np
import asyncio
import uuid
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from kerr import kerrInit, kerrStep

gen_data = {}

dataObj = null

app = FastHTML(ws_hdr=True, hdrs=(
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css"),
        Link(rel="stylesheet", href="static/main.css", type="text/css"),

))
app.mount("/static", StaticFiles(directory="static"), name="static")

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
            #Div(f"{id}", name="sid", id="sid", ws_send=True ),
            Div(
                    Div(Div(generate_chart(data1, "Power"), cls="box", style="background-color: #008080;", id="chart1")),
                    Div(Div(generate_chart(data2, "Spectrum"), cls="box", style="background-color: #008080;", id="chart2")),
                    Div(Div(generate_chart(data3, "Waist"), cls="box", style="background-color: #008080;", id="chart3")),
                    Div(Div(generate_chart(data4, "Phase"), cls="box", style="background-color: #008080;", id="chart4")),
                    cls="row"
                )
            ,cls="column"
        )
    except:
        return  "No data"
        
def menu_item(item_name, current_item):
    sel = "Sel" if item_name == current_item else ""
    return Div(item_name, cls=f"menuItem{sel}")

def content_table(current_page):
    menu_list = ["Simulation", "Design", "Help"]
    return Div(*[menu_item(x, current_page) for x in menu_list], cls="sideMenu")

def my_frame(current_page, content):
    return Body(
        Div(H1('Kerr Mode Locking Simulation')),
        Div(content_table(current_page), content, cls="row")
    )
 
@app.get("/")
def home(session):

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    data_obj = {'id': session['session_id'], 'seed': 0, 'count': 0, 'run_state': False,
                'data1': np.zeros(2049), 'data2': np.zeros(2049), 'data3': np.zeros(2049), 'data4': np.zeros(2049), }
    gen_data[session['session_id']] = data_obj

    return my_frame ("Simulation",
        Div(
            Button("Restart", hx_post=f"/init", hx_target="#charts", hx_include="#seedInit", hx_swap="innerHTML"), 
            Input(type="text", id="seedInit", name="seedInit", placeholder="Initial seed", style="width:90px;"),
            Button("Step", hx_post="/inc", hx_target="#charts", hx_swap="innerHTML"),
            Button("Run", hx_ext="ws", ws_connect="/run", ws_send=True, hx_target="#charts", hx_swap="innerHTML"),
            Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML"),
            
            Div(generate_all_charts(data_obj), id="charts"),

            style="width:1400px"
        
        )
    )      

@app.post("/init")
def init(session, seedInit: str):
    global dataObj
    dataObj = gen_data[session['session_id']] 

    seed = 0
    try:
        seed = int(seedInit)
    except:
        pass
    print(seedInit, seed)
    count = 0
    if seed == 0:
        seed = int(np.random.rand() * (2 ** 32 - 1))
    #seed = 693039070
    data1, data2, data3, data4 = kerrInit(seed)
    dataObj['seed'] = seed
    dataObj['count'] = count
    dataObj['run_state'] = False
    dataObj['data1'] = data1.tolist()
    dataObj['data2'] = data2.tolist()
    dataObj['data3'] = data3.tolist()
    dataObj['data4'] = data4.tolist()

    return generate_all_charts(dataObj)

@app.post("/stop")
def stop(session):
    dataObj = gen_data[session['session_id']] 

    dataObj['run_state'] = False
    return generate_all_charts(session)

@app.post("/inc")
def increment(session):
    dataObj = gen_data[session['session_id']] 

    count = dataObj['count']

    count = count + 1
    data1, data2, data3, data4 = kerrStep(count)
    dataObj['count'] = count
    dataObj['data1'] = data1
    dataObj['data2'] = data2
    dataObj['data3'] = data3
    dataObj['data4'] = data4

    return generate_all_charts(dataObj)

async def on_connect(session, send):
    print('Connected!')
    print(session)
    #await send(Div('Hello, you have connected', id="notifications"))

async def on_disconnect(ws):
    print('Disconnected!')

@app.ws('/run', conn=on_connect, disconn=on_disconnect)
async def run(send):
    global dataObj

    print("in run ========1")
    print(dataObj['id'])
    id = dataObj['id']

    dataObjLocal = gen_data[id] 
    dataObjLocal['run_state'] = True
    count = dataObjLocal['count']

    start_cpu_time = time.time()

    while dataObjLocal['run_state'] and count < 999:
        count = count + 1
        data1, data2, data3, data4 = kerrStep(count)
        dataObj['count'] = count
        dataObj['data1'] = data1
        dataObj['data2'] = data2
        dataObj['data3'] = data3
        dataObj['data4'] = data4

        if count % 10 == 0:
            end_cpu_time = time.time()
            print(end_cpu_time - start_cpu_time, count)
            start_cpu_time = end_cpu_time
        await send(Div(generate_all_charts(dataObj), id="charts", cls="row"))
        await asyncio.sleep(0.001)



    # print("send------")
    # count = count + 1
    # data1, data2, data3, data4 = kerrStep(count)

    # return generate_all_charts()


