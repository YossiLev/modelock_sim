from fasthtml import FastHTML
from fasthtml.common import *
import numpy as np
import asyncio
import uuid
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from kerr import kerrStep
from simulation import generate_all_charts
from geometry import generate_canvas
from cavity import CavityDataKerr

gen_data = {}
current_tab = "Simulation"

app = FastHTML(ws_hdr=True, hdrs=(
        Link(rel="shortcut icon", type="image/x-icon", href="static/favicon.ico"),
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css"),
        Link(rel="stylesheet", href="static/main.css", type="text/css"),
        Script(src="static/localid.js"),
))
app.mount("/static", StaticFiles(directory="static"), name="static")
setup_toasts(app)
        
@app.get("/menu/{new_tab}")
def menu(session, new_tab: str):
    global current_tab
    print(current_tab, new_tab)
    current_tab = new_tab
    return make_page(session)


def menu_item(item_name, current_item):
    sel = "Sel" if item_name == current_item else ""
    return Div(item_name, cls=f"menuItem{sel}", hx_get=F"/menu/{item_name}", hx_target="#fullPage")

def content_table(current_page):
    global gen_data
    menu_list = ["Design", "Simulation", "Geometry", "Help"]
    return Div(*[menu_item(x, current_page) for x in menu_list],
                Div(F"hh", name="localId", id="localId"),
                Div(F"n = {len(list(gen_data.keys()))}"),  cls="sideMenu")

def my_frame(current_page, content):
    return Div(
            Div(H1('Kerr Mode Locking Simulation')),
            Div(content_table(current_page), content, cls="row"),
            id="fullPage"
        )
 
def make_page(data_obj):
    global current_tab, gen_data
    
    match current_tab:
        case "Simulation":
            return my_frame("Simulation",
                Div(
                    Button("Restart", hx_post=f"/init", hx_target="#charts", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Input(type="text", id="seedInit", name="seedInit", placeholder="Initial seed", style="width:90px;"),
                    Button("Step", hx_post="/inc", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Run", hx_ext="ws", ws_connect="/run", ws_send=True, hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    
                    Div(generate_all_charts(data_obj), id="charts"),
                    hx_include="#localId",
                    style="width:1100px"
                )
            )
        case "Geometry":
            return my_frame("Geometry", Div(Div(generate_canvas(), cls="box", style="background-color: #008080;", id="chart4")))
        case _:
            return my_frame(current_tab, Div("not yet"))
    

@app.get("/")
def home(session, request:Request):
    global current_tab
    current_tab = "Simulation"

    return Body(make_page(None)), cookie('csid', "aaa")
   

@app.post("/init")
def init(session, seedInit: str, localId: str):
    global gen_data

    seed = 0
    try:
        seed = int(seedInit)
    except:
        pass
    print(F"Seed Init is {seedInit}, seed is {seed}")
    count = 0
    if seed == 0:
        seed = int(np.random.rand() * (2 ** 32 - 1))

    add_toast(session, f"Simulation initialized", "info")
    print(F"localId value is ----- {localId}")
    dataObj = {'id': localId, 'seed': 0, 'count': 0, 'run_state': False,
                'data1': np.zeros(2049), 'data2': np.zeros(2049), 'data3': np.zeros(2049), 'data4': np.zeros(2049), 
                'cavityData': CavityDataKerr()}
    #print(F"jksdhfkjads {len(list(gen_data.keys()))} {gen_data.keys()}")
    #print(F"data obj {dataObj['id']}")
    gen_data[localId] = dataObj
    #(F"jksdhfkjads2 {len(list(gen_data.keys()))} {gen_data.keys()}")


    #seed = 693039070
    data1, data2, data3, data4 = dataObj['cavityData'].get_state()
    dataObj['seed'] = seed
    dataObj['count'] = count
    dataObj['run_state'] = False
    dataObj['data1'] = data1.tolist()
    dataObj['data2'] = data2.tolist()
    dataObj['data3'] = data3.tolist()
    dataObj['data4'] = data4.tolist()

    print("=====A1")
    return generate_all_charts(dataObj)

@app.post("/stop")
def stop(localId: str):
    global gen_data
    dataObj = gen_data[localId] 
    if dataObj['run_state']:
        dataObj['run_state'] = False
    return generate_all_charts(dataObj)

@app.post("/inc")
def increment(localId: str):
    global gen_data
    dataObj = gen_data[localId]

    count = dataObj['count']

    count = count + 1
    data1, data2, data3, data4 = kerrStep(dataObj['cavityData'])
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
async def run(send, localId: str):
    global gen_data
    dataObj = gen_data[localId]

    dataObj['run_state'] = True
    count = dataObj['count']

    start_cpu_time = time.time()

    while dataObj['run_state'] and count < 999:
        count = count + 1
        data1, data2, data3, data4 = kerrStep(dataObj['cavityData'])
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


# async def on_connectx(session, send):
#     print('xxxxConnected!')
#     print(session)
#     #await send(Div('Hello, you have connected', id="notifications"))

# async def on_disconnectx(ws):
#     print('xxxxxDisconnected!')

# @app.ws('/wscon', conn=on_connectx, disconn=on_disconnectx)
# async def runx(msg:str, xmsg:str):
#     print('runxxxxxx')
#     print(F"msg {msg}")
#     print(F"xmsg {xmsg}")
