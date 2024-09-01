from fasthtml import FastHTML
from fasthtml.common import *
import numpy as np
import asyncio
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from simulation import generate_all_charts
from geometry import generate_canvas
from design import generate_design
from cavity import CavityDataPartsKerr, CavityData

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
        
def get_Data_obj(id):
    global gen_data
    try:
        dataObj = gen_data[id]
        return dataObj
    except:
        return None
@app.post("/menu/{new_tab}")
def menu(new_tab: str, localId: str):
    global current_tab
    dataObj = get_Data_obj(localId)
    current_tab = new_tab
    return make_page(dataObj)

def menu_item(item_name, current_item):
    sel = "Sel" if item_name == current_item else ""
    return Div(item_name, cls=f"menuItem{sel}", hx_post=F"/menu/{item_name}", hx_target="#fullPage", hx_vals='js:{localId: getLocalId()}')

def content_table(current_page, data_obj):
    global gen_data
    menu_list = ["Design", "Simulation", "Geometry", "Help"]
    return Div(*[menu_item(x, current_page) for x in menu_list],
                #Div(F"{data_obj['id']}" if data_obj else F"hh" , name="localId", id="localId"),
                Div(F"n = {len(list(gen_data.keys()))}"),  cls="sideMenu")

def my_frame(current_page, data_obj, content):
    return Div(
            Div(H1('Kerr Mode Locking Simulation')),
            Div(content_table(current_page, data_obj), content, cls="row"),
            id="fullPage"
        )
 
def make_page(data_obj):
    global current_tab, gen_data
    
    match current_tab:
        case "Simulation":
            return my_frame("Simulation", data_obj, 
                Div(
                    Button("Restart", hx_post=f"/init", hx_target="#charts", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Input(type="text", id="seedInit", name="seedInit", placeholder="Initial seed", style="width:90px;"),
                    Button("Step", hx_post="/inc", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Run", hx_ext="ws", ws_connect="/run", ws_send=True, hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    
                    Div(generate_all_charts(data_obj), id="charts"),
                    style="width:1100px"
                )
            )
        case "Geometry":
            return my_frame("Geometry", data_obj, 
                Div(
                    Div(generate_canvas(data_obj), cls="box", style="background-color: rgb(208 245 254);", id="charts2")))
        case "Design":
            return my_frame("Design", data_obj, 
                Div(
                    Button("Load", hx_post=f"/load", hx_target="#charts", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("Store", hx_post=f"/load", hx_target="#charts", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("New", hx_post=f"/load", hx_target="#charts", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    generate_design(data_obj), cls="box", style="background-color: rgb(208 245 254); width:90%;", id="charts2"))
        case _:
            return my_frame(current_tab, data_obj, Div("not yet"))
    

@app.get("/")
def home():
    global current_tab
    current_tab = "Simulation"

    return Body(make_page(None))
   

@app.post("/parnum/{id}")
def parameter_num(localId: str, id: str, param: str):
    print("parnum ", localId, id, param)
    global gen_data
    dataObj = gen_data[localId]
    cavity: CavityData = dataObj['cavityData']

    simParam, simComp  = cavity.getParameter(id)
    if simParam.set_value(param):
        if simComp:
            simComp.finalize()
            cavity.finalize()
        print(f"set value {param} to {id}")

    return simParam.render()


@app.post("/parpinn/{id}")
def parameter_num(localId: str, id: str):
    print("parpinn ", localId, id)
    global gen_data
    dataObj = gen_data[localId]
    cavity: CavityData = dataObj['cavityData']

    simParam, simComp  = cavity.getParameter(id)
    simParam.pinned = not simParam.pinned
    print(f"set pinned to {simParam.pinned} for {id}")

    return simParam.render()

@app.post("/init")
def init(session, seedInit: str, localId: str):
    global gen_data
    global gen_data
    if localId not in gen_data.keys():
        dataObj = {'id': localId, 'count': 0, 
                'run_state': False, 'cavityData': CavityDataPartsKerr()} 
        gen_data[localId] = dataObj
    
    dataObj = gen_data[localId]

    seed = 0
    try:
        seed = int(seedInit)
    except:
        pass
    if seed == 0:
        seed = int(np.random.rand() * (2 ** 32 - 1))

    dataObj['seed'] = seed
    dataObj['count'] = 0
    dataObj['run_state'] = False
    dataObj['cavityData'].restart(seed)

    add_toast(session, f"Simulation initialized", "info")

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
    dataObj['cavityData'].simulation_step()
    dataObj['count'] = count + 1

    return generate_all_charts(dataObj)

async def on_connect(session, send):
    print('Connected!')

async def on_disconnect(ws):
    print('Disconnected!')

@app.ws('/run', conn=on_connect, disconn=on_disconnect)
async def run(send, localId: str):
    global gen_data
    dataObj = gen_data[localId]

    dataObj['run_state'] = True
    count = dataObj['count']
    end_count = count + 1000

    start_cpu_time = time.time()

    while dataObj['run_state'] and count < end_count:
        
        dataObj['cavityData'].simulation_step()

        count = count + 1
        dataObj['count'] = count

        if count % 100 == 0:
            end_cpu_time = time.time()
            print(end_cpu_time - start_cpu_time, count)
            start_cpu_time = end_cpu_time
        await send(Div(generate_all_charts(dataObj), id="charts", cls="row"))
        await asyncio.sleep(0.001)
