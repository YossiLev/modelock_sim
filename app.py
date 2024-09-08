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
from iterations import generate_iterations, Iteration
from cavity import CavityDataPartsKerr, CavityData
import app
import jsonpickle
import dataset
        

gen_data = {}
current_tab = "Simulation"
db_path = "sqlite:///data/mydatabase.db"

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
    if id not in gen_data.keys():
        return None
    return gen_data[id]
        
def get_sim_obj(id):
    dataObj = get_Data_obj(id)
    if dataObj is None:
        return None
    return dataObj['cavityData']
    
@app.post("/menu/{new_tab}")
def menu(new_tab: str, localId: str):
    global current_tab
    dataObj = get_Data_obj(localId)
    current_tab = new_tab
    return make_page(dataObj)

def menu_item(item_name, current_item):
    sel = "Sel" if item_name == current_item else ""
    return Div(item_name, cls=f"menuItem{sel}", hx_post=F"/menu/{item_name}", hx_target="#fullPage", hx_vals='js:{localId: getLocalId()}')

def content_table(current_page):
    global gen_data
    menu_list = ["Design", "Simulation", "Geometry", "Iterations", "Settings"]
    return Div(*[menu_item(x, current_page) for x in menu_list],
                Div(F"n = {len(list(gen_data.keys()))}"),  cls="sideMenu")

def my_frame(current_page, content):
    return Div(
            Div(H1('Kerr Mode Locking Simulation')),
            Div(content_table(current_page), content, cls="rowx"),
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
                    Button("Run", hx_ext="ws", ws_connect="/run", ws_send=True, hx_target="#charts", hx_swap="innerHTML", hx_include="#cbxQuick", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Label(Input(id="cbxQuick", type='checkbox', name='quick',checked=False), "Speed"),
                    
                    Div(generate_all_charts(data_obj), id="charts"),
                    style="width:1100px;"
                )
            )
        case "Geometry":
            return my_frame("Geometry", 
                Div(
                    Div(generate_canvas(data_obj, 1), cls="box", style="background-color: rgb(208 245 254);", id="geometry"), style="width:1100px"))
        case "Design":
            return my_frame("Design",
                Div(
                    Button("Load", hx_post=f"/load", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("Store", hx_post=f"/store", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("New", hx_post=f"/new", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    generate_design(data_obj), cls="box", style="background-color: rgb(208 245 254); width:90%; padding:3px 6px;", id="design"))
        
        case "Iterations":
            return my_frame("Iterations", 
                Div( Button("Prepare", hx_post=f"/iterInit", hx_target="#iterate", 
                   hx_include="#iterSeedInit, #iterStartValue, #iterEndValue, #iterValueSteps", 
                   hx_vals='js:{localId: getLocalId()}', hx_swap="outerHTML"), 
                    Input(type="text", id="iterSeedInit", name="iterSeedInit", placeholder="Initial seed", style="width:90px;"),
                    Input(type="text", id="iterStartValue", name="iterStartValue", placeholder="Start value", style="width:90px;"),
                    Input(type="text", id="iterEndValue", name="iterEndValue", placeholder="End value", style="width:90px;"),
                    Input(type="text", id="iterValueSteps", name="iterValueSteps", placeholder="Iteration steps number", style="width:90px;"),
                    Button("Step", hx_post="/iterStep", hx_target="#iterate", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Run", hx_ext="ws", ws_connect="/iterRun", ws_send=True, hx_target="#iterate", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/iterStop", hx_target="#iterate", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Div(generate_iterations(data_obj)), style="width:1100px"))
        case _:
            return my_frame(current_tab, Div("not yet"))
    

@app.get("/")
def home():
    global current_tab
    current_tab = "Simulation"

    return Body(make_page(None))
   

@app.post("/parnum/{id}")
def parameter_num(localId: str, id: str, param: str):
    dataObj = get_Data_obj(localId)
    cavity: CavityData = dataObj['cavityData']

    simParam, simComp  = cavity.getParameter(id)
    if simParam.set_value(param):
        if simComp:
            simComp.finalize()
            cavity.finalize()

    return simParam.render()


@app.post("/parpinn/{id}")
def parameter_num(localId: str, id: str):
    dataObj = get_Data_obj(localId)
    cavity: CavityData = dataObj['cavityData']

    simParam, simComp  = cavity.getParameter(id)
    simParam.pinned = not simParam.pinned

    return simParam.render()

@app.post("/init")
def init(session, seedInit: str, localId: str):
    global gen_data
    if localId not in gen_data.keys():
        dataObj = {'id': localId, 'count': 0, 
                'run_state': False, 'cavityData': CavityDataPartsKerr(), 
                'iterationRuns': []} 
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
    dataObj = get_Data_obj(localId)
    if dataObj:
        if dataObj['run_state']:
            dataObj['run_state'] = False

    return generate_all_charts(dataObj)

@app.post("/inc")
def increment(localId: str):
    global gen_data

    if localId in gen_data.keys():
        dataObj = gen_data[localId]
        count = dataObj['count']
        dataObj['cavityData'].simulation_step()
        dataObj['count'] = count + 1
    else:
        dataObj = None

    return generate_all_charts(dataObj)

async def on_connect(session, send):
    print('Connected!')

async def on_disconnect(ws):
    print('Disconnected!')

@app.ws('/run', conn=on_connect, disconn=on_disconnect)
async def run(send, quick: bool, localId: str):
    global gen_data
    if localId not in gen_data.keys():
        return
    dataObj = gen_data[localId]

    if dataObj['run_state']:
        return
    dataObj['run_state'] = True
    count = dataObj['count']
    end_count = count + 1000

    while dataObj['run_state'] and count < end_count:
        
        dataObj['cavityData'].simulation_step()

        count = count + 1
        dataObj['count'] = count

        if count % 100 == 0:
            dataObj['cavityData'].get_state_analysis()
            if quick == 1:
                await send(Div(generate_all_charts(dataObj), id="charts", cls="rowx"))
                await asyncio.sleep(0.001)

        if quick != 1:
            await send(Div(generate_all_charts(dataObj), id="charts", cls="rowx"))
            await asyncio.sleep(0.001)

    if count >= end_count:
        dataObj['run_state'] = False

#------------------- iterations
@app.post("/iterInit")
def iterInit(iterSeedInit: str, iterStartValue:str, iterEndValue:str, iterValueSteps: str, localId: str):
    global gen_data

    if localId not in gen_data.keys():
        dataObj = {'id': localId, 'count': 0, 
                'run_state': False, 'cavityData': CavityDataPartsKerr(), 
                'iterationRuns': []} 
        gen_data[localId] = dataObj
    
    dataObj = gen_data[localId]
    iterations = dataObj['iterationRuns']
    sim = dataObj['cavityData']
    parameters = sim.getPinnedParameters()

    seed = 0
    value_start = 1.0
    value_end = 10.0
    n_values = 10
    values_mode = "log"
    try:
        seed = int(iterSeedInit)
    except:
        pass

    try:
        value_start = float(iterStartValue)
        value_end = float(iterEndValue)
        n_values = int(iterValueSteps)
        
    except:
        pass

    iterations.append(Iteration(sim, seed, parameters[0], value_start, value_end, n_values, values_mode, name = f"General"))

    return generate_iterations(dataObj)

@app.post("/iterStop")
def stop(localId: str):
    global gen_data
    dataObj = gen_data[localId] 
    if dataObj['run_state']:
        dataObj['run_state'] = False
    return generate_iterations(dataObj)

@app.post("/iterStep")
def step(localId: str):
    global gen_data
    dataObj = gen_data[localId]

    iterations = dataObj['iterationRuns']
    iterations[0].step()

    return generate_iterations(dataObj)

async def on_connect_iter(session, send):
    print('_iterConnected!')

async def on_disconnect_iter(ws):
    print('_iterDisconnected!')

@app.ws('/iterRun', conn=on_connect_iter, disconn=on_disconnect_iter)
async def iterRun(send, localId: str):
    global gen_data
    dataObj = gen_data[localId]

    dataObj['run_state'] = True

    iteration = dataObj['iterationRuns'][0]
    print("start steps")
    while iteration.step():
        if not dataObj['run_state']:
            return
        await send(Div(generate_iterations(dataObj), id="iterate"))
        await asyncio.sleep(0.001)

@app.post("/removeComp/{comp_id}")
def removeComp(session, comp_id: str, localId: str):
    sim = get_sim_obj(localId)
    sim.removeComponentById(comp_id)
    return sim.render()

@app.post("/addAfter/{comp_id}")
def removeComp(session, comp_id: str, localId: str):
    sim = get_sim_obj(localId)
    sim.addAfterById(comp_id)
    return sim.render()

@app.post("/addBefore/{comp_id}")
def removeComp(session, comp_id: str, localId: str):
    sim = get_sim_obj(localId)
    sim.addBeforeById(comp_id)
    return sim.render()


@app.post("/setCompType/{comp_id}/{tp}")
def removeComp(session, comp_id: str, tp: str, localId: str):
    sim = get_sim_obj(localId)
    sim.replaceTypeById(comp_id, tp)
    return sim.render()

@app.post("/store")
def stop(localId: str):
    dataObj = get_Data_obj(localId)
    sim = get_sim_obj(localId)
    s = jsonpickle.encode(sim)
    db = dataset.connect(db_path)
    table = db['simulation']
    table.insert(dict(name=sim.name, desctiption=sim.description, content=s))

    print(db['simulation'].columns)
    print(len(db['simulation']))
    for simx in db['simulation']:
        print(simx['name'], simx['desctiption'])

    return generate_design(dataObj)

@app.post("/design")
def store(localId: str):
    dataObj = get_Data_obj(localId)
    return generate_design(dataObj)

@app.post("/store")
def store(localId: str):
    dataObj = get_Data_obj(localId)
    sim = get_sim_obj(localId)
    db = dataset.connect(db_path)
    table = db['simulation']
    table.insert(dict(name=sim.name, desctiption=sim.description, content=jsonpickle.encode(sim)))

    return generate_design(dataObj)


@app.post("/load")
def load(localId: str):
    db = dataset.connect(db_path)
    table = db['simulation']

    return Div(Table(
                Tr(Th("ID"), Th("Name"), Th("Description")),
                *[Tr(Td(simx['id']), Td(simx['name']), Td(simx['desctiption']),
                     hx_post=f"/load/{simx['id']}", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML") for simx in table],
                id="simulations"),
                Button("Back", hx_post=f"/design", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
            )

@app.post("/load/{id}")
def load(id: str, localId: str):
    global gen_data
    if localId not in gen_data.keys():
        dataObj = {'id': localId, 'count': 0, 
                'run_state': False, 'cavityData': CavityDataPartsKerr(), 
                'iterationRuns': []} 
        gen_data[localId] = dataObj
    dataObj = gen_data[localId] 
    db = dataset.connect(db_path)
    table = db['simulation']

    for simx in db['simulation']:
         print(simx['id'], simx['name'], simx['desctiption'])
    rec = table.find_one(id=id)

    s = rec["content"]
    sim = jsonpickle.decode(s)
    sim.finalize()
    dataObj['cavityData'] = sim
    return generate_design(dataObj)

@app.post("/tabgeo/{tabid}")
def load(tabid: str, localId: str):
    dataObj = get_Data_obj(localId)
    return generate_canvas(dataObj, int(tabid))

@app.post("/moveonchart/{offset}")
def load(offset: int, localId: str):
    dataObj = get_Data_obj(localId)
    return generate_canvas(dataObj, 3, offset)
