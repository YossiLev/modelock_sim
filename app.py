from fastapi import WebSocket
from fasthtml import FastHTML
from fasthtml.common import *
import numpy as np
import asyncio
import time

#internal imports
from gen_data import *
from simulation import generate_all_charts
from geometry import generate_geometry, generate_beam_params
from fun import generate_multimode, collectData, generate_multi_on_server
from design import generate_design
from iterations import generate_iterations, Iteration
from cavity import CavityData
from calc import generate_calc
from settings import generate_settings

import app
import jsonpickle
import dataset   

current_tab = "Calculator"
db_path = "sqlite:///data/mydatabase.db"

app = FastHTML(htmx=False, ws_hdr=False, hdrs=(
        Link(rel="shortcut icon", type="image/x-icon", href="static/favicon.ico"),
        Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css"),
        Link(rel="stylesheet", href="static/main.css", type="text/css"),
        Link(rel="stylesheet", href="static/snackbar.css", type="text/css"),
        Script(src="static/htmx.min.js"),
        Script(src="static/ws.js"),
        Script(src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/3.3.0/math.min.js"),
        Script(src="https://cdn.plot.ly/plotly-latest.min.js"),
        Script(src="static/localid.js"),
        Script(src="static/memory.js"),
        Script(src="static/fieldValue.js"),
        Script(src="static/utils.js"),
        Script(src="static/stability.js"),
        Script(src="static/mover.js"),
        Script(src="static/graph2d.js"),
        Script(src="static/fourier.js"),
        Script(src="static/multimode.js"),
        Script(src="static/multitime.js"),
        Script(src="static/calculator.js"),

))
app.mount("/static", StaticFiles(directory="static"), name="static")
setup_toasts(app)

    
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
    menu_list = ["Design", "Simulation", "Geometry", "Iterations", "MultiMode", "Calculator", "Settings"]
    return Div(*[menu_item(x, current_page) for x in menu_list], cls="sideMenu")

def my_frame(current_page, content):
    return Div(
            Div("Some text some message..", id="snackbar"),
            Div(H1('Kerr Mode Locking Simulation')),
            Div(content_table(current_page), content, cls="rowx"),
            id="fullPage"
        )
 
def make_page(data_obj):
    global current_tab
    
    match current_tab:
        case "Simulation":
            return my_frame("Simulation",
                Div(
                    Button("Restart", hx_post=f"/init", hx_target="#charts", hx_include="#seedInit, #cbxmatlab", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Input(type="text", id="seedInit", name="seedInit", placeholder="Initial seed", style="width:90px;"),
                    Button("Step", hx_post="/inc", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Run", hx_ext="ws", ws_connect="/run", ws_send=True, hx_target="#charts", hx_swap="innerHTML", hx_include="#cbxQuick,#cbxmatlabt", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/stop", hx_target="#charts", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Label(Input(id="cbxQuick", type='checkbox', name='quick', checked=False), "Speed"),
                    Label(Input(id="cbxmatlab", type='checkbox', name='matlab', checked=False), "Matlab style"),
                    Div(generate_all_charts(data_obj), id="charts"),
                    style="width:1100px;"
                )
            )
        case "Geometry":
            return my_frame("Geometry", 
                Div(
                    Div(generate_geometry(data_obj, 1), cls="box", style="background-color: rgb(208 245 254);", id="geometry"), style="width:1100px"))
        case "Design":
            return my_frame("Design",
                Div(
                    Button("Load", hx_post=f"/load", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("Store", hx_post=f"/store", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    Button("New", hx_post=f"/new", hx_target="#cavity", hx_include="#seedInit", hx_vals='js:{localId: getLocalId()}', hx_swap="innerHTML"), 
                    generate_design(data_obj), cls="box", style="background-color: rgb(208 245 254); width:90%; padding:3px 6px;", id="design"))
        
        case "Iterations":
            return my_frame("Iterations", 
                Div( Button("Prepare", hx_post=f"/iterInit", hx_target="#iterateFull", 
                        hx_include="#iterSeedInit, #iterParams *, #iterName, #iterMaxCount",
                        hx_vals='js:{localId: getLocalId()}', hx_swap="outerHTML"), 
                    Input(type="text", id="iterSeedInit", name="iterSeedInit", placeholder="Initial seed", style="width:90px;"),
                    Input(type="text", id="iterName", name="iterName", placeholder="Label", style="width:190px;"),
                    Button("Step", hx_post="/iterStep", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Run", hx_ext="ws", ws_connect="/iterRun", ws_send=True, hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Button("Stop", hx_post="/iterStop", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Input(type="text", id="iterMaxCount", name="iterMaxCount", placeholder="End value", style="width:70px;", value="1500"),
                    Button("Run All ", hx_ext="ws", ws_connect="/iterRunAll", ws_send=True, hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                    Div(
                        *[Div(p[1].render(1), 
                            Input(type="text", id=f"iterStartValue{p[0]}", name=f"iterStartValue{p[0]}", placeholder="Start", style="width:70px; margin: 5px 4px 4px 8px;"),
                            Input(type="text", id=f"iterEndValue{p[0]}", name=f"iterEndValue{p[0]}", placeholder="End", style="width:70px; margin: 5px 4px 4px 0px;"),
                            Input(type="text", id=f"iterValueSteps{p[0]}", name=f"iterValueSteps{p[0]}", placeholder="Steps number", style="width:70px; margin: 5px 4px 4px 0px;"),
                            Select(Option("Linear"), Option("Logarithmic"), id=f"interpolationType{p[0]}"), cls="rowx"
                              ) for p in enumerate(data_obj.cavityData.getPinnedParameters(1))], 
                        Div(*[p.render() for p in data_obj.cavityData.getPinnedParameters(2)], cls="rowx"),
                        id="iterParams"
                    ),
                    Div(generate_iterations(data_obj), id="iterateFull"), style="width:1100px"))
        case "MultiMode":
            return my_frame("MultiMode", 
                Div(Div(generate_multimode(data_obj, 1), cls="box", style="background-color: rgb(208 245 254);", id="fun"), style="width:1100px"))
        case "Calculator":
            return my_frame("Calculator", 
                Div(Div(generate_calc(data_obj, 5), cls="box", style="background-color: rgb(208 245 254);", id="calculator"), style="width:1100px"))
        case "Settings":
            return my_frame("Settings", 
                Div(Div(generate_settings(data_obj), cls="box", style="background-color: rgb(208 245 254);", id="settings"), style="width:1100px"))
        
        case _:
            return my_frame(current_tab, Div("not yet"))
    

@app.get("/")
def home():
    global current_tab
    current_tab = "Calculator"

    localId = "init"
    dataObj = get_Data_obj(localId)

    return Body(make_page(dataObj))
   

@app.post("/parnum/{id}")
def parameter_num(localId: str, id: str, param: str):
    print(f"id = {id}, param = {param}")
    dataObj = get_Data_obj(localId)
    cavity: CavityData = dataObj.cavityData

    simParam, simComp  = cavity.getParameter(id)
    if simParam.set_value(param):
        if simComp:
            simComp.finalize()
        cavity.finalize()

    return simParam.render()


@app.post("/parpinn/{id}")
def parameter_num(localId: str, id: str):
    dataObj = get_Data_obj(localId)
    cavity: CavityData = dataObj.cavityData

    simParam, simComp  = cavity.getParameter(id)
    simParam.pinned = (simParam.pinned + 1) % 3

    return simParam.render()

@app.post("/init")
def init(session, seedInit: str, localId: str, matlab:bool = False):
    dataObj = get_Data_obj(localId)
    dataObj.assure('cavityData')
    
    seed = 0
    try:
        seed = int(seedInit)
    except:
        pass
    if seed == 0:
        seed = int(np.random.rand() * (2 ** 32 - 1))

    dataObj.seed = seed
    dataObj.count = 0
    dataObj.run_state = False
    dataObj.cavityData.restart(seed)

    add_toast(session, f"Simulation initialized", "info")

    return generate_all_charts(dataObj)

@app.post("/stop")
def stop(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('cavityData')
    dataObj.run_state = False

    return generate_all_charts(dataObj)

@app.post("/inc")
def increment(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('cavityData')
    dataObj.count += 1

    return generate_all_charts(dataObj)

async def on_connect(session, send):
    print('Connected!')

async def on_disconnect(ws):
    print('Disconnected!')

@app.ws('/run', conn=on_connect, disconn=on_disconnect)
async def run(send, quick: bool, localId: str, matlab:bool = False):
    dataObj = get_Data_obj(localId)
    sim = get_sim_obj(localId)

    if dataObj.run_state:
        return
    dataObj.run_state = True
    count = dataObj.count
    end_count = count + 1000
    sim.matlab = matlab
    sim.finalize()

    while dataObj.run_state and count < end_count:
        
        sim.simulation_step()

        count = count + 1
        dataObj.count = count

        if count % 100 == 0:
            sim.get_state_analysis()
            if quick == 1:
                await send(Div(generate_all_charts(dataObj), id="charts", cls="rowx"))
                await asyncio.sleep(0.001)

        if quick != 1:
            await send(Div(generate_all_charts(dataObj), id="charts", cls="rowx"))
            await asyncio.sleep(0.001)

    if count >= end_count:
        dataObj.run_state = False

#------------------- iterations
@app.post("/iterInit")
async def iterInit(request: Request, iterSeedInit: str, iterName: str, iterMaxCount: str, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('cavityData')

    form_data = await request.form()  # Get all form fields as a dict-like object

    iterations = dataObj.iterationRuns
    sim = dataObj.cavityData
    parameters = sim.getPinnedParameters(1)
    modifications = sim.getPinnedParameters(2)

    seed = 0
    value_start = []
    value_end = []
    n_values = []
    values_mode = []
    name = iterName.strip() if (len(iterName.strip()) > 0) else f"Iteration {len(iterations) + 1}"

    try:
        for i in range(len(parameters)):
            value_start.append(float(form_data.get(f"iterStartValue{i}")))
            value_end.append(float(form_data.get(f"iterEndValue{i}")))
            n_values.append(int(form_data.get(f"iterValueSteps{i}")))
            values_mode.append("log" if form_data.get(f"interpolationType{i}") == "Logarithmic" else "lin")
    except:
        pass

    try:
        seed = int(iterSeedInit)
    except:
        pass

    try:
        max_count = int(iterMaxCount)
    except:
        pass

    iterations.append(Iteration(sim, seed, modifications, parameters, value_start, value_end, n_values, values_mode, max_count, name = name))
    dataObj.iteration_focus = len(iterations) - 1

    return generate_iterations(dataObj)

@app.post("/iterStop")
def stop(localId: str):
    dataObj = get_Data_obj(localId)
    if dataObj.run_state:
        dataObj.run_state = False
    return generate_iterations(dataObj)

@app.post("/iterStep")
def step(localId: str):
    dataObj = get_Data_obj(localId)
    iterations = dataObj.iterationRuns
    index = dataObj.iteration_focus or 0

    iterations[index].step()

    return generate_iterations(dataObj)

@app.post("/iterChange/{index}")
def step(localId: str, index: int):
    dataObj = get_Data_obj(localId)
    dataObj.iteration_focus = index
    return generate_iterations(dataObj, True)

@app.post("/iterDelete/{index}")
def step(localId: str, index: int):
    dataObj = get_Data_obj(localId)
    if (len(dataObj.iterationRuns) > index):
        del dataObj.iterationRuns[index]
        dataObj.iteration_focus = 0
    return generate_iterations(dataObj)

@app.post("/iterUpdate/{index}")
def step(localId: str, index: int):
    dataObj = get_Data_obj(localId)
    if (len(dataObj.iterationRuns) > index):
        dataObj.iterationRuns[index].update_modifications()

    return generate_iterations(dataObj)

@app.post("/iterToggleShow/{index}")
def step(localId: str, index: int):
    dataObj = get_Data_obj(localId)
    if (len(dataObj.iterationRuns) > index):
        dataObj.iterationRuns[index].toggle_show()

    return generate_iterations(dataObj)

@app.post("/iterClear/{index}")
def step(localId: str, index: int):
    dataObj = get_Data_obj(localId)
    if (len(dataObj.iterationRuns) > index):
        dataObj.iterationRuns[index].clear()

    return generate_iterations(dataObj)

async def on_connect_iter(session, send):
    print('_iterConnected!')

async def on_disconnect_iter(ws):
    print('_iterDisconnected!')

@app.ws('/iterRun', conn=on_connect_iter, disconn=on_disconnect_iter)
async def iterRun(send, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.run_state = True

    index = dataObj.iteration_focus or 0
    indices = [index]

    while len(indices) > 0:
        index = indices.pop(0)
        dataObj.iteration_focus = index
        iteration = dataObj.iterationRuns[index]
        while iteration.step():
            if not dataObj.run_state:
                return
            if iteration.current_count % 20 == 0:
                await send(Div(generate_iterations(dataObj, full = False), id="iterate"))
                await asyncio.sleep(0.001)
        await send(Div(generate_iterations(dataObj), id="iterateFull"))
        await asyncio.sleep(0.001)

@app.ws('/iterRunAll', conn=on_connect_iter, disconn=on_disconnect_iter)
async def iterRunAll(send, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.run_state = True
    indices = [i for i in range(len(dataObj.iterationRuns))]

    while len(indices) > 0:
        index = indices.pop(0)
        dataObj.iteration_focus = index
        iteration = dataObj.iterationRuns[index]
        await send(Div(generate_iterations(dataObj), id="iterateFull"))
        await asyncio.sleep(0.001)        
        while iteration.step():
            if not dataObj.run_state:
                return
            if iteration.current_count % 20 == 0:
                await send(Div(generate_iterations(dataObj, full = False), id="iterate"))
                await asyncio.sleep(0.001)
        await send(Div(generate_iterations(dataObj), id="iterateFull"))
        await asyncio.sleep(0.001)

@app.post("/mmInit")
async def mmInit(request: Request, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
    
    form_data = await request.form()  # Get all form fields as a dict-like object
    dataObj.mmData.set({
        "beam_type": 0 if form_data.get("beamType") == "1-Dimensional" else 1,
        "seed": - 1 if len(form_data.get("seed").strip()) == 0 else int(form_data.get("seed")),
        "gain_factor": float(form_data.get("gainFactor")),
        "aperture": float(form_data.get("aperture")),
        "diffraction_waist": float(form_data.get("diffractionWaist")),
        "epsilon": float(form_data.get("epsilon")),
        "dispersion_factor": float(form_data.get("dispersionFactor")),
        "lensing_factor": float(form_data.get("lensingFactor")),
        "modulation_gain_factor": float(form_data.get("modulationGainFactor")),
        "is_factor": float(form_data.get("isFactor")),
        "crystal_shift": float(form_data.get("crystalShift")),
        "initial_range": float(form_data.get("initialRange")),
        "n_rounds_per_full": int(form_data.get("nRounds")),
        "report_every_step": int(form_data.get("reportEveryStep")),
        "steps_sounter": 0,
    })
    dataObj.mmData.init_multi_time()
    
    return generate_multi_on_server(dataObj)

@app.post("/mmUpdate")
async def mmUpdate(request: Request, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
    
    form_data = await request.form()  # Get all form fields as a dict-like object
    print(form_data)
    dataObj.mmData.set({
        "gain_factor": float(form_data.get("gainFactor")),
        "aperture": float(form_data.get("aperture")),
        "diffraction_waist": float(form_data.get("diffractionWaist")),
        "epsilon": float(form_data.get("epsilon")),
        "dispersion_factor": float(form_data.get("dispersionFactor")),
        "lensing_factor": float(form_data.get("lensingFactor")),
        "modulation_gain_factor": float(form_data.get("modulationGainFactor")),
        "is_factor": float(form_data.get("isFactor")),
        "crystal_shift": float(form_data.get("crystalShift")),
        "initial_range": float(form_data.get("initialRange")),
        "report_every_step": int(form_data.get("reportEveryStep")),
        "steps_sounter": int(form_data.get("stepsCounter")),
    })
    dataObj.mmData.update_helpData()
    
    return collectData(dataObj)

@app.post("/mmView/{part}/{action}")
async def mmView(part: int, action: str, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
    
    mmData = dataObj.mmData
    match action:
        case "Amp" | "Frq":
            mmData.view_on_amp_freq[part] = action
        case "Abs" | "Phs" | "Pow":
            mmData.view_on_abs_phase[part] = action
        case "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13" | "14":
            mmData.view_on_stage[part] = action

    return collectData(dataObj)

@app.put("/mmStop")
async def mmView(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.run_state = False

@app.ws('/mmRun')
async def mmRun(send, nRounds: str, gainFactor: str, aperture: str, diffractionWaist: str, epsilon: str, dispersionFactor: str,
                 lensingFactor: str, modulationGainFactor: str, isFactor: str, crystalShift: str, initialRange: str, reportEveryStep: str, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
    mmData = dataObj.mmData
    mmData.set({
         "gain_factor": float(gainFactor),
         "aperture": float(aperture),
         "epsilon": float(epsilon),
         "dispersion_factor": float(dispersionFactor),
         "diffraction_waist": float(diffractionWaist),
         "lensing_factor": float(lensingFactor),
         "modulation_gain_factor": float(modulationGainFactor),
         "is_factor": float(isFactor),
         "crystal_shift": float(crystalShift),
         "initial_range": float(initialRange),
         "report_every_step": int(reportEveryStep),
         "n_rounds_per_full": int(nRounds),
    #     "steps_counter": int(form_data.get("stepsCounter")),
    })
    mmData.update_helpData()
    last_sent = 0

    start_time = time.time()
    count = int(nRounds)
    dataObj.run_state = True
    for i in range(count):
        if not dataObj.run_state:
            break       
        mmData.multi_time_round_trip()
        if (i + 1) % mmData.report_every_step == 0:
            try:
                last_sent = i + 1
                if last_sent >= count:
                    dataObj.run_state = False
                await send(Div(collectData(dataObj, 0, more=last_sent < count), id="numData"))
                await asyncio.sleep(0.001)
            except Exception as e:
                print(f"Error sending data: {e}")
                dataObj.run_state = False
                return
    
    if last_sent < count:
        dataObj.run_state = False
        await send(Div(collectData(dataObj), id="numData"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    await asyncio.sleep(0.001)

@app.post("/mmGraph/{sample}/{x}/{y}")
async def mmGraph(request: Request, sample: int, x: int, y: int):
    jj = await request.json()

    localId = jj["localId"]
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')

    mmData = dataObj.mmData
    mmData.view_on_sample = sample
    mmData.view_on_x = x
    mmData.view_on_y = y

    return mmData.serialize_mm_graphs()
        
@app.post("/mmCenter")
async def mmCenter(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
   
    dataObj.mmData.center_multi_time()
    
    return collectData(dataObj)

@app.post("/mmSaveState")
async def mmSaveState(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
   
    dataObj.mmData.saveState()
    return collectData(dataObj)

@app.post("/mmRestoreState")
async def mmRestoreState(localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('mmData')
   
    dataObj.mmData.restoreState()
    return collectData(dataObj)

def collect_mat_data(M, form_data, name):
    updated = False
    try:
        M[0][0] = float(form_data.get(f"{name}_A"))
        updated = True
    except:
        pass
    try:
        M[0][1] = float(form_data.get(f"{name}_B"))
        updated = True
    except:
        pass
    try:
        M[1][0] = float(form_data.get(f"{name}_C"))
        updated = True
    except:
        pass
    try:
        M[1][1] = float(form_data.get(f"{name}_D"))
        updated = True
    except:
        pass
    if not updated:
        raise ValueError("Matrix data not updated") 
    return M

def pushParam(target, name, extractor):
    try:
        value = extractor()
        #print(name, value)
        if value is not None:
            #print(f"pushParam Setting {name} to {value} {target}")
            target.set({name: value})
    except:
        pass

@app.post("/clUpdate/{tab}")
async def clUpdate(request: Request, tab: int, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('calcData')

    calcData = dataObj.calcData
    form_data = await request.form()

    doCalcUpdate(calcData, form_data)

    return generate_calc(dataObj, tab)

def doCalcUpdate(calcData, form_data):
    try:
        pushParam(calcData, "M1", lambda: collect_mat_data(calcData.M1, form_data, "M1"))
        pushParam(calcData, "M2", lambda: collect_mat_data(calcData.M2,form_data, "M2"))
        pushParam(calcData, "M3", lambda: collect_mat_data(calcData.M3,form_data, "M3"))
        pushParam(calcData, "t_fixer", lambda: float(form_data.get("MatFixer")))
    except:
        pass
    try:
        ct = form_data.get("cavityText")
        if ct is not None:
            calcData.set({
                "cavity_text": ct,
            })
        pushParam(calcData, "cavity_mat", lambda: collect_mat_data(calcData.cavity_mat, form_data, "MCavity"))
    except:
        pass
    try:
        pushParam(calcData, "fresnel_mat", lambda: collect_mat_data(calcData.fresnel_mat, form_data, "MFresnel"))
        pushParam(calcData, "fresnel_N", lambda: int(form_data.get("FresnelN")))
        pushParam(calcData, "fresnel_factor", lambda: float(form_data.get("FresnelFactor")))
        pushParam(calcData, "fresnel_dx_in", lambda: float(form_data.get("FresnelDXIn")))
        pushParam(calcData, "fresnel_dx_out", lambda: float(form_data.get("FresnelDXOut")))
        pushParam(calcData, "fresnel_waist", lambda: float(form_data.get("FresnelWaist")))
        pushParam(calcData, "select_front", lambda: form_data.get("CalcSelectFront"))
    except:
        pass
    try:
        pushParam(calcData, "harmony", lambda: int(form_data.get("pulseHarmony")))
    except:
        pass
    try:
        pushParam(calcData, "calculation_rounds", lambda: int(form_data.get("DiodeRounds")))
        pushParam(calcData, "diode_pulse_width", lambda: float(form_data.get("DiodePulseWidth")))
        pushParam(calcData, "diode_alpha", lambda: float(form_data.get("DiodeAlpha")))
        pushParam(calcData, "diode_gamma0", lambda: float(form_data.get("DiodeGamma0")))
        pushParam(calcData, "diode_saturation", lambda: float(form_data.get("DiodeSaturation")))
        pushParam(calcData, "absorber_half_time", lambda: float(form_data.get("AbsorberHalfTime")))
        pushParam(calcData, "gain_half_time", lambda: float(form_data.get("GainHalfTime")))

        pushParam(calcData, "Ta", lambda: float(form_data.get("Ta")))
        pushParam(calcData, "Tb", lambda: float(form_data.get("Tb")))
        pushParam(calcData, "Pa", lambda: float(form_data.get("Pa")))
        pushParam(calcData, "Pb", lambda: float(form_data.get("Pb")))
        pushParam(calcData, "Ga", lambda: float(form_data.get("Ga")))
        pushParam(calcData, "Gb", lambda: float(form_data.get("Gb")))
        pushParam(calcData, "N0a", lambda: float(form_data.get("N0a")))
        pushParam(calcData, "N0b", lambda: float(form_data.get("N0b")))
        pushParam(calcData, "dt", lambda: float(form_data.get("dt")))
        pushParam(calcData, "volume", lambda: float(form_data.get("volume")))
        pushParam(calcData, "initial_photons", lambda: float(form_data.get("initial_photons")))
        pushParam(calcData, "cavity_loss", lambda: float(form_data.get("cavity_loss")))
        pushParam(calcData, "h", lambda: float(form_data.get("h")))
        pushParam(calcData, "diode_update_pulse", lambda: form_data.get("CalcDiodeUpdatePulse"))
        pushParam(calcData, "diode_intensity", lambda: form_data.get("CalcDiodeSelectIntensity"))
        pushParam(calcData, "start_gain", lambda: float(form_data.get("start_gain")))
        pushParam(calcData, "start_absorber", lambda: float(form_data.get("start_absorber")))


    except:
        pass

@app.post("/doCalc/{tab}/{cmd}/{params}")
async def doCalc(request: Request, tab: int, cmd: str, params: str, localId: str):
    print(f"doCalc: {tab}, {cmd}, {params}, {localId}")
    dataObj = get_Data_obj(localId)
    dataObj.assure('calcData')
    calcData = dataObj.calcData

    print(f"calcData after assure: {calcData}")
    form_data = await request.form()
    doCalcUpdate(calcData, form_data)

    dataObj.calcData.doCalcCommand(cmd, params, dataObj)

    return generate_calc(dataObj, tab)

@app.post("/settings/{cmd}/{params}")
async def settings(cmd: str, params: str, localId: str, password: str | None = None):
    dataObj = get_Data_obj(localId)
    match cmd:
        case "delete":
            clear_data_obj(localId)
        case "authenticate":            
            if params == "1" and password is not None:
                rc = dataObj.authenticate(password)
            else:
                dataObj.authenticate("stam")            

    return generate_settings(dataObj)

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

# @app.post("/store")
# def stop(localId: str):
#     dataObj = get_Data_obj(localId)
#     sim = get_sim_obj(localId)
#     s = jsonpickle.encode(sim)
#     db = dataset.connect(db_path)
#     table = db['simulation']
#     table.insert(dict(name=sim.name, desctiption=sim.description, content=s))

#     for simx in db['simulation']:
#         print(simx['name'], simx['desctiption'])

#     return generate_design(dataObj)

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
    dataObj = get_Data_obj(localId)
    dataObj.assure('cacityData')

    db = dataset.connect(db_path)
    table = db['simulation']

    for simx in db['simulation']:
         print(simx['id'], simx['name'], simx['desctiption'])
    rec = table.find_one(id=id)

    s = rec["content"]
    sim = jsonpickle.decode(s)
    sim.finalize()
    dataObj.cavityData = sim
    return generate_design(dataObj)

@app.post("/store_iter/{label}")
def store(label: str, localId: str):
    dataObj = get_Data_obj(localId)
    iterations = dataObj.iterationRuns
    db = dataset.connect(db_path)
    table = db.iterations
    for iteration in iterations:
        print(dict(labels=f"@{label}@", content=jsonpickle.encode(iteration)))
        #table.insert(dict(labels=f"@{label}@", content=jsonpickle.encode(iteration)))

    return generate_iterations(dataObj)

@app.post("/tabgeo/{tabid}")
def load(tabid: int, localId: str):
    return generate_geometry(get_Data_obj(localId), tabid)

@app.post("/tabfun/{tabid}")
def load(tabid: int, localId: str):
    return generate_multimode(get_Data_obj(localId), tabid)

@app.post("/tabcalc/{tabid}")
async def tabcalc(request: Request, tabid: int, localId: str):
    dataObj = get_Data_obj(localId)
    dataObj.assure('calcData')

    calcData = dataObj.calcData
    form_data = await request.form()

    doCalcUpdate(calcData, form_data)

    return generate_calc(get_Data_obj(localId), tabid)

@app.post("/moveonchart/{offset}/{tab}")
def load(offset: int, tab: int, localId: str):
    return generate_geometry(get_Data_obj(localId), tab, offset)

@app.post("/recordstep/{offset}")
def load(offset: int, localId: str):
    sim = get_sim_obj(localId)
    sim.get_record_steps(offset)
    return generate_geometry(get_Data_obj(localId), 4, offset)

@app.post("/beamParams/{tabid}")
def parameter_num(tabid: str, localId: str, beam_x: str, beam_theta: str):
    print(f"beam_x {beam_x} beam_theta {beam_theta}")
    dataObj = get_Data_obj(localId)
    sim = get_sim_obj(localId)

    try:
        sim.str_beam_x = beam_x
        sim.beam_x = 0.001 * float(beam_x)
        sim.beam_x_error = False
    except ValueError as ve:
        print(f"*** Error set beam_x {beam_x} ***")
        sim.valuebeam_x_error_error = True
    
    try:
        sim.str_beam_theta = beam_theta
        sim.beam_theta = math.radians(float(beam_theta))
        sim.beam_theta_error = False
    except ValueError as ve:
        print(f"*** Error set beam_theta {beam_theta} ***")
        sim.beam_theta_error = True
    
    sim.build_beam_geometry()
    return generate_geometry(dataObj, int(tabid))

from fun import elements

@app.post("/addElement/{tab}")
def addlens(localId: str, tab: int):
    elements[tab - 1].append({"t": "L", "par":[0.2, 0.1]})
    return generate_multimode(get_Data_obj(localId), tab)

@app.post("/setcavity/{tab}/{cavity_name}")
def setcavity(localId: str, tab: int, cavity_name: str):
    dataObj = get_Data_obj(localId)
    dataObj.current_cavity_name = cavity_name
    return generate_multimode(dataObj, tab)

@app.post("/removeElements/{tab}/{index}")
def addlens(index: int, localId: str, tab: int):
    elements[tab - 1].pop(index)
    return generate_multimode(get_Data_obj(localId), tab)


# uvicorn app:app --host 0.0.0.0 --port 443 --ssl-keyfile=sim_key.pem --ssl-certfile=sim_cert.pem