import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
from fasthtml.common import *
from controls import *
import asyncio
from gen_data import *


def register_simulation(app):
    
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

        
def generate_simulation(data_obj, tab, offset = 0):
    return Div(
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

def generate_all_charts(dataObj):
    # try:
        if dataObj is None:
            return  "No data"
        dataObj.assure('cavityData')
        count = dataObj.count
        seed = dataObj.seed
        charts = dataObj.cavityData.get_state()
        analysis = dataObj.cavityData.get_state_analysis()
        analysisP = {k: f"{v:.2e}" if isinstance(v,float) else v for k,v in analysis.items()}

        return Div(
            Div(f"Seed {seed} - Step {count}", id="count"),
            Div(json.dumps(analysisP)),
            Div(
                Div(*[p.render() for p in dataObj.cavityData.getPinnedParameters(1)], cls="rowx"), 
                Div(*[Div(generate_chart([chart.x], [chart.y], chart.name), cls="box", style="background-color: #008080;") for chart in charts],
                    cls="rowx"
                ))

            , cls="column"
        )
    # except:
    #     return  "Error in data"