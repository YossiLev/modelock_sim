import datetime
import numpy as np
import json
from fasthtml.common import *
from controls import *

from app import get_Data_obj

class Iteration():
    def __init__(self, sim, seed, modifications, parameter, value_start, value_end, n_values, values_mode, max_count, name = f"General"):
        self.sim = sim
        self.seed = seed
        self.modifications = modifications
        self.parameterId = parameter.id
        self.parameterName = parameter.name
        self.value_start = value_start
        self.value_end = value_end
        self.n_values = n_values
        self.values_mode = values_mode
        self.name = name
        self.date_create = datetime.datetime.now().strftime("%d-%m-%Y%H:%M:%S")
        if values_mode == "log":
            self.values = np.exp(np.linspace(np.log(value_start), np.log(value_end), n_values))
        else:
            self.values = np.linspace(value_start, value_end, n_values)
        self.state = ['----------' for v in self.values]
        self.reports = [[] for v in self.values]
        self.reportsFinal = ["Wait.." for v in self.values]
        self.seeds = [-1 for v in self.values]

        self.current_index = 0
        self.current_count = 0
        self.max_count = max_count

    def step(self):
        if self.current_index < self.n_values:
            if self.current_count < self.max_count:
                if self.current_count == 0:
                    if self.seed != 0:
                        seed = self.seed
                    else:
                        seed = int(np.random.rand() * (2 ** 32 - 1))
                    self.seeds[self.current_index] = seed

                    self.sim.restart(seed)
                    p, part = self.sim.getParameter(self.parameterId)
                    v = self.values[self.current_index]
                    p.set_value(str(v))
                    if (part):
                        part.finalize()
                    self.sim.finalize()
                self.sim.simulation_step()
                self.current_count += 1
                if self.current_count % (self.max_count / 10) == 0:
                    analysis = self.sim.get_state_analysis()
                    self.reports[self.current_index].append(analysis)
                    analysisP = {k: f"{v:.3e}" if isinstance(v,float) else v for k,v in analysis.items()}
                    self.reportsFinal[self.current_index] = json.dumps(analysisP)
                    *state_list, = self.state[self.current_index]
                    state_list[round(self.current_count / (self.max_count / 10)) - 1] = analysis['code']
                    self.state[self.current_index] = ''.join(map(str, state_list))
                    #cb(self.current_index, self.current_count, sim_report(self.sim))                
            else:
                self.current_count = 0
                self.current_index += 1
        else:
            return False
        
        return True

    def render(self):
        return Div(
            Div(self.name), 
            Div(Div(*[p.render() for p in self.modifications], cls="rowx"), cls="rowx"),
            Table(
                Tr(Th(self.parameterName), Th("Seed"), Th("State", style="min-width:140px;"), Th("Report")),
                *[Tr(Td(f"{value:.4f}", cls="monoRight"), 
                     Td(f"{seed}", cls="monoRight"), 
                     Td(state, cls="mono", style="min-width:100px;"), 
                     Td(report, style="font-size:11px;")) for value, seed, state, report in zip(self.values, self.seeds, self.state, self.reportsFinal)]
            )
        )

def extract_paramater_value(rep, paramaterName):
    obj = json.loads(rep)
    if (paramaterName in obj):
        return float(obj[paramaterName])
    return 0.0

def generate_iter_chart(dataObj, parameterName):
    vecX = []
    vecY = []
    vecL = []
    for iteration in dataObj['iterationRuns']:
        if (iteration.current_index >= iteration.n_values):
            power = list(map(lambda x: extract_paramater_value(x, parameterName), iteration.reportsFinal))
            vecX.append(iteration.values)
            vecY.append(power)
            vecL.append(iteration.name)

    chart = Div(generate_chart(vecX, vecY, vecL, parameterName, w=5, h=3))

    return chart

def generate_iterations(dataObj, full = True):
    # try:
        index = dataObj['iteration_focus'] if 'iteration_focus' in dataObj.keys() else 0
        if dataObj is not None and dataObj is not None and len(dataObj['iterationRuns']) > index:
            iteration = dataObj['iterationRuns'][index]
        else:
            iteration = None

        if iteration is not None:
            counts = f"Seed {iteration.seed} - Index {iteration.current_index} - Step {iteration.current_count}"
        else:
            counts = ""
           
        return Div(
            FlexN([
                Div(
                    Div(counts, id="iter_count") ,
                    Div(Div(*[p.render() for p in dataObj['cavityData'].getPinnedParameters(1)], cls="rowx"), cls="rowx"),
                    Div(
                        *list(map(lambda x: Button(x[1].name, hx_post=f"/iterChange/{x[0]}", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'), enumerate(dataObj['iterationRuns']))),
                        Div(Button("Delete", hx_post=f"/iterDelete/{index}", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}')),
                    ),
                    iteration.render() if iteration is not None else Div("no values"),
                    cls="column", id="iterate"
                ),
                Div(Div(generate_iter_chart(dataObj, "peakPower")),
                    Div(generate_iter_chart(dataObj, "power")),
                    Div(generate_iter_chart(dataObj, "w1")),
                    Div(generate_iter_chart(dataObj, "p1")),
                    Div(generate_iter_chart(dataObj, "nPulse")),
                     id="iterCharts") if full else Div()
            ]), 
            id="iterateFull"
        )
    # except:
    #     return  "Error in data"
