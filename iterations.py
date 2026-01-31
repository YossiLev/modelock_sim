import datetime
import numpy as np
import json
from fasthtml.common import *
from controls import *
from itertools import product


from app import get_Data_obj

class Iteration():
    def __init__(self, sim, seed, modifications, parameter, value_start, value_end, n_values, values_mode, max_count, name = f"General"):
        self.sim = sim
        self.seed = seed
        self.modifications = modifications
        self.modification_values = list(map(lambda _: "", modifications))
        self.parameterId = list(map(lambda p: p.id, parameter))
        self.parameterName = list(map(lambda p: p.name, parameter))
        self.value_start = value_start
        self.value_end = value_end
        self.n_values = n_values
        self.values_mode = values_mode
        self.name = name
        self.show = 1
        self.date_create = datetime.datetime.now().strftime("%d-%m-%Y%H:%M:%S")
        self.values = list(product(*[self.build_values(i) for i in range(len(parameter))]))
        self.clear()
        self.max_count = max_count

    def build_values(self, i):
        if self.values_mode[i] == "log":
            return list(np.exp(np.linspace(np.log(self.value_start[i]), np.log(self.value_end[i]), self.n_values[i])))
        else:
            return list(np.linspace(self.value_start[i], self.value_end[i], self.n_values[i]))
        
    def clear(self):
        for i in range(len(self.parameterId)):
            self.state = ['----------' for v in self.values]
            self.reports = [[] for v in self.values]
            self.reportsFinal = ["Wait.." for v in self.values]
            self.seeds = [-1 for v in self.values]

        self.current_index = 0
        self.current_count = 0

    
    def toggle_show(self):
        self.show = 1 - self.show

    def step(self):
        if self.current_index < len(self.values): #self.n_values[self.current_param_index]:
            if self.current_count < self.max_count:
                if self.current_count == 0:
                    if self.seed != 0:
                        seed = self.seed
                    else:
                        seed = int(np.random.rand() * (2 ** 32 - 1))
                    self.seeds[self.current_index] = seed

                    self.sim.restart(seed)
                    vs = self.values[self.current_index]
                    ps = self.parameterId
                    for vi, pi in zip(vs, ps):
                        p, part = self.sim.getParameter(pi)
                        p.set_value(str(vi))
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
        for i in range(len(self.modification_values)):
            if (len(self.modification_values[i]) > 0):
                p, _ = self.sim.getParameter(self.modifications[i].id)
                #print(F"setting value for rendering {self.modifications[i].id} as {self.modification_values[i]}")
                p.set_value(self.modification_values[i])
        return Div(
            Div(Div(*[p.render() for p in self.modifications], cls="rowx"), cls="rowx"),
            Table(
                Tr(*[Th(pn) for pn in self.parameterName], Th("Seed"), Th("State", style="min-width:140px;"), Th("Report")),
                *[Tr(*[Td(f"{v:.4f}", cls="monoRight") for v in list(value)], 
                     Td(f"{seed}", cls="monoRight"), 
                     Td(state, cls="mono", style="min-width:100px;"), 
                     Td(report, style="font-size:11px;")) for value, seed, state, report in zip(self.values, self.seeds, self.state, self.reportsFinal)]
            )
        )


    def update_modifications(self):
        for i in range(len(self.modification_values)):
            p, _ = self.sim.getParameter(self.modifications[i].id)
            self.modification_values[i] = str(p.get_value())
            #print(F"updated value {self.modifications[i].id} to {self.modification_values[i]}")
        return

def extract_paramater_value(rep, paramaterName):
    obj = json.loads(rep)
    if (paramaterName in obj):
        return float(obj[paramaterName])
    return 0.0

def generate_iter_chart(dataObj, parameterName):
    vecX = []
    vecY = []
    vecL = []
    
    for iteration in dataObj.iterationRuns:
        fullIndex = math.prod(iteration.n_values)
        skip = math.prod(iteration.n_values[1:])  
        if (iteration.current_index >= fullIndex and iteration.show != 0):
            power = list(map(lambda x: extract_paramater_value(x, parameterName), iteration.reportsFinal[0::skip]))
            vecX.append(list(map(lambda x: x[0], iteration.values[0::skip])))
            vecY.append(power)
            vecL.append(iteration.name)

    chart = Div(generate_chart(vecX, vecY, parameterName, l=vecL,  w=5, h=3))

    return chart

def generate_iterations(dataObj, full = True):
    # try:
        index = dataObj.iteration_focus if  hasattr(dataObj, 'iteration_focus') else 0
        if dataObj is not None and dataObj is not None and len(dataObj.iterationRuns) > index:
            iteration = dataObj.iterationRuns[index]
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
                    Div(
                        *list(map(lambda x: Button(x[1].name, hx_post=f"/iterChange/{x[0]}", 
                                                   hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}',
                                                   cls=("buttonH" if x[0] == index else "")), enumerate(dataObj.iterationRuns))),
                        Div(
                            Button("Update", hx_post=f"/iterUpdate/{index}", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                            Button("Delete", hx_post=f"/iterDelete/{index}", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                            Button("Clear", hx_post=f"/iterClear/{index}", hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}'),
                            Button("Hide" if (iteration.show != 0) else "Show", hx_post=f"/iterToggleShow/{index}", 
                                   hx_target="#iterateFull", hx_swap="innerHTML", hx_vals='js:{localId: getLocalId()}')
                        ) if iteration else Div(),
                    ),
                    iteration.render() if iteration is not None else Div(),
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
