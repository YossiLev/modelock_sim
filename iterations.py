import datetime
import numpy as np
from fasthtml.common import *

class Iteration():
    def __init__(self, sim, seed, parameter, value_start, value_end, n_values, values_mode, name = f"General"):
        self.sim = sim
        self.seed = seed
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
        self.state = ['---------------' for v in self.values]
        self.reports = [[] for v in self.values]
        self.reportsFinal = ["Wait.." for v in self.values]
        self.seeds = [-1 for v in self.values]

        self.current_index = 0
        self.current_count = 0
        self.max_count = 1500

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
                    part.finalize()
                    self.sim.finalize()
                self.sim.simulation_step()
                self.current_count += 1
                if self.current_count % 100 == 0:
                    analysis = self.sim.get_state_analysis()
                    self.reports[self.current_index].append(analysis)
                    analysisP = {k: f"{v:.2e}" if isinstance(v,float) else v for k,v in analysis.items()}
                    self.reportsFinal[self.current_index] = str(analysisP)
                    *state_list, = self.state[self.current_index]
                    state_list[round(self.current_count / 100) - 1] = analysis['state_code']
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
            Table(
                Tr(Th(self.parameterName), Th("Seed"), Th("State", style="min-width:140px;"), Th("Report")),
                *[Tr(Td(f"{value:.1f}", cls="monoRight"), Td(f"{seed}", cls="monoRight"), Td(state, cls="mono", style="min-width:140px;"), Td(report)) for value, seed, state, report in zip(self.values, self.seeds, self.state, self.reportsFinal)]
            )
        )

def generate_iterations(dataObj, iterationIndex = - 1):
    try:
        if dataObj is not None and len(dataObj['iterationRuns']) > 0:
            iteration = dataObj['iterationRuns'][-1]
        else:
            iteration = None

        if iteration is not None:
            counts = f"Seed {iteration.seed} - Index {iteration.current_index} - Step {iteration.current_count}"
        else:
            counts = ""
        return Div(
           
            Div(counts, id="iter_count") ,
            Div(Div(*[p.render() for p in dataObj['cavityData'].getPinnedParameters()], cls="rowx"), cls="rowx"),
            iteration.render() if iteration is not None else Div("no values"),
            cls="column", id="iterate"
        )
    except:
        return  "Error in data"
