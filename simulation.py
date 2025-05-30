import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
from fasthtml.common import *
from controls import *


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
                Div(*[Div(generate_chart([chart.x], [chart.y], [""], chart.name), cls="box", style="background-color: #008080;") for chart in charts],
                    cls="rowx"
                ))

            , cls="column"
        )
    # except:
    #     return  "Error in data"