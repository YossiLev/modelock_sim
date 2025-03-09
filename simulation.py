import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fasthtml.common import *
from controls import *


def generate_all_charts(dataObj):
    try:
        if dataObj is None:
            return  "No data"
        count = dataObj['count']
        seed = dataObj['seed']
        charts = dataObj['cavityData'].get_state()

        return Div(
            Div(f"Seed {seed} - Step {count}", id="count"),
            Div(
                Div(*[p.render() for p in dataObj['cavityData'].getPinnedParameters(1)], cls="rowx"), 
                Div(*[Div(generate_chart([chart.x], [chart.y], [""], chart.name), cls="box", style="background-color: #008080;") for chart in charts],
                    cls="rowx"
                ))

            , cls="column"
        )
    except:
        return  "Error in data"