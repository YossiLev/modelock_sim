
import re
from fasthtml.common import *
from controls import *
from gen_data import *

def generate_settings(data_obj):
                        
    def InputCalcS(id, title, value, step=0.01, width = 150):
        return Div(
                Div(title, cls="floatRight", style="font-size: 10px; top:-3px; right:10px;background: #e7edb8;"),
                Input(type="number", id=id, title=title,
                    value=value, step=f"{step}", 
                    # hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
                    # hx_vals='js:{localId: getLocalId()}',
                    style=f"width:{width}px; margin:2px;"),
                style="display: inline-block; position: relative;"
        )
    def InputCalcM(id, title, value, step=0.01, width = 150):
        return Div(
                Div(title, cls="floatRight", style="font-size: 10px; top:-3px; right:10px;background: #e7edb8;"),
                Input(type="number", id=id, title=title,
                    value=value, step=f"{step}", 
#                    hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
#                    hx_vals='js:{localId: getLocalId()}',
                    style=f"width:{width}px; margin:2px;",
                    **{'onkeyup':f"validateMat(event);",
                       'onpaste':f"validateMat(event);",}),
                style="display: inline-block; position: relative;"
        )
    dev = "cpu"
    try:
        import cupy
        if cupy.cuda.is_available():
            dev = "Cuda GPU"
    except ImportError:
        pass

    added = Div(
        Div(f"Processor used {dev}"),
        Div(Span(f"Server sessions {len(get_data_keys())}"), 
            Span(Button("Delete All", escapse=False, hx_post=f'/settings/delete/0', hx_target="#gen_settings", hx_vals='js:{localId: getLocalId()}'))
        ),
        Div(Button("Restart site", escapse=False, hx_post=f'/settings/restart/0', hx_target="#gen_settings", hx_vals='js:{localId: getLocalId()}'))
    )
    return Div(
        Div(added, id="settingsForm"),

        id="gen_settings"
    )


