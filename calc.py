try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
    else:
        import numpy as np
except ImportError:
    import numpy as np
from fasthtml.common import *
from controls import *
from multi_mode import cget

class CalculatorData:
    def __init__(self):
        print("CalculatorData init")
        self.M1 = [[1, 0], [0, 1]]
        self.M2 = [[1, 0], [0, 1]]
        self.M3 = [[1, 0], [0, 1]]
        self.M4 = [[1, 0], [0, 1]]
        self.M5 = [[1, 0], [0, 1]]

    def set(self, params):
        print(params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, np.asarray(value))

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
        return params
    
def InputCalcS(id, title, value, step=0.01, width = 50):
    return Input(type="number", id=id, title=title, 
                 value=value, step=f"{step}", 
                 hx_trigger="input changed delay:1s", hx_post="/clUpdate", hx_include="#calcForm *", 
                 hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;")

def ABCDMatControl(name, M):
    return Div(
        Div(
            InputCalcS(f'{name}_A', "A", f'{M[0][0]}', width = 80),
            InputCalcS(f'{name}_B', "B", f'{M[0][1]}', width = 80),
        ),
        Div(
            InputCalcS(f'{name}_C', "C", f'{M[1][0]}', width = 80),
            InputCalcS(f'{name}_D', "D", f'{M[1][1]}', width = 80),
            Button("X", onclick=f"removeElement('{name}')"),
        ),
        cls="ABCDMatControl"
    )
def generate_calc(data_obj, tab, offset = 0):

    if data_obj is None:
        return Div()
    
    if "calcData" not in data_obj or data_obj["calcData"] is None:
        print("Creating new calcData")
        data_obj["calcData"] = CalculatorData()
        print("calcData created")
        print(data_obj["calcData"])
        a = CalculatorData()
        print(a)

    calcData = data_obj["calcData"]
    added = Div()

    match tab:
        case 1:
            added = Div(
                ABCDMatControl("M1", calcData.M1),
                ABCDMatControl("M2", calcData.M2),
                ABCDMatControl("M3", calcData.M3),
            )
        case 2: 
            pass
        case 3: 
            pass
        case 4: 
            pass
        case 5: # MultiTime on server
            pass

    return Div(
        Div(
            TabMaker("Matrix", "/tabcalc/1", tab == 1),
            TabMaker("TBD1", "/tabcalc/2", tab == 2),
            TabMaker("TBD2", "/tabcalc/3", tab == 3),
            TabMaker("TBD3", "/tabtabcalcfun/4", tab == 4),
            TabMaker("TBD4", "/tabcalc/5", tab == 5),
        ),
        Div(added, id="calcForm"),

        id="gen_calc"
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


