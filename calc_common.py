import numpy as np

from controls import cget

class CalcCommon:
    def __init__(self):
        self.current_run_parameters = []
        pass

    def set(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                if type(value) is list:
                    setattr(self, key, np.asarray(value))
                else:
                    setattr(self, key, value)

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
        return params

    def keep_current_run_parameters(self, parameters):
        self.current_run_parameters = parameters.copy()

    def verify_current_run_parameters(self, parameters):
        return self.current_run_parameters == parameters

class CalcCommonBeam(CalcCommon):
    def __init__(self):
        super().__init__()
        self.beam_view_from = -1
        self.beam_view_to = -1
        self.beam_N = 4096 # * 4
        self.beam_time = 3.95138389E-09 #4E-09
        self.beam_dt = self.beam_time / self.beam_N       
    
    def zoom_view(self, factor):
        if (self.beam_view_from == -1) or (self.beam_view_to == -1):
            self.beam_view_from = 0
            self.beam_view_to = self.beam_N
        center = (self.beam_view_from + self.beam_view_to) / 2
        half_range = (self.beam_view_to - self.beam_view_from) / 2 / factor
        self.beam_view_from = int(center - half_range)
        self.beam_view_to = int(center + half_range)

    def shift_view(self, shift_factor):
        if (self.beam_view_from == -1) or (self.beam_view_to == -1):
            self.beam_view_from = 0
            self.beam_view_to = self.beam_N
        center = (self.beam_view_from + self.beam_view_to) / 2
        half_range = (self.beam_view_to - self.beam_view_from) / 2
        shift = int(half_range * shift_factor)
        self.beam_view_from = int(center - half_range + shift)
        self.beam_view_to = int(center + half_range + shift)

    def shift_center(self):
        if (self.beam_view_from == -1) or (self.beam_view_to == -1):
            self.beam_view_from = 0
            self.beam_view_to = self.beam_N
        center = self.beam_N / 2
        half_range = (self.beam_view_to - self.beam_view_from) / 2
        self.beam_view_from = int(center - half_range)
        self.beam_view_to = int(center + half_range)                

    def shrink_def(self, arrp):
        return shrink_with_max(arrp, 1024, self.beam_view_from, self.beam_view_to)

    def shrink_list(self, arrp):
        return cget(self.shrink_def(arrp)).tolist()

    def shrink_lists(self, arrp_vec):
        return list(map(lambda x: self.shrink_list(x), arrp_vec))

    def doCalcCommand(self, params):

        match params:
            case "view":
                return 1
            case "zoomin":
                self.zoom_view(2.0)
                return 1
            case "zoomout":
                self.zoom_view(0.5)
                return 1
            case "shiftright":
                self.shift_view(-0.5)
                return 1
            case "shiftleft":
                self.shift_view(0.5)
                return 1 
            case "center":
                self.shift_center()
                return 1
        return 0

    def serialize_graphs_data(self):
        graphs = []
        graphs.append({
            "id": "diode_pulse_chart",
            "title": "Pulse in (photons/sec)",
            "x_label": "Time (s)",
            "y_label": "Intensity",
            "lines": [
                {"color": "orange", "values": cget(intens(self.diode_pulse)).tolist(), "text": f"rrrrrrrrrr"}
            ],
            # "x_values": cget(self.diode_t_list).tolist(),
            # "y_values": cget(intens(self.diode_pulse)).tolist(),
        })

        return graphs
    
    def collectCommonData(self, delay=20, more=False):
        data = {
            "type": "diode",
            "delay": delay,
            "more": more,
            "title": "Pulse in (photons/sec)",
            "graphs": self.serialize_graphs_data(),
        }
        try:
            s = json.dumps(data)
        except TypeError as e:
            print("error: json exeption", data)
            s = ""
        return s  
  
def intens(arr):
    if len(arr) == 0 or arr.dtype != np.complex128:
        return arr
    if (np.isnan(arr.real)).any():
        print("NaN in real part **** could be an error")
        print(f"arr={arr[0]},{arr[1]},{arr[2]},{arr[3]}, ")
        raise Exception("NaN in real part **** could be an error")
    return np.square(arr.real) + np.square(arr.imag)

def shrink_with_max(arrp, max_size, fromX=-1, toX=-1):
    if (fromX >= 0 and toX > fromX and toX <= arrp.size):
        arr = arrp[fromX:toX]
    else:
        arr = arrp
    if arr.size <= max_size:
        return arr
    
    rc = arr.reshape(max_size, arr.size // max_size).max(axis=1)
    return rc

def cget(x):
    return x.get() if hasattr(x, "get") else x