import numpy as np

class CalcCommon:
    def __init__(self):
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
    if arr.size <= 40000:
        return arr
    
    rc = arr.reshape(max_size, arr.size // max_size).max(axis=1)
    return rc

def cget(x):
    return x.get() if hasattr(x, "get") else x