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