import hashlib
from cavity import CavityDataPartsKerr
from multi_mode import MultiModeSimulation
from calc import CalculatorData

class Data_object():
    def __init__(self, id):
        self.id = id
        self.count = 0
        self.run_state = False
        self.current_cavity_name = ""
        self.user_type = 0
        self.pass_hash = b'J\xdcv_\xf0Ge\x83\x1f\xc2\x9e\xa5\xd6\xbd\xd7\xa1'

    def authenticate(self, password):
        res = hashlib.md5(password.encode()).digest()
        print(res)
        if res == self.pass_hash:
            self.user_type = 1
            return True
        else:
            self.user_type = 0
        return False

    def assure(self, part):
        print(f"Assuring part: {part} for Data_object with id: {self.id}")
        match part:
            case 'cavityData':
                if not hasattr(self, 'cavityData'):
                    self.cavityData = CavityDataPartsKerr()
                    self.iterationRuns = []
                    self.seed = 0
            case 'mmData':
                if not hasattr(self, 'mmData'):
                    self.mmData = MultiModeSimulation()
            case 'calcData':
                if not hasattr(self, 'calcData') or self.calcData is None:
                    print("Creating new CalculatorData instance")
                    self.calcData = CalculatorData()

gen_data = {}

def insert_data_obj(id, obj):
    global gen_data
    gen_data[id] = obj

def clear_data_obj(id):
    dataObj = get_Data_obj(id)
    if dataObj.user_type == 1:
        global gen_data
        gen_data = {}

def get_Data_obj(id):
    global gen_data
    if id not in gen_data.keys():
        data_obj = Data_object(id)
        insert_data_obj(id, data_obj)
        return data_obj
    return gen_data[id]
        
def get_sim_obj(id):
    dataObj = get_Data_obj(id)
    dataObj.assure('cavityData')
    return dataObj.cavityData

def get_data_keys():
    global gen_data
    return list(gen_data.keys())


