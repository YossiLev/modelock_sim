from cavity import CavityDataPartsKerr, CavityData
from multi_mode import MultiModeSimulation
from calc import CalculatorData

class Data_object():
    def __init__(self, id):
        self.id = id
        self.count = 0
        self.run_state = False

    def assure(self, part):
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
                if not hasattr(self, 'calcData'):
                    self.calcData = CalculatorData()

gen_data = {}

def insert_data_obj(id, obj):
    global gen_data
    gen_data[id] = obj

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


