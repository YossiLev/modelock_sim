gen_data = {}

def insert_data_obj(id, obj):
    global gen_data
    gen_data[id] = obj

def get_Data_obj(id):
    global gen_data
    if id not in gen_data.keys():
        return None
    return gen_data[id]
        
def get_sim_obj(id):
    dataObj = get_Data_obj(id)
    if dataObj is None:
        return None
    return dataObj['cavityData']

def get_data_keys():
    global gen_data
    return list(gen_data.keys())

