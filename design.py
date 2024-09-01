from cavity import CavityData

def generate_design(data_obj):

    if data_obj is None:
        return "No design data"
    
    cavity: CavityData = data_obj['cavityData']

    return cavity.render()