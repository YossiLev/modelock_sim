from cavity import CavityData

def generate_design(data_obj):

    cavity: CavityData = data_obj['cavityDataParts']
    print("----- ", len(cavity.parameters))

    return cavity.render()