from fasthtml.common import *
from cavity import CavityData

def generate_design(data_obj):

    if data_obj is None:
        return Div(
            Div("No simulation loaed please load"), cls="rowx", id="cavity" )
    
    cavity: CavityData = data_obj['cavityData']

    return cavity.render()