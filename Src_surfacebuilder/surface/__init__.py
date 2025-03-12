from .ase import * 
from .hpc import *
from .parser import * 
from .slab_builder import *
from .surface_energy import *
from .tools import *

def print_logo() -> None :
    """
        A friendly welcome message :)
    """
    welcome = f"""

    ┏┓    ┏     ┳┓  •┓ ┓    
    ┗┓┓┏┏┓╋┏┓┏┏┓┣┫┓┏┓┃┏┫┏┓┏┓
    ┗┛┗┻┛ ┛┗┻┗┗ ┻┛┗┻┗┗┗┻┗ ┛ 
                        
    copyright CEA by ...                                        
    ... C. Lapointe                               
    email: clovis.lapointe@cea.fr"""
    print(welcome)
    print()
    return

print_logo()