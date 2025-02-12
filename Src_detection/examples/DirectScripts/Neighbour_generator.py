import sys, os
import numpy as np

#Line needed to work with ovito graphical mode
sys.path.append(os.getcwd())

sys.path.insert(0,'../../')
from Src import NeighboursCoordinates


def print_correctly_arrays(array : np.ndarray) -> None : 
    """Litte function to plot coorectly array"""
    full_str = ''
    for line in array : 
        str_line = '['
        for el in line : 
            str_line += f' {el:.5f},'
        str_line = f'{str_line[:-1]}],\n'
        full_str += str_line
    print(f'[{full_str[:-2]}]')

list_lattice2do = ['BCC','FCC','HCP']
for lattice in list_lattice2do : 
    print(f'... Computing lattice {lattice} ...')
    lattice_neigh_coordinates = NeighboursCoordinates(lattice)
    dic_shell, neigh = lattice_neigh_coordinates.BuildNeighboursCoordinates(Nneigh=80,
                                                                            shell=3)
    #print(f'Shells : {dic_shell}')
    print(f'{[ (key,len(dic_shell[key])) for key in dic_shell.keys()]}')
    print(f'Full number of neighbours up to shell : {np.sum([ len(dic_shell[key]) for key in dic_shell.keys()])}')
    print(f'Array : {neigh}')
    print()