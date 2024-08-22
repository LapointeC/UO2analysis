import numpy as np
import time
from functools import wraps

import ase
import re

from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import reader
from typing import List

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        #print(f'Function {func.__name__} results : {result}')
        return result
    return timeit_wrapper

def check_format(list_str : List[str]) -> List[str] : 
    new_list_str = []
    for str in list_str :
        new_str = re.sub('D','E',str)
        new_list_str.append(new_str)
    return new_list_str

@reader
def my_cfg_reader(fd, extended_properties : List[str] = None) -> Atoms : 
    """Read atomic configuration from a CFG-file (native AtomEye format).
       See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    nat = None
    naux = 0
    aux = None
    auxstrs = None

    cell = np.zeros([3, 3])
    transform = np.eye(3)
    eta = np.zeros([3, 3])

    current_atom = 0
    current_symbol = None
    current_mass = None

    if extended_properties is not None : 
        dic_extend_properties = {prop:None for prop in extended_properties}
    
    L = fd.readline()
    while L:
        L = L.strip()
        if len(L) != 0 and not L.startswith('#'):
            if L == '.NO_VELOCITY.':
                vels = None
                naux += 3
            else:
                s = L.split('=')
                if len(s) == 2:
                    key, value = s
                    key = key.strip()
                    value = [x.strip() for x in value.split()]
                    if key == 'Number of particles':
                        nat = int(value[0])
                        spos = np.zeros([nat, 3])
                        masses = np.zeros(nat)
                        syms = [''] * nat
                        vels = np.zeros([nat, 3])
                        if extended_properties is not None : 
                            for prop in dic_extend_properties.keys() : 
                                dic_extend_properties[prop] = np.zeros(nat)

                        if naux > 0:
                            aux = np.zeros([nat, naux])
                    elif key == 'A':
                        pass  # unit = float(value[0])
                    #elif key == 'entry_count':
                    #    naux += int(value[0]) - 6
                    #    auxstrs = [''] * naux
                    #    if nat is not None:
                    #        aux = np.zeros([nat, naux])
                    elif key.startswith('H0('):
                        i, j = [int(x) for x in key[3:-1].split(',')]
                        cell[i - 1, j - 1] = float(value[0])
                    elif key.startswith('Transform('):
                        i, j = [int(x) for x in key[10:-1].split(',')]
                        transform[i - 1, j - 1] = float(value[0])
                    elif key.startswith('eta('):
                        i, j = [int(x) for x in key[4:-1].split(',')]
                        eta[i - 1, j - 1] = float(value[0])
                    #elif key.startswith('auxiliary['):
                    #    i = int(key[10:-1])
                    #    auxstrs[i] = value[0]
                else:
                    # Everything else must be particle data.
                    # First check if current line contains an element mass or
                    # name. Then we have an extended XYZ format.
                    s = [x.strip() for x in L.split()]
                    if len(s) == 1:
                        if L in chemical_symbols:
                            current_symbol = L
                        else:
                            current_mass = float(L)
                    elif current_symbol is None and current_mass is None:
                        # Standard CFG format
                        masses[current_atom] = float(s[0])
                        syms[current_atom] = s[1]
                        spos[current_atom, :] = [float(x) for x in s[2:5]]
                        vels[current_atom, :] = [float(x) for x in s[5:8]]
                        current_atom += 1
                    elif (current_symbol is not None and
                          current_mass is not None):
                        # Extended CFG format
                        masses[current_atom] = current_mass
                        syms[current_atom] = current_symbol
                        props = [float(x) for x in check_format(s)]
                        spos[current_atom, :] = props[0:3]
                        if extended_properties is not None : 
                            for id_prop, prop in enumerate(dic_extend_properties.keys()) : 
                                dic_extend_properties[prop][current_atom] = props[3+id_prop]
                        off = 3
                        if vels is not None:
                            off = 6
                            vels[current_atom, :] = props[3:6]
                        #aux[current_atom, :] = props[off:]
                        current_atom += 1
        L = fd.readline()

    # Sanity check
    if current_atom != nat:
        raise RuntimeError('Number of atoms reported for CFG file (={0}) and '
                           'number of atoms actually read (={1}) differ.'
                           .format(nat, current_atom))

    if np.any(eta != 0):
        raise NotImplementedError('eta != 0 not yet implemented for CFG '
                                  'reader.')
    cell = np.dot(cell, transform)

    if vels is None:
        a = ase.Atoms(
            symbols=syms,
            masses=masses,
            scaled_positions=spos,
            cell=cell,
            pbc=True)
    else:
        a = ase.Atoms(
            symbols=syms,
            masses=masses,
            scaled_positions=spos,
            momenta=masses.reshape(-1, 1) * vels,
            cell=cell,
            pbc=True)

    if extended_properties is not None : 
        for prop in dic_extend_properties.keys() : 
            a.set_array(prop,dic_extend_properties[prop])

    return a
