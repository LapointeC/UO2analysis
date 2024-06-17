"""
This module contains functionality for reading and writing an ASE
Atoms object in Milday format 

"""

import re, os
import csv

import numpy as np

from ase import Atoms
from ase.utils import reader, writer
from ase.io.utils import ImageIterator
from ase.io import ParseError
from pathlib import Path
from typing import Union, List, Dict, Tuple

__all__ = [
    'read_milady_poscar', 'read_milady_descriptor', 'write_milady_poscar','read_database_milady'
]


mass_el = {"H":1.008,"He":4.003,"Li":6.941,"Be":9.012,"B":10.811,"C":12.011,"N ":14.007,"O":15.999,"F":18.998,
          "Ne":20.180,"Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.086,"P":30.974,"S":32.065,"Cl":35.453,"Ar":39.948,
          "K":39.098,"Ca":40.078,"Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,"Fe":55.845,"Co":58.933,
          "Ni":58.693,"Cu":63.546,"Zn":65.390,"Ga":69.723,"Ge":72.640,"As":74.922,"Se":78.960,"Br":79.904,"Kr":83.800,
          "Rb":85.468,"Sr":87.620,"Y":88.906,"Zr":91.224,"Nb":92.906,"Mo":95.940,"Tc":98.000,"Ru":101.070,"Rh":102.906,
          "Pd":106.420,"Ag":107.868,"Cd":112.411,"In":114.818,"Sn":118.710,"Sb":121.760,"Te":127.600,"I":126.905,"Xe":131.293,
          "Cs":132.906,"Ba":137.327,"La":138.906,"Ce":140.116,"Pr":140.908,"Nd":144.240,"Pm":145.000,"Sm":150.360,"Eu":151.964,
          "Gd":157.250,"Tb":158.925,"Dy":162.500,"Ho":164.930,"Er":167.259,"Tm":168.934,"Yb":173.040,"Lu":174.967,"Hf":178.490,
          "Ta":180.948,"W":183.840,"Re":186.207,"Os":190.230,"Ir":192.217,"Pt":195.078,"Au":196.967,"Hg":200.590,"Tl":204.383,
          "Pb":207.200,"Bi":208.980,"Po":209.000,"At":210.000,"Rn":222.000,"Fr":223.000,"Ra":226.000,"Ac":227.000,"Th":232.038,
          "Pa":231.036,"U":238.029,"Np":237.000,"Pu":244.000,"Am":243.000,"Cm":247.000,"Bk":247.000,"Cf":251.000,"Es":252.000,
          "Fm":257.000,"Md":258.000,"No":259.000,"Lr":262.000,"Rf":261.000,"Db":262.000,"Sg":266.000,"Bh":264.000,"Hs":277.000,"Mt":268.000}

def _symbol_count_from_symbols(symbols):
    """Reduce list of chemical symbols into compact VASP notation

    args:
        symbols (iterable of str)

    returns:
        list of pairs [(el1, c1), (el2, c2), ...]
    """
    sc = []
    psym = symbols[0]
    count = 0
    for sym in symbols:
        if sym != psym:
            sc.append((psym, count))
            psym = sym
            count = 1
        else:
            count += 1
    sc.append((psym, count))
    return sc

def get_atomtypes_from_formula(formula : str) -> list[str]:
    """Return atom types from chemical formula (optionally prepended
    with and underscore).
    """
    from ase.symbols import string2symbols
    symbols = string2symbols(formula.split('_')[0])
    atomtypes = [symbols[0]]
    for s in symbols[1:]:
        if s != atomtypes[-1]:
            atomtypes.append(s)
    return atomtypes

@reader
def read_milady_poscar(filename : str ='00_000_000001.poscar') -> Atoms : 
    """Import MILADY type file.

    Reads unitcell, atom positions and constraints from the POSCAR/CONTCAR
    file and tries to read atom types from POSCAR/CONTCAR header, if this fails
    the atom types are read from OUTCAR or POTCAR file.
    """

    fd = filename
    lines = fd.readlines()

    scale_lattice = float(lines[1].split()[0])

    # Now the lattice vectors
    a = []
    for ii in range(3):
        s = lines[2+ii].split()
        floatvect = float(s[0]), float(s[1]), float(s[2])
        a.append(floatvect)

    basis_vectors = np.array(a) * scale_lattice
    # Chemical symbols
    atom_symbols = []
    atomtypes = lines[5].split()
    numofatoms = lines[6].split()
    for i, num in enumerate(numofatoms):
        numofatoms[i] = int(num)
        [atom_symbols.append(atomtypes[i]) for na in range(numofatoms[i])]   

    # Filling atomic coordinate (Milday format is always written in cartesian coordinates)
    tot_natoms = np.sum(numofatoms)
    atoms_pos = np.empty((tot_natoms,3))
    atoms_forces = np.empty((tot_natoms,3))

    for atom in range(tot_natoms) : 
        coord = lines[8+atom].split()
        forces = lines[9+tot_natoms+atom].split() 
        atoms_pos[atom] = (float(coord[0]), float(coord[1]), float(coord[2]))
        atoms_forces[atom] = (float(forces[0]), float(forces[1]), float(forces[2]))
    
    atoms = Atoms(symbols=atom_symbols, cell=basis_vectors, pbc=True)
    atoms.set_positions(atoms_pos)
    atoms.set_array('forces', atoms_forces, dtype=float)

    return atoms

@writer
def write_milady_poscar(filename : str, 
                        atoms : Atoms, 
                        energy : float = None,
                        forces : np.ndarray = None,
                        stress : np.ndarray = None) -> None : 
    """Method to write Milady poscar files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.
    """  

    fd = filename

    # Create a list sc of (symbol, count) pairs
    symbols = atoms.get_chemical_symbols()
    sc = _symbol_count_from_symbols(symbols)
    first_line = '111 %2d '%(len(sc))
    for pairs in sc :
        first_line += '%s %3d '%(pairs[0],int(round(mass_el[pairs[0]],0)))
    if energy is None : 
        energy = 0.0
    first_line += '%4.9f %4.9f %4.9f \n'%(energy,0.0,0.0)
    fd.write(first_line)
    fd.write('1.0000000\n')
    # Cell writing
    cell = atoms.cell[:]
    for i in range(len(cell)):
        fd.write('%8.9f %8.9f %8.9f \n'%(cell[i,0],cell[i,1],cell[i,2]))
    # Element lines writing
    line_el = ''
    line_nb_at = ''
    for pairs in sc :
        line_el += '%s '%(pairs[0])
        line_nb_at += '%3d '%(pairs[1])
    fd.write('%s\n'%(line_el))
    fd.write('%s\n'%(line_nb_at))
    fd.write('Cartesian \n')
    # Positions writing
    atomic_position = atoms.get_positions()
    for j in range(len(atomic_position)):
        fd.write('%8.9f %8.9f %8.9f \n'%(atomic_position[j,0],atomic_position[j,1],atomic_position[j,2]))
    fd.write('\n')
    # Forces writing
    if forces is None :
        forces = np.zeros(atomic_position.shape) 
    for k in range(forces.shape[0]):
        fd.write('%8.9f %8.9f %8.9f \n'%(forces[k,0],forces[k,1],forces[k,2]))
    fd.write('\n')
    #Stress writing 
    if stress is None : 
        stress = np.zeros(6)
    fd.write('%5.5f %5.5f %5.5f %5.5f %5.5f %5.5f \n'%(stress[0],stress[1],stress[2],stress[3],stress[4],stress[5]))
    fd.write('\n')
    fd.write('0')


def read_database_milady(pathway : str) -> Tuple[Union[List[Atoms],Atoms], Dict[str,any]] :
    """Generate Atoms | List[Atoms] obejct from a given database of milady POSCAR
    
    Parameters: 
    ----------

    pathway: str
        Path to a directory or a file 

    Returns:
    --------

        Union[List[Atoms], Atoms]
            Atoms (if pathway is a file) or List[Atoms] (if pathway is a directory) associated to the database in pathway
    """

    dic_dbmanager = {}

    if os.path.isdir(pathway) : 
        list_atoms : List[Atoms] = []
        all_file_mld = ['%s/%s'%(pathway,f) for f in os.listdir(pathway)]
        for f_mld in all_file_mld : 
            atoms_mld = read_milady_poscar(f_mld)
            list_atoms.append(atoms_mld)
            dic_dbmanager[os.path.basename(f_mld)] = {'atoms':atoms_mld,'energy':None,'forces':atoms_mld.get_array('forces'),'stress':None}
        return list_atoms, dic_dbmanager
    
    elif os.path.isfile(pathway) : 
        atoms_mld = read_milady_poscar(pathway)
        dic_dbmanager[pathway] = {'atoms':atoms_mld,'energy':None,'forces':atoms_mld.get_array('forces'),'stress':None}
        return atoms_mld, dic_dbmanager
    
    else : 
        raise TypeError('The specified path is not a directory or a file...')


def read_milady_descriptor(filename : str,
                           ext : str = 'eml') -> np.ndarray : 
    if ext == 'eml' : 
        return np.loadtxt('{:}.eml'.format(filename))[:,1:]

    elif ext == 'csv' : 
        with open('{:}.csv'.format(filename),'r') as f : 
            return np.array(list(csv.reader(f)))[:,1:]

    elif ext == 'npz' : 
        return np.load(filename)
    
def fill_milady_descriptor(atoms : Atoms, 
                           file_descriptors : str, 
                           name_property : str = 'milady-descriptors',
                           ext : str = 'eml') -> Atoms : 
    milady_descriptors = read_milady_descriptor(file_descriptors, ext=ext)
    atoms.set_array(name_property, milady_descriptors, dtype = float)
    return atoms