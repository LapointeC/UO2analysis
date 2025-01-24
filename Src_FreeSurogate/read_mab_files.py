import glob
import numpy as np
import os, re, shutil
import pickle
import warnings

from typing import Dict, TypedDict, List
from tools import RecursiveBuilder

from ase import Atoms
from ase.io import read

import h5py
from h5py import Group
from re import Pattern


class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    atoms : Atoms

def add_or_update_FE(hdf5_group : Group, 
                     name_config : str,
                     data_object : Data) -> None :
    fe_group_name = name_config
    cell = data_object['atoms'].cell[:]
    positions = data_object['atoms'].positions

    if fe_group_name not in hdf5_group:
        # Create group for the dynamical matrix if it doesn't exist
        fe_group = hdf5_group.create_group(fe_group_name)
        fe_group.create_dataset("positions", data=positions, compression="gzip", compression_opts=9)
        fe_group.create_dataset("cell", data=cell, compression="gzip", compression_opts=9)
        fe_group.create_dataset("temperature", data=data_object['array_temperature'], compression="gzip", compression_opts=9)
        fe_group.create_dataset("ref_FE", data=data_object['array_ref_FE'], compression="gzip", compression_opts=9)
        fe_group.create_dataset("anha_FE", data=data_object['array_anah_FE'], compression="gzip", compression_opts=9)
        fe_group.create_dataset("full_FE", data=data_object['array_full_FE'], compression="gzip", compression_opts=9)       
        fe_group.create_dataset("sigma_FE", data=data_object['array_sigma_FE'], compression="gzip", compression_opts=9)
        fe_group.create_dataset("delta_FE", data=data_object['array_delta_FE'], compression="gzip", compression_opts=9)

    else : 
        hdf5_group[name_config].create_dataset("positions", data=positions, compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("cell", data=cell, compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("temperature", data=data_object['array_temperature'], compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("ref_FE", data=data_object['array_ref_FE'], compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("anha_FE", data=data_object['array_anah_FE'], compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("full_FE", data=data_object['array_full_FE'], compression="gzip", compression_opts=9)       
        hdf5_group[name_config].create_dataset("sigma_FE", data=data_object['array_sigma_FE'], compression="gzip", compression_opts=9)
        hdf5_group[name_config].create_dataset("delta_FE", data=data_object['array_delta_FE'], compression="gzip", compression_opts=9)
    return 

class DataMAB : 
    def __init__(self, root_dir : os.PathLike[str],
                 list_temperature : List[float]) -> None : 
        self.Data : Dict[str,Data] = {}
        self.root_dir = root_dir
        self.list_temperature = list_temperature

    def GenerateData(self, hdf5 : Group,
                     nb_langevin : float,
                     alpha : float = 2.0) -> None : 
        list_all_calculations = RecursiveBuilder(self.root_dir, file2find='out_mab')
        gather_dictionnary = self.GatheringPath(list_all_calculations)
        print(f'... I found {len(gather_dictionnary)} configurations ...')
        print()
        for name, collection_path_name in gather_dictionnary.items() : 
            print('Extracting data : {:s}'.format(name))
            self.UpdateData(hdf5,
                            name,
                            collection_path_name,
                            nb_langevin,
                            alpha)

    def GatheringPath(self, list_path : List[os.PathLike[str]]) -> Dict[str,List[os.PathLike[str]]] :
        gather_dic : Dict[str, List[os.PathLike[str]]] = {}
        for path in list_path : 
            name_calculation = path.split('/')[-2]
            if name_calculation not in gather_dic.keys() : 
                gather_dic[name_calculation] = [path]
            else : 
                gather_dic[name_calculation].append(path)

        return gather_dic

    def change_all_symbols(self, atoms : Atoms, symbol : str) -> Atoms : 
        atoms.set_chemical_symbols([symbol] * len(atoms))
        return atoms

    def read_inputs_lammps(self, path : str) -> str : 
        return open(path,'r').read()

    def extract_value(self, patern : Pattern[str], string : str) -> float : 
        try : 
            return float(patern.search(string).group(1))
        except : 
            return np.nan

    def ReadOutMab(self, path_out : os.PathLike[str], 
                   nb_langevin : float, 
                   alpha : float,
                   temperature : float) -> np.ndarray :
        try : 
            str_FE = open(path_out,'r').read()
            header = re.compile(r'----------FREE ENERGY FINAL RESULTS---------')
            header_str = header.search(str_FE)
        except :
            print('... No free energy results ...')
            return np.array([temperature, np.nan, np.nan, np.nan, np.nan, np.nan])

        reference_pattern = re.compile(r'F\(Reference\)\s+\(eV\)\s+\.+:\s+([-\d\.]+)')
        full_ref_pattern = re.compile(r'F\(Full\) - F\(Reference\)\s+\(eV\)\s+\.+:\s+([-\d\.]+)')
        full_pattern = re.compile(r'F\(Full\)\s+\(eV\)\s+\.+:\s+([-\d\.]+)')
        sigma_pattern = re.compile(r'sigma\(F\(full\)\)\s+\(eV\)\s+\.+:\s+([-\d\.]+)')
        delta_pattern = re.compile(r'F\(Full\) - F\(Reference\) - F\(block\)\s+\(eV\)\s+\.+:\s+([-\d\.]+)')

        # Extract data using the regular expressions
        reference_value = self.extract_value(reference_pattern, str_FE) #reference_pattern.search(str_FE).group(1)
        full_ref_value =  self.extract_value(full_ref_pattern, str_FE) #full_ref_pattern.search(str_FE).group(1)
        full_value = self.extract_value(full_pattern, str_FE) #full_pattern.search(str_FE).group(1)
        sigma_value =  alpha*self.extract_value(sigma_pattern, str_FE)/np.sqrt(nb_langevin) # alpha*sigma_pattern.search(str_FE).group(1)/np.sqrt(nb_langevin)
        delta_F_value = self.extract_value(delta_pattern, str_FE)

        return np.array([temperature ,reference_value, full_ref_value, full_value, sigma_value, delta_F_value])

    def UpdateData(self,
                   hdf5 : Group, 
                   name : str, 
                   collection_path : List[os.PathLike[str]], 
                   nb_langevin : float, 
                   alpha : float) -> None :
        collection_path.sort()
        array_data = np.zeros((len(self.list_temperature),6))
        root_path = os.path.dirname(collection_path[0])
        for id_temperature, temperature in enumerate(self.list_temperature) :
            path_temperature = '{:s}/{:4.1f}'.format(root_path,temperature) 
            #print(path_temperature)            
            array_data[id_temperature] = self.ReadOutMab(f'{path_temperature}/out_mab',
                                                         nb_langevin,
                                                         alpha,
                                                         temperature)
        atoms = read('{:s}/in.lmp'.format(collection_path[0]),format='lammps-data',style='atomic')
        atoms = self.change_all_symbols(atoms, 'Fe')

        #updating Data object 
        self.Data[name] = {'atoms':atoms,
                           'array_temperature':array_data[:,0],
                           'array_ref_FE':array_data[:,1],
                           'array_anah_FE':array_data[:,2],
                           'array_full_FE':array_data[:,3],
                           'array_sigma_FE':array_data[:,4],
                           'array_delta_FE':array_data[:,5]}
        print(self.Data[name])

        add_or_update_FE(hdf5, 
                         name,
                         self.Data[name])


    def WritePickle(self, path2write : str = '.') -> None :
        if os.path.exists('{:s}/mab.pickle'.format(path2write)) : 
            os.remove('{:s}/mab.pickle'.format(path2write))

        if self.Data is None : 
            warnings.warn('Data dictionnary is empty !')
        pickle.dump(self.Data, open('{:s}/mab.pickle'.format(path2write),'wb'))

####################
### INPUTS
####################
root_path = '/lustre/fsn1/projects/rech/yxs/uef77fk/FreeEnergyFe'
nb_langevin = 40000.0 
alpha = 5.0
list_temperature = [300., 600., 900., 1200.]
####################

mab_object = DataMAB(root_path,
                     list_temperature)

if os.path.exists('free_energy.h5') :
    os.remove('free_energy.h5')

file = h5py.File('free_energy.h5', 'w')
# Create a group for potentials
free_energy_group = file.create_group('free_energy')
mab_object.GenerateData(free_energy_group,
                        nb_langevin,
                        alpha=alpha)
mab_object.WritePickle()
