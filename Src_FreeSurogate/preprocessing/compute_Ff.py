import os, sys 
import numpy as np
import pickle
import scipy

from ase import Atoms
from typing import Dict, Tuple, List, TypedDict

import scipy.linalg

class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    atoms : Atoms
    stress : np.ndarray
    volume : float
    energy : float
    Ff : np.ndarray

class AddFormationFreeEnergy : 
    def __init__(self, path_pkl : os.PathLike[str],
                 path_writing : os.PathLike[str]) : 
        self.path_pkl = path_pkl
        self.path_writing = path_writing
        self.all_data = self.load_data()

        if not self.chek_bulk_integrity(self.all_data['bulk']) : 
            raise ValueError(f'Problem with bulk data {self.all_data["bulk"]["array_full_FE"]}...')

    def load_data(self) -> Dict[str,Data] : 
        return pickle.load(open(self.path_pkl, 'rb'))        

    def chek_bulk_integrity(self, data_bulk : Data) -> bool :
        if np.isnan(data_bulk['array_full_FE']).any() : 
            return False, 
        else : 
            return True

    def compute_formation_free_energy(self, data_bulk : Data,
                                      data_dfct : Data) -> np.ndarray : 
        Nbulk = float(len(data_bulk['atoms']))
        Ndfct = float(len(data_dfct['atoms']))
        return (data_dfct['array_full_FE'] + data_dfct['energy']) - Ndfct*(data_bulk['array_full_FE'] + data_bulk['energy'])/Nbulk
    
    def ComputeAllFormationFreeEnergy(self) -> None : 
        for key_dfct in self.all_data.keys() :
            array_Ff = self.compute_formation_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
            self.all_data[key_dfct]['Ff'] = array_Ff
            print(key_dfct, array_Ff)
        return 
    
    def write_pickle(self) -> None :
        pickle.dump(self.all_data, open(f'{self.path_writing}/desc_mab_Ff.pickle','wb'))
        return
    
#################################
### INPUTS
#################################
path_pkl = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2/mab.pickle'
path_write = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2'
#################################

obj_Ff = AddFormationFreeEnergy(path_pkl,
                                path_write)
obj_Ff.ComputeAllFormationFreeEnergy()
obj_Ff.write_pickle()