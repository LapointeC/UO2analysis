import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

from ase import Atoms
from typing import TypedDict, List, Dict

class PCA_(TypedDict) : 
    PCA : PCA

class PCAModel :
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.models : Dict[str, PCA_] = {}

    def _get_pca_model(self, list_atoms : List[Atoms], species : str, n_component : int = 2) -> np.ndarray : 
        """Build PCA model from data"""
        self.models[species] = {'PCA':None}
        self.models[species]['PCA'] = PCA(n_components=n_component)
        descriptors_array = np.concatenate([ atoms.get_array('milady-descriptors') for atoms in list_atoms ], axis=0)
        return self.models[species]['PCA'].fit_transform(descriptors_array)
    
    def _write_pkl(self, path_writing : os.PathLike[str] = './mcd.pkl') -> None : 
        """Write pickle file for ```MCDModel``` object
        
        Parameters
        ----------

        path_writing : os.PathLike[str]
            Writing path for pickle file
        """
        pickle.dump(self, open(path_writing,'w'))
        return 

    def _load_pkl(self, path_reading : os.PathLike[str]) -> None : 
        """Reading previous pickle file ```MCDModel```
        
        Parameters
        ----------

        path_reading : os.PathLike[str] 
            Readning path for previous pickle file
        """
        pkl = pickle.load(open(path_reading,'rb'))
        self.__dict__.update(pkl.__dict__)
        return 