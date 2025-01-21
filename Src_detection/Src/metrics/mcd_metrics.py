import os
import pickle
import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import KernelDensity

from ..tools import timeit

from ase import Atoms
from typing import TypedDict, List, Dict

class MCD(TypedDict) : 
    mcd : MinCovDet
    distribution : KernelDensity

class MCDModel :
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.models : Dict[str, MCD] = {}
            self.name = 'MCD'

    def _update_name(self, name : str) -> None : 
        """Update name of ```MCDModel```
        
        Parameters
        ----------

        name : str
            New name for ```MCDModel```
        """
        
        self.name = name 
        return 

    def _fit_mcd_model(self, desc_selected : np.ndarray, species : str, contamination : float = 0.05) -> None : 
        """Build the mcd model for a given species
        
        Parameters
        ----------

        desc_selected : np.ndarray 
            Selected descriptor array to perform mcd selection

        species : str
            Selected species 

        contamination : float
            percentage of outlier for mcd hyper-elliptic envelop fitting

        """
        self.models[species] = {'mcd':None,
                                'distribution':None}
        self.models[species]['mcd'] = MinCovDet(support_fraction=1.0-contamination)
        self.models[species]['mcd'].fit(desc_selected)
        
        return 
    
    def _fit_distribution(self, mcd_distances : np.ndarray, species : str) -> None :
        """Build KDE estimation for a given species
        
        Parameters
        ----------

        mcd_distances : np.ndarray
            MCD distance vector associated to a given system

        species : str
            Selected species 

        """      
        self.models[species]['distribution'] = KernelDensity(kernel='gaussian')
        self.models[species]['distribution'].fit(mcd_distances.reshape(len(mcd_distances),1))
        return 
    
    def _predict_probability(self, mcd_distances : np.ndarray, species : str) -> np.ndarray : 
        """Compute probability of a given mcd_distances vector based on MCD distances kde estimation
        
        Parameters
        ----------

        mcd_distances : np.ndarray
            MCD distance vector associated to a given system

        species : str
            Selected species 

        Returns
        -------

        np.ndarray : 
            Probability vector
        """     
        return self.models[species]['distribution'].score(mcd_distances.reshape(len(mcd_distances),1))

    def _get_mcd_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
        """Compute mcd distances based for a given species and return updated Atoms objected with new array : mcd-distance
        
        Parameters
        ----------

        list_atoms : List[Atoms]
            List of Atoms objects where mcd distance will be computed

        species : str
            Species associated to list_atoms

            
        Returns:
        --------

        List[Atoms]
            Updated List of Atoms with the new array "mcd-distance"
        """
        
        def local_setting_mcd(atoms : Atoms) -> None : 
            mcd_distance = self.models[species]['mcd'].mahalanobis(atoms.get_array('milady-descriptors'))
            atoms.set_array('mcd-distance',np.sqrt(mcd_distance), dtype=float)

        [local_setting_mcd(atoms) for atoms in list_atoms]

        return list_atoms
    
    def _write_pkl(self, path_writing : os.PathLike[str] = './mcd.pkl') -> None : 
        """Write pickle file for ```MCDModel``` object
        
        Parameters
        ----------

        path_writing : os.PathLike[str]
            Writing path for pickle file
        """
        pickle.dump(self, open(path_writing,'wb'))
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