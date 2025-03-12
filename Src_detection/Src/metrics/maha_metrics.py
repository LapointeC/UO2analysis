import os
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity

from ..tools import timeit

from ase import Atoms
from typing import TypedDict, List, Dict

class Mahalanobis(TypedDict) :
    mean_vector : np.ndarray
    inv_covariance_matrix : np.ndarray 
    distribution : KernelDensity

class MahalanobisModel : 
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.models : Dict[str, Mahalanobis] = {}
            self.name = 'Mahalanobis'

    def _update_name(self, name : str) -> None : 
        """Update name of ```MCDModel```
        
        Parameters
        ----------

        name : str
            New name for ```MCDModel```
        """
        
        self.name = name 
        return 

    def _fit_mahalanobis_model(self, desc_selected : np.ndarray, species : str) -> None : 
        """Build the mcd model for a given species
        
        Parameters
        ----------

        desc_selected : np.ndarray 
            Selected descriptor array to perform mcd selection

        species : str
            Selected species 

        """
        self.models[species] = {'mean_vector':None,
                                'covariance_matrix':None,
                                'distribution':None}

        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!desc_seleted.shape : {desc_selected.shape}")
        mean_vector = desc_selected.mean(axis=0)
        desc_selected += - mean_vector
        cov_desc = np.cov(desc_selected.T)
        inv_covmat = np.linalg.pinv(cov_desc)

        self.models[species]['mean_vector'] = mean_vector[:,None]
        self.models[species]['inv_covariance_matrix'] = inv_covmat
        return 
    
    def _fit_mahalanobis_distribution(self, mcd_distances : np.ndarray, species : str) -> None :
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
    
    def _predict_mahalanobis_probability(self, mcd_distances : np.ndarray, species : str) -> np.ndarray : 
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

    def _get_mahalanobis_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
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
            Updated List of Atoms with the new array "mahalanobis-distance-name"
        """
        
        def local_setting_mahalanobis(atoms : Atoms) -> None : 
            mean_vector = self.models[species]['mean_vector']
            inv_covariance = self.models[species]['inv_covariance_matrix']
            desc = atoms.get_array('milady-descriptors')
            #debug_cos ... I change that  
            #mcd_distance = np.sqrt( np.trace( ( desc - mean_vector )@inv_covariance@( desc.T - mean_vector ) ) )
            # into that ...
             
            #old dist_matrix = ( desc - mean_vector.T )@inv_covariance@( desc.T - mean_vector )  
            #old dist = np.diagonal(dist_matrix)
            
            delta = desc - mean_vector.T 
            temp  = delta @ inv_covariance   
            dist = np.sum(temp * delta, axis=1)  # Avoid   MxM matrix 
            
            mcd_distance = np.sign(dist) * np.sqrt(np.abs(dist))
            #debug_cos I replaced that ...
            #atoms.set_array(f'mahalanobis-distance-{self.name}',np.sqrt(mcd_distance), dtype=float)
            #debug_cos into that ...
            atoms.set_array(f'maha-distance-{self.name}',mcd_distance, dtype=float)

        [local_setting_mahalanobis(atoms) for atoms in list_atoms]

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