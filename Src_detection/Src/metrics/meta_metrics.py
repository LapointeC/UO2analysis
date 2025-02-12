import numpy as np
import os
import pickle

from .gmm_metrics import GMM, GMMModel
from .maha_metrics import Mahalanobis, MahalanobisModel
from .mcd_metrics import MCD, MCDModel

from ..tools import timeit

from ase import Atoms
from typing import List, Dict, Any

class MetaModel : 
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.meta : Dict[str, GMMModel | MahalanobisModel | MCDModel] = {}
            self.meta_kind : Dict[str,str] = {}
            self.meta_data : Dict[str,Any] = {}

    def _sanity_check(self, kind : str) -> None : 
        implemented_kind = ['GMM', 'MCD', 'MAHA']
        if kind not in implemented_kind :
            raise NotImplementedError(f'... Model type {kind} is not implemented ...')

    def _update_meta_data(self, name_model : str, meta_data : Any) -> None : 
        """Update name in ```MetaModel```
        
        Parameters
        ----------

        name : str
            New name in ```MetaModel```
        """
        
        self.meta_data[name_model] = meta_data
        return 

    def _update_meta_kind(self, name_model : str, kind : str) -> None : 
        """Update kind in ```MetaModel```
        
        Parameters
        ----------

        name : str
            New kind in ```MetaModel```
        """
        
        self.meta_kind[name_model] = kind
        return 

    def _fit_model(self, desc_selected : np.ndarray, 
                   name_model : str,
                   kind : str,
                   species : str, 
                   **kwargs) -> None : 
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
        self._sanity_check(kind)
        self._update_meta_kind(name_model,kind)
        
        if kind == 'GMM' : 
            self.meta[name_model] = GMMModel()
            self.meta[name_model]._update_name(name_model)
            self.meta[name_model]._fit_gaussian_mixture_model(desc_selected,
                                                              species,
                                                              kwargs['dict_gaussian'])
        elif kind == 'MCD' :
            self.meta[name_model] = MCDModel()
            self.meta[name_model]._update_name(name_model)
            self.meta[name_model]._fit_mcd_model(desc_selected,
                                                 species,
                                                 kwargs['contamination'])
            
        elif kind == 'MAHA' : 
            self.meta[name_model] = MahalanobisModel()
            self.meta[name_model]._update_name(name_model)
            self.meta[name_model]._fit_mahalanobis_model(desc_selected,
                                                         species)

        return 
    
    def _fit_distribution(self, distances : np.ndarray, 
                          name_model : str,
                          kind : str,
                          species : str) -> None :
        """Build KDE estimation for a given species
        
        Parameters
        ----------

        mcd_distances : np.ndarray
            MCD distance vector associated to a given system

        species : str
            Selected species 

        """      

        self._sanity_check(kind)
        if kind == 'GMM' : 
            self.meta[name_model]._fit_gmm_distribution(distances,
                                                        species)
        elif kind == 'MCD' :
            self.meta[name_model]._fit_mcd_distribution(distances,
                                                        species) 
        elif kind == 'MAHA' : 
            self.meta[name_model]._fit_mahalanobis_distribution(distances,
                                                                species)

        return 
    
    def _predict_probability(self, distances : np.ndarray, 
                          name_model : str,
                          kind : str,
                          species : str) -> np.ndarray : 
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

        self._sanity_check(kind)
        if kind == 'GMM' : 
            return self.meta[name_model]._predict_gmm_probability(distances, species)
        
        elif kind == 'MCD' :
            return self.meta[name_model]._predict_mcd_probability(distances, species)
        
        elif kind == 'MAHA' : 
            return self.meta[name_model]._predict_mahalanobis_probability(distances, species)


    def _get_statistical_distances(self, list_atoms : List[Atoms], 
                                  name_model : str,
                                  kind : str,
                                  species : str) -> List[Atoms] :
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
        
        self._sanity_check(kind)
        if kind == 'GMM' :
            return self.meta[name_model]._get_gmm_distance(list_atoms,
                                                           species)
        
        elif kind == 'MCD' :
            return self.meta[name_model]._get_mcd_distance(list_atoms,
                                                           species)

        elif kind == 'MAHA' : 
            return self.meta[name_model]._get_mahalanobis_distance(list_atoms,
                                                                   species)  

    
    def _write_pkl(self, path_writing : os.PathLike[str] = './meta_models.pkl') -> None : 
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