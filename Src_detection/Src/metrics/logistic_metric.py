import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from ase import Atoms
from typing import TypedDict, List, Dict

class Logistic(TypedDict) : 
    logistic_regressor : LogisticRegression
    metadata : List[str]

class LogisticRegressor : 
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.models : Dict[str, Logistic] = {}
            self.name = 'Logistic'

    def _update_name(self, name : str) -> None : 
        """Update name of ```LogisticRegressor```
        
        Parameters
        ----------

        name : str
            New name for ```LogisticRegressor```
        """
        
        self.name = name 
        return 

    def _get_metadata(self) -> List[str] : 
        """Get metadata for ```LogisticRegressor``` object
        
        Returns
        -------

        List[str]
            List of metadata 
        """
        metadata = []
        for _, val in self.models.items() :
            if val['metadata'] not in metadata : 
                metadata += val['metadata']
        
        return metadata

    def _fit_logistic_model(self, Xdata : np.ndarray, Ytarget : np.ndarray, species : str, input_properties : List[str] = ['mcd-distance']) -> None : 
        """Build the mcd model for a given species
        
        Parameters
        ----------

        list_atoms : List[Atoms]
            List of Atoms objects to perform MCD analysis. Each Atoms object containing only 1 Atom with its 
            associated properties ...

        species : str
            Selected species 

        contamination : float
            percentage of outlier for mcd hyper-elliptic envelop fitting

        """
        self.models[species] = {'logistic_regressor':LogisticRegression(),
                                'metadata':None}
        self.models[species]['logistic_regressor'].fit(Xdata,Ytarget)
        self.models[species]['metadata'] = input_properties
        return 
    
    def _predict_logistic(self, species : str, array_desc : np.ndarray) -> np.ndarray : 
        """Predict the logistic score for a given array of descriptor
        
        Parameters
        ----------

        species : str 
            Species associated to the descritpor array 

        array_desc : np.ndarray 
            Descriptor array (M,D)

        Returns:
        --------

        np.ndarray 
            Associated logistic score probabilities array (M,N_c) where N_c is the number of logistic classes
        """
        return self.models[species]['logistic_regressor'].predict_proba(array_desc)
    

    
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