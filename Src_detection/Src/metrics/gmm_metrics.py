import os
import pickle
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

from ase import Atoms
from typing import TypedDict, List, Dict

class GMM(TypedDict) : 
    gmm : BayesianGaussianMixture
    distribution : List[KernelDensity]

class GMMModel :
    def __init__(self, path_pkl : os.PathLike[str] = 'Nothing') : 
        if os.path.exists(path_pkl) : 
            self._load_pkl(path_pkl)
        else : 
            self.models : Dict[str, GMM] = {}
            self.name = 'GMM'

    def _update_name(self, name : str) -> None : 
        """Update name of ```GMMModel```
        
        Parameters
        ----------

        name : str
            New name for ```GMMModel```
        """
        
        self.name = name 
        return 

    def _fit_gaussian_mixture_model(self, desc_selected : np.ndarray, species : str, 
                                    dict_gaussian : dict = {'n_components':2,
                                                            'covariance_type':'full',
                                                            'init_params':'kmeans', 
                                                            'max_iter':100,
                                                            'weight_concentration_prior_type':'dirichlet_process',
                                                            'weight_concentration_prior':0.5}) -> None : 
        """Build gaussian mixture model for a given species
        
        Parameters
        ----------

        list_atoms : List[Atoms]
            List of Atoms objects to perform MCD analysis. Each Atoms object containing only 1 Atom with its 
            associated properties ...

        species : str
            Selected species 

        dict_gaussian : dict 
            Dictionnary of parameters for GMM : (i) n_components, (ii) covariance_type and (iii) init_params

        """

        defaults_dict_gaussian = {'n_components':2,
                              'covariance_type':'full',
                              'init_params':'kmeans', 
                              'max_iter':100,
                              'weight_concentration_prior_type':'dirichlet_process',
                              'weight_concentration_prior':0.5}
        
        for key, val in dict_gaussian.items() : 
            defaults_dict_gaussian[key] = val

        self.models[species] = {'gmm':None,
                                'distribution':None}
        self.models[species]['gmm'] = BayesianGaussianMixture(n_components=defaults_dict_gaussian['n_components'], 
                                                          covariance_type=defaults_dict_gaussian['covariance_type'],
                                                          init_params=defaults_dict_gaussian['init_params'],
                                                          max_iter=defaults_dict_gaussian['max_iter'],
                                                          weight_concentration_prior_type=defaults_dict_gaussian['weight_concentration_prior_type'],
                                                          weight_concentration_prior=defaults_dict_gaussian['weight_concentration_prior'],
                                                          n_init=10)
        self.models[species]['gmm'].fit(desc_selected)
        self.n_components = dict_gaussian['n_components']
        return 
    
    def _fit_gmm_distribution(self, gmm_distances : np.ndarray, species : str) -> None :
        """Build KDE estimation for a given species
        
        Parameters
        ----------

        mcd_distances : np.ndarray
            MCD distance vector associated to a given system

        species : str
            Selected species 

        """      
        self.models[species]['distribution'] = []
        for k in range(gmm_distances.shape[1]) : 
            mask = k == np.argmin(gmm_distances, axis = 1)
            self.models[species]['distribution'] += [KernelDensity(kernel='gaussian')]
            self.models[species]['distribution'][-1].fit( gmm_distances[:,k][mask].reshape(-1,1) )
        return 
    
    def _predict_gmm_probability(self, gmm_distances : np.ndarray, species : str) -> np.ndarray : 
        """Compute probability of a given mcd_distances vector based on MCD distances kde estimation
        
        Parameters
        ----------

        mcd_distances : np.ndarray
            GMM distance vector associated to a given system (n_sample, n_component)

        species : str
            Selected species 

        Returns
        -------

        np.ndarray : 
            Probability vector (n_sample, n_component)
        """
        n_samples, n_components = gmm_distances.shape
        probabilities = np.zeros((n_samples, n_components))
    
        # Loop over components, but vectorize over samples
        for k in range(n_components):
            distances_k = gmm_distances[:, k].reshape(-1, 1)
            kde = self.models[species]['distribution'][k]
            probabilities[:, k] = kde.score_samples(distances_k)
        
        return probabilities
    
    def _get_gmm_distance(self, list_atoms : List[Atoms], species : str) -> List[Atoms] :
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
        
        def mahalanobis_gmm(model : BayesianGaussianMixture, X : np.ndarray) -> np.ndarray : 
            mean_gmm = model.means_
            invcov_gmm = model.precisions_ 
            array_distance = np.empty((X.shape[0],invcov_gmm.shape[0]))

            # diagonal case for gmm
            if len(invcov_gmm.shape) < 3 :
                invcov_gmm = np.array([ np.diag(invcov_gmm[i,:]) for i in range(invcov_gmm.shape[0]) ])

            for i in range(array_distance.shape[1]):
                delta = X - mean_gmm[i, :]           # Shape: (M, D)
                temp  = delta @ invcov_gmm[i, :, :]  # Shape: (M, D)
                dist = np.sum(temp * delta, axis=1)  # Avoid   MxM matrix 
                array_distance[:, i] = np.sign(dist) * np.sqrt(np.abs(dist))
   
            return array_distance

        def local_setting_gmm(atoms : Atoms) -> None : 
            gmm_distance = mahalanobis_gmm(self.models[species]['gmm'],atoms.get_array('milady-descriptors')) 
            atoms.set_array(f'gmm-distance-{self.name}',gmm_distance, dtype=float)

        [local_setting_gmm(atoms) for atoms in list_atoms]

        return list_atoms

    def mahalanobis_gmm(self, species : str, X : np.ndarray) -> np.ndarray : 
        """Predict the distance array associated to gaussian mixture. Element i of the array 
        corresponding to d_i(X) the distance from X to the center of Gaussian i
        
        Parameters
        ----------

        species : str
            Species associated to the GMM 
        
        X : np.ndarray 
            Data to compute distances 

        Returns 
        -------

        np.ndarray 
            Distances array

        """
        mean_gmm = self.models[species]['gmm'].means_
        invcov_gmm = self.models[species]['gmm'].precisions_
        array_distance = np.empty((X.shape[0],invcov_gmm.shape[0]))
        for i in range(array_distance.shape[1]):
                array_distance[:,i] = np.diag((X-mean_gmm[i,:])@invcov_gmm[i,:,:]@(X-mean_gmm[i,:]).T)
                array_distance[:,i] = np.where(array_distance[:,i] < 0.0, 0.0, np.sqrt(array_distance[:,i]))
        return array_distance

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