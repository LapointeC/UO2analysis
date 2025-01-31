import numpy as np

import os
from ase.io import read
from ase import Atoms

from ..metrics import MetaModel
from ..mld import DBManager
from .library_mcd_multi import MetricAnalysisObject

from ..tools import timeit
from ..parser import BaseParser

import matplotlib
matplotlib.use('Agg')

from typing import Dict, Any


class ReferenceBuilder : 
    def __init__(self, xml_file : os.PathLike[str]) -> None :
        self.xml_file = xml_file
        self.meta_metric = MetricAnalysisObject()

        self.defaults_gmm = {'nb_bin_histo':100,
                             'nb_selected':10000,
                             'dic_gaussian':{'n_components':2,
                                             'covariance_type':'full',
                                             'init_params':'kmeans'}}
        
        self.defaults_mcd = {'nb_bin_histo':100,
                             'nb_selected':10000,
                             'contamination':0.05}
        
    def kwargs_manager(self, kwargs : Dict[str,Any], kind : str) -> Dict[str, Any] : 
        dic = {}
        if kind == 'MCD' : 
            for arg, value in self.defaults_mcd : 
                try : 
                    dic[arg] = kwargs[arg]
                except : 
                    dic[arg] = value

        elif kind == 'GMM' : 
            for arg, value in self.defaults_mcd : 
                try : 
                    dic[arg] = kwargs[arg]
                except : 
                    dic[arg] = value

        return dic

    def get_extension(self, file : os.PathLike[str]) -> str : 
        return os.path.basename(file).split('.')[-1]

    def build_model(self, directory : os.PathLike[str],
                           species : str,
                           kind : str,
                           name_model : str,
                           **kwargs) -> None :
        list_config_file = [f'{directory}/{f}' for f in os.listdir(directory)]
        list_atoms = []
        for file in list_config_file : 
            atoms_config = read(file, format=self.get_extension(file))
            if 'selection_mask' in kwargs : 
                atoms_config = atoms_config[kwargs['selection_mask']]
            
            list_atoms.append(atoms_config)

        if kind == 'GMM' : 
            dic_gmm = self.kwargs_manager(kwargs, 'GMM')
            self.meta_metric.fit_gmm_envelop(species, 
                                         name_model,
                                         list_atoms=list_atoms,
                                         nb_bin_histo=dic_gmm['nb_bin_histo'],
                                         nb_selected=dic_gmm['nb_selected'],
                                         dict_gaussian=dic_gmm['dict_gaussian'])
            
        elif kind == 'MCD' : 
            dic_mcd = self.kwargs_manager(kwargs, 'MCD')
            self.meta_metric.fit_mcd_envelop(species,
                                             name_model,
                                             list_atoms=list_atoms,
                                             contamination=dic_mcd['contamination'],
                                             nb_bin=dic_mcd['nb_bin'],
                                             nb_selected=dic_mcd['nb_selected'])
            
        elif kind == 'Mahalanobis' : 
            self.meta_metric.fit_mahalanobis_envelop(species,
                                                     name_model,
                                                     list_atoms=list_atoms)
            
        return 