import numpy as np
import pickle
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

from typing import Dict, Any, List, Union  


#class ReferenceBuilder : 
#    def __init__(self, xml_file : os.PathLike[str]) -> None :
#        self.xml_file = xml_file
#        self.meta_metric = MetricAnalysisObject()
#
#        self.defaults_gmm = {'nb_bin_histo':100,
#                             'nb_selected':10000,
#                             'dic_gaussian':{'n_components':2,
#                                             'covariance_type':'full',
#                                             'init_params':'kmeans'}}
#        
#        self.defaults_mcd = {'nb_bin_histo':100,
#                             'nb_selected':10000,
#                             'contamination':0.05}
#        
#    def kwargs_manager(self, kwargs : Dict[str,Any], kind : str) -> Dict[str, Any] : 
#        dic = {}
#        if kind == 'MCD' : 
#            for arg, value in self.defaults_mcd : 
#                try : 
#                    dic[arg] = kwargs[arg]
#                except : 
#                    dic[arg] = value
#
#        elif kind == 'GMM' : 
#            for arg, value in self.defaults_mcd : 
#                try : 
#                    dic[arg] = kwargs[arg]
#                except : 
#                    dic[arg] = value
#
#        return dic
#
#    def get_extension(self, file : os.PathLike[str]) -> str : 
#        return os.path.basename(file).split('.')[-1]
#
#    def build_model(self, directory : os.PathLike[str],
#                           species : str,
#                           kind : str,
#                           name_model : str,
#                           **kwargs) -> None :
#        list_config_file = [f'{directory}/{f}' for f in os.listdir(directory)]
#        list_atoms = []
#        for file in list_config_file : 
#            atoms_config = read(file, format=self.get_extension(file))
#            if 'selection_mask' in kwargs : 
#                atoms_config = atoms_config[kwargs['selection_mask']]
#            
#            list_atoms.append(atoms_config)
#
#        if kind == 'GMM' : 
#            dic_gmm = self.kwargs_manager(kwargs, 'GMM')
#            self.meta_metric.fit_gmm_envelop(species, 
#                                         name_model,
#                                         list_atoms=list_atoms,
#                                         nb_bin_histo=dic_gmm['nb_bin_histo'],
#                                         nb_selected=dic_gmm['nb_selected'],
#                                         dict_gaussian=dic_gmm['dict_gaussian'])
#            
#        elif kind == 'MCD' : 
#            dic_mcd = self.kwargs_manager(kwargs, 'MCD')
#            self.meta_metric.fit_mcd_envelop(species,
#                                             name_model,
#                                             list_atoms=list_atoms,
#                                             contamination=dic_mcd['contamination'],
#                                             nb_bin=dic_mcd['nb_bin'],
#                                             nb_selected=dic_mcd['nb_selected'])
#            
#        elif kind == 'Mahalanobis' : 
#            self.meta_metric.fit_mahalanobis_envelop(species,
#                                                     name_model,
#                                                     list_atoms=list_atoms)
#            
#        return 
    
    
    
class ReferenceBuilder:
    def __init__(self, species: str,  auto_config: dict, custom_config: List[dict]) -> None:
        """
        Initialize with configuration dictionaries from UNSEENConfigParser
        
        Parameters
        ----------
        auto_config : dict
            Dictionary containing Auto configuration
        custom_config : List[dict]
            List of dictionaries for Custom references
        """
        #TODO_cos this should be changed I keep here it in order to be compatible with the former version ... 
        self.species = species
        self.auto_config = auto_config
        self.custom_config = custom_config
        
        

    def process_auto_config(self) -> None:
        """Process the Auto configuration from provided dictionary"""
        if not self.auto_config:
            print("No Auto configuration available")
            return

        print("\n" + "="*50)
        print("Processing Auto Configuration".center(50))
        print("="*50)
        
        directory = self.auto_config['directory']
        #TODO_cos this should be changed I keep here it in order to be compatible with the former version ... 
        name_label = self.auto_config['name']
        

        for model_kind in ['MCD', 'GMM', 'MAHA']:
            if self.auto_config['models'].get(model_kind, False):
                print(f"\nBuilding {model_kind} model for {name_label}")
                self._build_model(
                    config=self.auto_config,
                    directory=directory,
                    species=self.species,
                    name_label = name_label,
                    model_kind = model_kind,
                    name_model=f"Auto_{name_label}"
                )

    def process_custom_references(self) -> None:
        """Process all custom references from configuration list"""
        if not self.custom_config:
            print("\nNo custom references to process")
            return

        print("\n" + "="*50)
        print("Processing Custom References".center(50))
        print("="*50)
        
        for ref in self.custom_config:
            print(f"\nProcessing reference: {ref['name']}")
            directory = ref['directory']
            #species = ref['name'].split('_')[-1]  # Extract species from name
            name_label = ref['name']
            for model_kind in ['MCD', 'GMM', 'MAHA']:
                if ref['models'].get(model_kind, False):
                    print(f"Building {model_kind} model for {name_label}")
                    self._build_model(
                        config=ref,
                        directory=directory,
                        species=self.species,
                        name_label = name_label, 
                        model_kind = model_kind,
                        name_model=f"Refs_{name_label}"
                    )
                    
    def _build_model(self, config: dict, directory: str, species: str,
                    name_label: str, model_kind: str, name_model: str) -> None:
        """Internal method to handle model building"""
        # Validate directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found for {name_model}")
        
        
        # # Remove any trailing slashes (unless the path is just '/' itself)
        # normalized_path = directory.rstrip(os.sep)
        # # Get the parent directory
        # parent = os.path.dirname(normalized_path)
        # # Add a trailing slash if needed and if parent is not empty
        # if parent and not parent.endswith(os.sep):
        #     parent += os.sep
        file_data_pickle = os.path.join(config['directory_path'], config['pickle_data'])
        if not os.path.exists(file_data_pickle):
            raise FileNotFoundError(f"Data pickle file with descriptors {file_data_pickle} not found for {name_model}")
    
        previous_dbmodel = pickle.load(open(file_data_pickle,'rb'))
        self.meta_metric = MetricAnalysisObject(previous_dbmodel)


        # Get model parameters from config
        model_params = config.get(name_label, {})
        
        #debig_cos print('Config', config)        
        #debig_cos print(directory)
        #debig_cos print(species)
        #debig_cos print(model_kind)
        #debig_cos print(name_model)
        #debig_cos os.system('pwd')
        #debig_cos exit(0) 
        
        # Read configuration files
        list_config_file = [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.endswith(config.get('md_format', 'cfg'))
        ]
        #debug_cos print('HHHH', list_config_file)
        
        

        # Read and process atoms
        list_atoms = []
        #TODO_cos unify with other places 
        # Define the allowed file extensions and a mapping to ASE read formats. 
        #allowed_formats = {'cfg', 'poscar', 'data', 'xyz', 'dump', 'mixed', 'unseen'}
        format_mapping = {
            'cfg':    'cfg',  # same as before
            'dump':   'lammps-dump-text',
            'poscar': 'vasp',
            'data':   'lammps-data',
            'xyz':    'xyz',
            ## For 'mixed' and 'unseen', you can choose the appropriate formats or default to something.
            #'mixed':  'lammps-dump-text',  
            #'unseen': 'lammps-dump-text'
        }  

        for file in list_config_file:
            try:
                # this read_function should be improved ....  
                if self._get_extension(file) != config.get('md_format', 'cfg') : 
                    print(f"Malheur: extension of {file } not match the input c{config.get('md_format', 'cfg')} ")
                    exit(1)
                # Get the appropriate format string for ASE.read, defaulting to 'lammps-dump-text' if not mapped.
                
                read_format = format_mapping.get(config.get('md_format', 'cfg'), 'lammps-dump-text')    
                #atoms = read(file, format=self._get_extension(file))
                atoms = read(file, format=read_format)

                #TODO_cos we should unify selection_mask and id_atoms
                nat = len(atoms)
                if 'id_atoms' in config:
                   if config['id_atoms'] == 'all' : 
                       list_selected = list(range(nat))
                   else :    
                       list_selected = config['id_atoms']
                                                                 
                   atoms = atoms[list_selected]      

                if 'selection_mask' in config:  # If using selection mask from config
                   atoms = atoms[config['selection_mask']]
                
                list_atoms.append(atoms)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
                continue

        #debug_cos print('HALLOOOOOOOOOOOOOOO')
        #debug_cos print(config['id_atoms'])
        #debug_cos print('LUUUUUUUUUL', list_config_file)
        #debug_cos print(config)

        # Build the specified model
        where_is_the_model = os.path.join(config['directory_path'],config['pickle_model'])
        try:
            if model_kind == 'GMM':
                self.meta_metric.fit_gmm_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms,
                    nb_bin_histo=model_params.get('nb_bin_histo', 100),
                    nb_selected=model_params.get('nb_selected', 10000),
                    dict_gaussian=model_params.get('dic_gaussian', {
                        'n_components': 2,
                        'covariance_type': 'full',
                        'init_params': 'kmeans',
                        'max_iter': 100,
                        'weight_concentration_prior_type':'dirichlet_process',
                        'weight_concentration_prior':0.5
                    })
                )
                self.meta_metric.store_model_pickle(f'{where_is_the_model}')
            elif model_kind == 'MCD':
                self.meta_metric.fit_mcd_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms,
                    contamination=model_params.get('contamination', 0.05),
                    nb_bin=model_params.get('nb_bin_histo', 100),
                    nb_selected=model_params.get('nb_selected', 10000)
                )
                self.meta_metric.store_model_pickle(f'{where_is_the_model}')    
                
            elif model_kind == 'MAHA':
                self.meta_metric.fit_mahalanobis_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms
                )
                self.meta_metric.store_model_pickle(f'{where_is_the_model}')
                
            print(f"Successfully built {model_kind} model for {species}")
        except Exception as e:
            print(f"Error building {model_kind} model: {str(e)}")
        #print('Here is the exit 0)')
        #exit(0) 

    def _get_extension(self, file: str) -> str:
        return os.path.basename(file).split('.')[-1]
    
    
    
    