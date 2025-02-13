import numpy as np
import pickle
import os
from ase.io import read
from ase import Atoms

from ..metrics import MetaModel, PCAModel
from ..mld import DBManager
from .library_mcd_multi import MetricAnalysisObject

from ..tools import timeit
from ..parser import UNSEENConfigParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, Any, List, Union  

    
class ReferenceBuilder:
    def __init__(self,
                 auto_config : dict, 
                 custom_config : List[dict]) -> None:
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
        self.auto_config = auto_config
        self.custom_config = custom_config
        self.meta_metric = MetricAnalysisObject(dbmodel=None)
        

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
                print(self.auto_config)
                print(f"\nBuilding {model_kind} model for {name_label}")
                for s in self.auto_config.get('species',['Fe']) : 
                    self._build_model(
                        config=self.auto_config,
                        directory=directory,
                        species=s,
                        name_label = name_label,
                        model_kind = model_kind,
                        name_model=f"Auto_{name_label}"
                    )

        print(f"... Auto pickle MetaModel is stored : {self.auto_config['path_metamodel_pkl']} ...")
        self.meta_metric.store_model_pickle(f"{self.auto_config['path_metamodel_pkl']}/auto_metamodel.pkl")

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
                    print(ref)
                    for s in ref.get('species',['Fe']) :
                        print(s) 
                        self._build_model(
                            config=ref,
                            directory=directory,
                            species=s,
                            name_label = name_label, 
                            model_kind = model_kind,
                            name_model=f"Refs_{name_label}"
                        )

        # Build the specified model
        #where_is_the_model =               
        
        path_meta_pkl = self.custom_config[0]['path_metamodel_pkl']
        print(f"... Custom pickle MetaModel is stored : {self.auto_config['path_metamodel_pkl']} ...")
        self.meta_metric.store_model_pickle(f"{path_meta_pkl}/custom_metamodel.pkl")
        return 

    def _build_model(self, config: dict, 
                    directory: str, 
                    species: str,
                    name_label: str, 
                    model_kind: str, 
                    name_model: str) -> None:
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
    
        previous_dbmodel : DBManager = pickle.load(open(file_data_pickle,'rb'))
        configuration = previous_dbmodel.model_init_dic
        list_atoms = []
        for key_c, item_c in configuration.items() : 
            try : 
                atoms = item_c['atoms']
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
                print(f"Error reading {key_c}")
                continue          

        # Get model parameters from config
        #model_params = config.get(name_model, {})
        model_params = config.get(model_kind, {})
        
        try:
            if model_kind == 'GMM':
                self.meta_metric.fit_gmm_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms,
                    contour=config.get('contour', False),
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
        
            elif model_kind == 'MCD':
                self.meta_metric.fit_mcd_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms,
                    contamination=model_params.get('contamination', 0.05),
                    nb_bin=model_params.get('nb_bin_histo', 100),
                    nb_selected=model_params.get('nb_selected', 10000),
                    contour=config.get('contour', False)
                )   
                
            elif model_kind == 'MAHA':
                self.meta_metric.fit_mahalanobis_envelop(
                    species=species,
                    name_model=name_model,
                    list_atoms=list_atoms,
                    contour=config.get('contour', False),
                )
                #self.meta_metric.store_model_pickle(f'{where_is_the_model}')
                
            print(f"Successfully built {model_kind} model for {species}")
        
        except Exception as e:
            print(f"Error building {model_kind} model: {str(e)}")
        #print('Here is the exit 0)')
        #exit(0)

        return 

    def _get_extension(self, file: str) -> str:
        return os.path.basename(file).split('.')[-1]
    

class InferenceBuilder:
    def __init__(self, 
                 inference_config: dict, 
                 auto_config: dict, 
                 custom_config: List[dict]) -> None:    
        """
        Initialize with inference configuration from UNSEENConfigParser
        
        Parameters
        ----------
        species : str
            Target species for inference
        inference_config : dict
            Dictionary containing Inference configuration
        """
        self.infer_config = inference_config
        self.auto_config = auto_config
        self.custom_config = custom_config
        self.metamodel = MetaModel()
        self.dict_data : DBManager = None 

        self._load_models()
   
    def _load_models(self) -> None:
        """Load pre-trained models from pickle files"""
        model_dir = self.infer_config.get('path_metamodel_pkl', '')
        try : 
            self.metamodel._load_pkl(f"{model_dir}/custom_metamodel.pkl")
        except : 
            self.metamodel._load_pkl(f"{model_dir}/auto_metamodel.pkl")
        return

    def run_inference(self) -> None:
        """Main method to execute full inference pipeline"""
        print("\n" + "="*50)
        print("Running Inference Analysis".center(50))
        print("="*50)
        
        file_data_pickle = self.infer_config['directory']
        storage_pickle_data = self.infer_config['storage_pickle']

        where_is_pkl = os.path.join(os.path.dirname(file_data_pickle),f"{self.infer_config['name']}_inf_data.pickle")
        
        # Read previous pickle file
        previous_dbmodel : DBManager = pickle.load(open(where_is_pkl,'rb'))
        configuration = previous_dbmodel.model_init_dic
        self.dict_data = previous_dbmodel

        for key_c, item_c in configuration.items() : 
            try : 
                print(f"... Analysing {key_c} configuration ...")
                atoms = item_c['atoms']
                nat = len(atoms)
                if 'id_atoms' in self.infer_config:
                   if self.infer_config['id_atoms'] == 'all' : 
                       list_selected = list(range(nat))
                   else :    
                       list_selected = self.infer_config['id_atoms']

                   atoms = atoms[list_selected]      
                if 'selection_mask' in self.infer_config:  # If using selection mask from config
                   atoms = atoms[self.infer_config['selection_mask']]
                
                tmp_atoms = [atoms]
                for key, kind_k in self.metamodel.meta_kind.items() : 
                    tmp_atoms = self._process_model(kind_k,
                                                    key,
                                                    tmp_atoms)
                # Save results
                output_dir = os.path.join(self.infer_config['directory'], 'results')
                os.makedirs(output_dir, exist_ok=True)

                # Save modified atoms with distances
                output_path = os.path.join(output_dir, f"filled_{key_c}.xyz")
                atoms.write(output_path)
                print(f"... .xyz file for {key_c} configuration is written ...")
                print(f"... {output_path} ...")
                print()

                self.dict_data.model_init_dic[key_c]['atoms'] = tmp_atoms[0]


            except Exception as e:
                print(f"Error reading {key_c}")
                continue    
        
        self.store_pkl_data(storage_pickle_data)
        return
        
    def _process_model(self, model_type: str,
                       name : str,
                       atoms_list: List[Atoms]) -> List[Atoms]:
        """Process inference for a specific model type"""
        print(f"    Processing {name}/{model_type} inference")
        
        # Get statistical distances
        for s in self.infer_config.get('species',['Fe']) :
            atoms_list = self.metamodel._get_statistical_distances(
                atoms_list,
                name,
                model_type,
                s
            )

            # Generate visualizations
            self._generate_visualizations(model_type, s, name, atoms_list)

        return atoms_list
    
    def _generate_visualizations(self, model_type: str, 
                                 species : str, 
                                 name : str,
                                 atoms_list: List[Atoms]) -> None:
        """Generate analysis plots for the model results"""
        distances = np.concatenate(
            [atoms.get_array(f'{model_type.lower()}-distance-{name}') 
             for atoms in atoms_list]
        )
        
        # Create histogram plot
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, density=True, alpha=0.7)
        plt.xlabel(f'{model_type} Distance')
        plt.ylabel('Probability Density')
        plt.title(f'{model_type} Distance Distribution for {species}')
        plt.savefig(os.path.join(self.infer_config['directory'], f'{species}_{model_type}_distribution.png'))
        plt.close()

        # Create PCA plot if applicable
        if model_type in ['MCD', 'MAHA']:
            try:
                for s in self.infer_config.get('species',['Fe']) :
                    desc_transform = PCAModel()._get_pca_model(
                        atoms_list, 
                        s, 
                        n_component=2
                    )

                    plt.figure(figsize=(10, 6))
                    plt.scatter(desc_transform[:,0], desc_transform[:,1], c=distances, cmap='viridis')
                    plt.xlabel('First Principal Component')
                    plt.ylabel('Second Principal Component')
                    plt.colorbar(label=f'{model_type} Distance')
                    plt.title(f'PCA Visualization for {s}')
                    plt.savefig(os.path.join(self.infer_config['directory'], f'{s}_{model_type}_pca.png'))
                    plt.close()
            except Exception as e:
                print(f"Could not generate PCA plot: {str(e)}")
    
    def _get_extension(self, file: str) -> str:
        return os.path.basename(file).split('.')[-1]

    
    def store_pkl_data(self, path_pkl : os.PathLike[str]) -> None : 
        pickle.dump(self.dict_data, open(path_pkl,'wb'))
        return()
    
    