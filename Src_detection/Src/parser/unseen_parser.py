from __future__ import annotations
import numpy as np
import os, sys, shutil
import xml.etree.ElementTree as ET
import numpy as np
import pathlib 
from typing import Union,Any,List, Dict, Tuple
ScriptArg = Union[int,float,str]


class UNSEENConfigParser: 
    def __init__(self, xml_path: os.PathLike[str]):
        self.xml_path = xml_path
        self.path_metamodel_pkl : os.PathLike[str] = None
        self.auto_config: Dict[str, Any] = {}
        self.custom_config: List[Dict[str, Any]] = []
        self.inference_config: Dict[str, Any] = {}
        
        if os.path.exists(xml_path):
            print(xml_path)
            self.tree = ET.parse(xml_path)
            self.root = self.tree.getroot()
            self.parse()
        else:
            print(f"XML file {xml_path} not found. Using default parameters.")
            self.set_defaults()
            
    def set_defaults(self) -> None:
        """Initialize configuration with default values"""
        # Auto configuration defaults
        self.auto_config = {
            'name': 'main_analysis',
            'directory': './',
            'md_format': 'cfg',
            'id_atoms': 'all',
            'models': {'MCD': True, 'GMM': False, 'MAHA': False},
            'species':'Fe',
            'MCD': {'nb_bin_histo': 100, 'nb_selected': 10000, 'contamination': 0.05},
            'GMM': {
                'nb_bin_histo': 100,
                'nb_selected': 10000,
                'dic_gaussian': {
                    'n_components': 2,
                    'covariance_type': 'full',
                    'init_params': 'kmeans'
                }
            },
            'MAHA': {'nb_bin_histo': 100, 'nb_selected': 10000}
        }
        self.custom_config = []  
        self.inference_config = {}    
        self.path_metamodel_pkl = os.getcwd()

    def parse(self) -> None:
        """Main parsing method to process Auto and Custom tags"""
        for child in self.root:
        
            if child.tag == 'PickleMetaModel': 
                self.set_metamodel_path(child)
            elif child.tag == 'Auto':
                self.parse_auto(child)
            elif child.tag == 'Custom':
                self.parse_custom(child)
            elif child.tag == 'Inference':
                self.parse_inference(child)

    def set_metamodel_path(self, element : ET.Element) -> None :
        self.path_metamodel_pkl = element.findtext('path', default=os.getcwd()).strip()
        return 

    def parse_auto(self, element: ET.Element) -> None:
        """Parse <Auto> configuration"""
        self.auto_config['name'] = element.findtext('name', default='main_analysis').strip()
        self.auto_config['directory'] = element.findtext('directory', default='./').strip()
        
        
        # Remove any trailing slashes (unless the path is just '/' itself)
        normalized_path = self.auto_config['directory'].rstrip(os.sep)
        # Get the parent directory
        parent = os.path.dirname(normalized_path)
        # Add a trailing slash if needed and if parent is not empty
        if parent and not parent.endswith(os.sep):
           parent += os.sep
        self.auto_config['directory_path'] = parent 
        self.auto_config['directory_name'] = os.path.basename(normalized_path)
        #debug_cos print('DIRECTORIESSSSSSSS  ', self.auto_config['directory'], self.auto_config['directory_path'], self.auto_config['directory_name'])
        
        auto_pickle_file = f"{self.auto_config.get('name', 'ref')}_ref_data.pickle"
        #auto_pickle_model = f"{self.auto_config.get('name', 'ref')}_ref_model.pickle"
        self.auto_config['pickle_data'] = auto_pickle_file   
        #self.auto_config['pickle_model'] = auto_pickle_model

        
        
        # Convert to a pathlib object
        directory_path = pathlib.Path(self.auto_config['directory'])
        
        # Check if the directory exists and is not a symbolic link
        if not directory_path.exists():
             print(f"Error: The specified Auto directory '{directory_path}' does not exist. Please check the configuration.")
             sys.exit(1) 
        #elif directory_path.is_symlink() or directory_path.is_dir():
        #    raise ValueError(f"Error: The directory '{directory_path}' is a symbolic link, which is not allowed.")
        if not directory_path.is_dir():
             print(f"Error: The path '{directory_path}' exists but is not a directory.")
             sys.exit(1) 

        
        # species option
        self.auto_config['species'] = element.findtext('species', default='Fe').strip().split()           
        
        self.auto_config['md_format'] = element.findtext('md_format', default='cfg').strip()            
        allowed_formats = {'cfg', 'poscar', 'data', 'xyz', 'dump', 'mixed', 'unseen'}
        md_format = element.findtext('md_format', default='cfg').strip()
        if md_format not in allowed_formats:
            print(f"Invalid md_format for Auto: '{md_format}'. ")
            print(f"Allowed formats are: {', '.join(sorted(allowed_formats))}")
            sys.exit(1) 
            

        self.auto_config['id_atoms'] = self.parse_id_atoms(element.findtext('id_atoms', default='all'))
        
        
        model_text = element.findtext('model', default='MCD').strip()
        models = model_text.split()
        allowed_models = {'MCD', 'GMM', 'MAHA'}
        invalid_models = [model for model in models if model not in allowed_models]
        
        if invalid_models:
            print(f"Invalid model(s): {invalid_models}. Allowed options in Auto are MCD, GMM, MAHA")
            sys.exit(1)
        
        self.auto_config['models'] = {
            'MCD': 'MCD' in models,
            'GMM': 'GMM' in models,
            'MAHA': 'MAHA' in models
        }
        
        # Parse model options
        self.auto_config['MCD'] = self.parse_mcd(element.find('MCD'))
        self.auto_config['GMM'] = self.parse_gmm(element.find('GMM'))
        self.auto_config['MAHA'] = self.parse_maha(element.find('MAHA'))

        # Add metamodel path 
        self.auto_config['path_metamodel_pkl'] = self.path_metamodel_pkl

    def parse_custom(self, element: ET.Element) -> None:
        """Parse <Custom> configuration containing multiple <Reference> entries"""

        for ref_elem in element.findall('Reference'):
            ref: Dict[str, Any] = {}
            ref['name'] = ref_elem.findtext('name', default='ref_01').strip()
            ref['directory'] = ref_elem.findtext('directory', default='./References').strip()
            # Convert to a pathlib object
            
            # Remove any trailing slashes (unless the path is just '/' itself)
            normalized_path = ref['directory'].rstrip(os.sep)
            # Get the parent directory
            parent = os.path.dirname(normalized_path)
            # Add a trailing slash if needed and if parent is not empty
            if parent and not parent.endswith(os.sep):
               parent += os.sep
            ref['directory_path'] = parent 
            ref['directory_name'] = os.path.basename(normalized_path)
            #debug_cos print('DIRECTORIESSSSSSSS  ', ref['directory'], ref['directory_path'], ref['directory_name'])
            
            ref_pickle_file = f"{ref.get('name', 'ref')}_ref_data.pickle"
            #ref_pickle_model = f"{ref.get('name', 'ref')}_ref_model.pickle"
            ref['pickle_data'] = ref_pickle_file   
            #ref['pickle_model'] = ref_pickle_model
            
            
            directory_path = pathlib.Path(ref['directory'])
            
            # Check if the directory exists and is not a symbolic link
            if not directory_path.exists():
                 print(f"Error: The specified Custom directory '{directory_path}' does not exist. Please check the configuration.")
                 sys.exit(1) 
            #elif directory_path.is_symlink() or directory_path.is_dir():
            #    raise ValueError(f"Error: The directory '{directory_path}' is a symbolic link, which is not allowed.")
            if not directory_path.is_dir():
             print(f"Error: The path '{directory_path}' exists but is not a directory.")
             sys.exit(1) 
            ref['md_format'] = ref_elem.findtext('md_format', default='cfg').strip()
            
            # species
            ref['species'] = element.findtext('species', default='Fe').strip().split()            
        

            allowed_formats = {'cfg', 'poscar', 'data', 'xyz', 'dump', 'mixed', 'unseen'}
            md_format = ref_elem.findtext('md_format', default='cfg').strip()
            if md_format not in allowed_formats:
                print(f"Invalid md_format for Reference: '{md_format}'. ")
                print(f"Allowed formats are: {', '.join(sorted(allowed_formats))}")
                sys.exit(1) 
            
            try : 
                ref['id_atoms'] = self.parse_id_atoms(ref_elem.findtext('id_atoms', default='all'))
            except : 
                ref['selection_mask'] = self.parse_slice_atoms(ref_elem.findtext('selection_mask',default='[:]'))

            model_text = ref_elem.findtext('model', default='MCD').strip()
            models = model_text.split()
            allowed_models = {'MCD', 'GMM', 'MAHA'}
            invalid_models = [model for model in models if model not in allowed_models]
        
            if invalid_models:
                print(f"Invalid model(s): {invalid_models}. Allowed options in Reference are MCD, GMM, MAHA")
                sys.exit(1)
        
            
            ref['models'] = {
                'MCD': 'MCD' in model_text.split(),
                'GMM': 'GMM' in model_text.split(),
                'MAHA': 'MAHA' in model_text.split()
            }
            
            ref['MCD'] = self.parse_mcd(ref_elem.find('MCD'))
            ref['GMM'] = self.parse_gmm(ref_elem.find('GMM'))
            ref['MAHA'] = self.parse_maha(ref_elem.find('MAHA'))
            
            # Add metamodel path 
            ref['path_metamodel_pkl'] = self.path_metamodel_pkl

            self.custom_config.append(ref)

    def parse_inference(self, element: ET.Element) -> None:
        """Parse <Inference> configuration"""
        try : 
            self.inference_config = {
                'name': element.findtext('name', default='inference').strip(),
                'directory': element.findtext('directory', default='./Inference').strip(),
                'storage_pickle': element.findtext('storage_pickle', default='./inference.pkl').strip(),
                'species': element.findtext('species', default='Fe').strip().split(),
                'md_format': element.findtext('md_format', default='cfg').strip(),
                'id_atoms': self.parse_id_atoms(element.findtext('id_atoms', default='all'))
            }
        except : 
            self.inference_config = {
                'name': element.findtext('name', default='inference').strip(),
                'directory': element.findtext('directory', default='./Inference').strip(),
                'storage_pickle': element.findtext('storage_pickle', default='./inference.pkl').strip(),
                'species': element.findtext('species', default='Fe').strip().split(),
                'md_format': element.findtext('md_format', default='cfg').strip(),
                'selection_mask': self.parse_slice_atoms(element.findtext('selection_mask',default='[:]'))
            }  

        # Validate MD format
        allowed_formats = {'cfg', 'poscar', 'data', 'xyz', 'dump', 'mixed', 'unseen'}
        if self.inference_config['md_format'] not in allowed_formats:
            print(f"Invalid md_format for Inference: '{self.inference_config['md_format']}'")
            print(f"Allowed formats: {', '.join(allowed_formats)}")
            sys.exit(1)

        # Process directory paths
        normalized_path = self.inference_config['directory'].rstrip(os.sep)
        self.inference_config['directory_path'] = os.path.dirname(normalized_path)
        self.inference_config['directory_name'] = os.path.basename(normalized_path)

        # Validate directory exists
        directory_path = pathlib.Path(self.inference_config['directory'])
        if not directory_path.exists():
            print(f"Inference directory {directory_path} does not exist")
            sys.exit(1)
        if not directory_path.is_dir():
            print(f"Inference path {directory_path} is not a directory")
            sys.exit(1)
            
        ref_pickle_file = f"{self.inference_config.get('name', 'infer')}_inf_data.pickle"
        #ref_pickle_model = f"{self.inference_config.get('name', 'infer')}_inf_model.pickle"
        self.inference_config['pickle_data'] = ref_pickle_file   
        #self.inference_config['pickle_model'] = ref_pickle_model
        self.inference_config['path_metamodel_pkl'] = self.path_metamodel_pkl

    def parse_slice_atoms(self, text_slice : str) -> Tuple[slice] : 
        """Parse numpy array slice to select id_atoms"""
        return tuple((slice(*(int(i) if i else None for i in part.strip().split(':'))) 
                      if ':' in part else int(part.strip())) 
                      for part in text_slice.strip('[]').split(','))

    def parse_id_atoms(self, id_text: str) -> List[int] | str:
        """Parse id_atoms string into list of integers or 'all'"""
        id_text = id_text.strip()
        if id_text.lower() == 'all':
            return 'all'
        
        ids = []
        ranges = id_text.split(',')
        for r in ranges:
            if '-' in r:
                start, end = map(int, r.split('-'))
                #ids.extend(range(start, end + 1))
                # python baby ! 
                ids.extend(range(start-1, end))
            else:
                #ids.append(int(r))
                # python baby ! 
                ids.append(int(r)-1)
        return ids

    def parse_mcd(self, mcd_elem: ET.Element | None) -> Dict[str, Any]:
        """Parse MCD options with defaults"""
        defaults = {'nb_bin_histo': 100, 'nb_selected': 10000, 'contamination': 0.05}
        if mcd_elem is None:
            return defaults
        
        opts = mcd_elem.find('mcd_options')
        if opts is None:
            return defaults
        
        mcd = defaults.copy()
        for child in opts:
            tag = child.tag
            if tag == 'nb_bin_histo':
                mcd[tag] = int(child.text)
            elif tag == 'nb_selected':
                mcd[tag] = int(child.text)
            elif tag == 'contamination':
                mcd[tag] = float(child.text)
        return mcd

    def parse_gmm(self, gmm_elem: ET.Element | None) -> Dict[str, Any]:
        """Parse GMM options with defaults"""
        defaults = {
            'nb_bin_histo': 100,
            'nb_selected': 10000,
            'dic_gaussian': {
                'n_components': 2,
                'covariance_type': 'full',
                'init_params': 'kmeans'
            }
        }
        if gmm_elem is None:
            return defaults
        
        opts = gmm_elem.find('gmm_options')
        if opts is None:
            return defaults
        
        gmm = defaults.copy()
        for child in opts:
            print('child gmm')
            tag = child.tag
            if tag in ['nb_bin_histo', 'nb_selected']:
                gmm[tag] = int(child.text)
            elif tag == 'dic_gaussian':
                for g_child in child:
                    g_tag = g_child.tag
                    if g_tag == 'n_components':
                        gmm['dic_gaussian'][g_tag] = int(g_child.text)
                    else:
                        gmm['dic_gaussian'][g_tag] = g_child.text.strip()
        return gmm

    def parse_maha(self, maha_elem: ET.Element | None) -> Dict[str, Any]:
        """Parse MAHA options with defaults"""
        defaults = {'nb_bin_histo': 100, 'nb_selected': 10000}
        if maha_elem is None:
            return defaults
        
        opts = maha_elem.find('maha_options')
        if opts is None:
            return defaults
        
        maha = defaults.copy()
        for child in opts:
            tag = child.tag
            if tag in ['nb_bin_histo', 'nb_selected']:
                maha[tag] = int(child.text)
        return maha

