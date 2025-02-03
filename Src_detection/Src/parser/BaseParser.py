from __future__ import annotations
import numpy as np
import os, sys, shutil
import xml.etree.ElementTree as ET
import numpy as np
import pathlib 
from typing import Union,Any,List
ScriptArg = Union[int,float,str]


class BaseParser:
    def __init__(self,
                xml_path:None|os.PathLike[str]=None) -> None:
        """Base reader of MAB XML configuration file
        
        Parameters
        ----------
        xml_path : os.PathLike[str]
            path to XML file, default None
        
        """
        self.parameters = {}
        #self.set_default_parameters()
        self.xml_path = xml_path
    
        if not xml_path is None:    
            assert os.path.exists(xml_path)
            xml_tree = ET.parse(xml_path)
            for branch in xml_tree.getroot():
                if branch.tag=="Directory":
                    self.read_parameters(branch)
                else:
                    raise IOError("Error in XML file!")
        else:
            raise TimeoutError('No xml file for Unseen Token...')

    
    def set_default_parameters(self, name : str) -> None:
        """Set default values for <Parameters> data
        read_parameters() will *only* overwrite these values
        """

        tmp_dictionnary = {}

        tmp_dictionnary['Mode'] = 'Auto'

        # data
        tmp_dictionnary["PathData"] = ['Somewhere'] 
        tmp_dictionnary["SelectionMask"] = []
        
        # meta data
        tmp_dictionnary['Name'] = 'default_data'
        tmp_dictionnary['StatisticalDistance'] = 'MCD'

        # mcd settings
        tmp_dictionnary['contamination'] = 0.05
        tmp_dictionnary['nb_selected'] = 10000
        tmp_dictionnary['nb_bin'] = 100

        # gmm 
        tmp_dictionnary['dic_gaussian'] = {'n_components':2,
                                             'covariance_type':'full',
                                             'init_params':'kmeans'}

        self.parameters[name] = tmp_dictionnary
        return

    def read_parameters(self,xml_parameters:ET.Element) -> None:
        """Read in simulation parameters defined in the XML file 

        Parameters
        ----------
        xml_parameters : xml.etree.ElementTreeElement
            The <Parameters> branch of the configuration file,
            represented as an ElementTree Element
        
        """
        def boolean_converter(str_xml : str) -> bool :
            """Convert string chain into boolean 

            Parameters 
            ----------
            str_xml : str
                string to convert 
            Returns 
            -------

            bool
            """
            if str_xml in ['true' ,'True', '.TRUE.','.True.', 'Yes'] : 
                return True
            if str_xml in ['false', 'False', '.FALSE.', '.False.' ,'No'] :
                return False

        for var in xml_parameters:
            tag = var.tag.strip()
            if not tag in self.parameters:
                print(f"Undefined parameter {tag}!!, skipping")
                continue
            else:
                o = self.parameters[tag]
                n = var.text
                if isinstance(o,bool):
                    self.parameters[tag] = boolean_converter(n)
                elif isinstance(o,int):
                    self.parameters[tag] = int(n)
                elif isinstance(o,float):
                    self.parameters[tag] = float(n)
                elif isinstance(o,str):
                    self.parameters[tag] = n
                elif isinstance(o,list):
                    self.parameters[tag] = [float(dat) for dat in n.split()]
                elif isinstance(o,dict):
                    for child in var : 
                        tag_child = child.tag
                        co = self.parameters[tag][tag_child]
                        cn = child.text
                        if isinstance(co,int) :
                            self.parameters[tag][tag_child] = int(cn)
                        elif isinstance(co,float) :
                            self.parameters[tag][tag_child] = float(cn)
                        elif isinstance(co,str) :
                            self.parameters[tag][tag_child] = cn            
                        elif isinstance(co,bool) : 
                            self.parameters[tag][tag_child] = boolean_converter(cn)


    def read_scripts(self,xml_scripts : ET.Element) -> None:
        """Read in scripts defined in the XML file 

        Parameters
        ----------
        xml_parameters : xml.etree.ElementTreeElement
            The <Scripts> branch of the configuration file,
            represented as an ElementTree Element
        
        """
        for script in xml_scripts:
            if not script.text is None:
                tag = script.tag.strip()
                if not tag in self.scripts:
                    print(f"adding script {tag}")
                self.scripts[tag] = script.text.strip()
                
        
    def replace(self,field:str,key:str,value: ScriptArg) -> str:
        """Wrapper around string replace()
           
        Parameters
        ----------
        field : str
            string to be searched
        key : str
            will search for %key%
        value : ScriptArg
            replacement value
        Returns
        -------
        str
            the string with replaced values
        """
        return field.replace("%"+key+"%",str(value))
    
    def parse_script(self,script_key:str,
                     arguments:None|dict=None) -> str:
        """Parse an input script
            If script_key is not a key of self.scripts, it 
            it is treated as a script itself

        Parameters
        ----------
        script_key : str
            key for <Script> in XML file
        args : None | dict, optional
            Dictionary of key,value pairs for replace(), by default None

        Returns
        -------
        str
            The script with any keywords replaced
        """
        if not script_key in self.scripts:
            script = script_key
        else:
            script = self.scripts[script_key]
        if arguments is None:
            _args = {}
        else:
            _args = arguments.copy()
        _args["Configuration"] = self.Configuration
        _args["DELTAT"] = self.parameters['DeltaTau']
        if not self.PotentialLocation is None:
            _args["Potential"] = self.PotentialLocation
        if not self.Species is None:
            _args["Species"] = " ".join(self.Species)
        for key,value in _args.items():
            script = self.replace(script,key,value)
        return script
    
    
    
class UNSEENConfigParser: 
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.auto_config: Dict[str, Any] = {}
        self.custom_config: List[Dict[str, Any]] = []
        
        if os.path.exists(xml_path):
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

    def parse(self) -> None:
        """Main parsing method to process Auto and Custom tags"""
        for child in self.root:
            if child.tag == 'Auto':
                self.parse_auto(child)
            elif child.tag == 'Custom':
                self.parse_custom(child)

    def parse_auto(self, element: ET.Element) -> None:
        """Parse <Auto> configuration"""
        self.auto_config['name'] = element.findtext('name', default='main_analysis').strip()
        self.auto_config['directory'] = element.findtext('directory', default='./').strip()
        
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

        
        self.auto_config['md_format'] = element.findtext('md_format', default='cfg').strip()
        print(self.auto_config['md_format'])
        exit(0) 
        self.auto_config['id_atoms'] = self.parse_id_atoms(element.findtext('id_atoms', default='all'))
        
        # Parse model activation
        model_text = element.findtext('model', default='MCD').strip()
        self.auto_config['models'] = {
            'MCD': 'MCD' in model_text.split(),
            'GMM': 'GMM' in model_text.split(),
            'MAHA': 'MAHA' in model_text.split()
        }
        
        # Parse model options
        self.auto_config['MCD'] = self.parse_mcd(element.find('MCD'))
        self.auto_config['GMM'] = self.parse_gmm(element.find('GMM'))
        self.auto_config['MAHA'] = self.parse_maha(element.find('MAHA'))

    def parse_custom(self, element: ET.Element) -> None:
        """Parse <Custom> configuration containing multiple <Reference> entries"""
        for ref_elem in element.findall('Reference'):
            ref: Dict[str, Any] = {}
            ref['name'] = ref_elem.findtext('name', default='ref_01').strip()
            ref['directory'] = ref_elem.findtext('directory', default='./References').strip()
            # Convert to a pathlib object
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
            ref['id_atoms'] = self.parse_id_atoms(ref_elem.findtext('id_atoms', default='all'))
            
            model_text = ref_elem.findtext('model', default='MCD').strip()
            ref['models'] = {
                'MCD': 'MCD' in model_text.split(),
                'GMM': 'GMM' in model_text.split(),
                'MAHA': 'MAHA' in model_text.split()
            }
            
            ref['MCD'] = self.parse_mcd(ref_elem.find('MCD'))
            ref['GMM'] = self.parse_gmm(ref_elem.find('GMM'))
            ref['MAHA'] = self.parse_maha(ref_elem.find('MAHA'))
            
            self.custom_config.append(ref)

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
                ids.extend(range(start, end + 1))
            else:
                ids.append(int(r))
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

