from __future__ import annotations
import numpy as np
import os, shutil
import xml.etree.ElementTree as ET
import numpy as np
from typing import Union,Any,List
ScriptArg = Union[int,float,str]

class BaseParser:
    """
    """
    def __init__(self,
                xml_path:None|os.PathLike[str]=None,
                rank=0) -> None:
        """Base reader of MAB XML configuration file
        
        Parameters
        ----------
        xml_path : os.PathLike[str]
            path to XML file, default None
        
        """
        self.set_default_parameters()
        self.set_default_scripts()
        self.xml_path = xml_path
        self.rank=rank
        self.PotentialLocation = None
        self.Species = None
        self.Configuration = None

        self.seeded = False
        if not xml_path is None:    
            assert os.path.exists(xml_path)
            xml_tree = ET.parse(xml_path)
            for branch in xml_tree.getroot():
                if branch.tag=="Parameters":
                    self.read_parameters(branch)
                elif branch.tag=="Scripts":
                    self.read_scripts(branch)
                elif branch.tag=="Pathways" :
                    self.read_pathways(branch)
                else:
                    raise IOError("Error in XML file!")
        else:
            raise TimeoutError('No xml file for MC...')

    def check(self)->None:
        """Internal check of parameters
        """
        if self.check_configuration() : 
            return True
        else : 
            return False
    
    def ready(self)->bool:
        """Check if all parameters are set

        Returns
        -------
        bool
        """
        if not self.has_potential:
            raise TypeError("\n\nNo Potential set!\n")
        if not self.check_configuration() :
            raise TypeError("\n\nIntial configuration file does not exist!\n")

        return True
                    
    def check_configuration(self) -> bool : 
        """
            Ensure configuration file exists
        """        
        configuration = self.Configuration
        return os.path.exists(configuration)

    def to_dict(self) -> dict:
        """Export axes and parameters as a nested dictionary

        Returns
        -------
        dict
            Dictionary-of-dictionaries' and 'parameters'
        """
        out_dict = {}
        out_dict['parameters'] = self.parameters.copy()
        return out_dict
    
    def set_default_parameters(self) -> None:
        """Set default values for <Parameters> data
        read_parameters() will *only* overwrite these values
        """
        self.parameters = {}

        self.parameters["Restart"] = False
        self.parameters["CoresPerWorker"] = 1
        self.parameters["Verbose"] = 0
        self.parameters["Temperature"] = 10.0 
        self.parameters["MuGrid"] = {"Min":0.0,"Max":1.0,"NumberMu":100}        
        self.parameters["ThermalisationSteps"] = 1000
        self.parameters["GlobalSeed"] = 1285237
        self.parameters["LogLammps"] = False

        self.parameters["WritingDirectory"] = './dump'

        self.parameters['Mode'] = 'SGC-MC' 
        #SGC-MC
        self.parameters["DeltaTau"] = 0.1
        self.parameters['DampingT'] = 0.5
        self.parameters['DampingP'] = 6.25
        self.parameters['FrequencyMC'] = 20
        self.parameters['NumberNPTSteps'] = 1000
        self.parameters['FractionSwap'] = 0.25

        #VC-SGC-MC 
        self.parameters["MuArray"] = [0.0,1.0]
        self.parameters["ConcentrationArray"] = [0.5,0.5]
        self.parameters["Kappa"] = 0.1
        self.parameters["WritingStep"] = 1000

    def create_writing_directory(self, path : os.PathLike[str]) -> None : 
        """Build the directory to write dump files 
        
        Parameters
        ----------

        path : os.PathLike[str]
            Path to the writing directory
        
        """
        if os.path.exists(path) and not self.parameters["Restart"] : 
            shutil.rmtree(path)
            os.mkdir(path)
        
        elif os.path.exists(path) and self.parameters["Restart"] : 
            pass

        else : 
            os.mkdir(path)
        return 

    def read_pathways(self, xml_parameters:ET.Element) -> None : 
        """Read in pathway configuration paths defined in the XML file 

        Parameters
        ----------
        xml_path_data : xml.etree.ElementTreeElement
            The <PathwayConfigurations> branch of the configuration file,
            represented as an ElementTree Element
        """

        for var in xml_parameters :
            if var.tag == "Configuration" : 
                self.Configuration = var.text.strip()
            elif var.tag == "Potential" :
                potential = var.text.strip()
            elif var.tag == "WritingDirectory" : 
                self.parameters["WritingDirectory"] = var.text
        if self.rank == 0 :
            self.create_writing_directory(self.parameters["WritingDirectory"])
        self.set_potential(potential)
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

        
    def set_potential(self,\
                    path:os.PathLike[str]|list[os.PathLike[str]],\
                    type="eam/fs",
                    species=None)->None:
        """Set potential pathway

        Parameters
        ----------
        path : os.PathLike[str] or list[os.PathLike[str]]
            path to potential file or list of files (for e.g. SNAP)
        """
        path = [path] if not isinstance(path,list) else path
        for p in path:
            if not os.path.exists(p):
                raise IOError(f"Potential {p} not found!")
        self.PotentialLocation = " ".join(path)
        self.PotentialType = type
        self.has_potential = True
        self.check()

    def set_default_scripts(self) -> None:
        """Set default values for the <Scripts> branch
        """
        self.scripts={}
        self.scripts["Input"] = """
            units metal
            atom_style atomic
            atom_modify map array sort 0 0.0
            read_data  %FirstPathConfiguration%
            pair_style    eam/fs
            pair_coeff * * %Potential% %Species%
            timestep       %DELTAT%
            thermo 10
        """

        self.scripts["NPTScript"] = """
        fix             npt_samp  all npt temp %TEMPERATURE% %TEMPERATURE% %DAMPT% iso 0.0 0.0 %DAMPP% fixedpoint 0.0 0.0 0.0
        """

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
        
    def seed(self,worker_instance:int)->None:
        """Generate random number seed

        Parameters
        ----------
        worker_instance : int
            unique to each worker, to ensure independent seed
        """
        if not self.seeded:
            self.randseed = self.parameters["GlobalSeed"] * (worker_instance+1)
            self.rng = np.random.default_rng(self.randseed)
            self.rng_int = self.rng.integers(low=100, high=10000)
            self.seeded=True

    def randint(self)->int:
        """Generate random integer.
            Gives exactly the same result each time unless reseed=True

        Returns
        -------
        int
            a random integer
        """
        if not self.seeded:
            print("NOT SEEDED!!")
            exit(-1)

        if self.parameters["FreshSeed"]:
            self.rng_int = self.rng.integers(low=100, high=10000)
        return str(self.rng_int)