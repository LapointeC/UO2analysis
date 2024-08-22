from __future__ import annotations
import numpy as np
import os,glob
import xml.etree.ElementTree as ET
import numpy as np
from typing import Union,Any,List
ScriptArg = Union[int,float,str]
from ..results.ResultsBABF import ResultsBABF

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
        
        Methods
        ----------
        __call__
        find_suffix_and_write
        

        Raises
        ------
        IOError
            If path is not found
        """
        self.set_default_parameters()
        self.set_default_scripts()
        self.xml_path = xml_path
        self.rank=rank
        self.PotentialLocation = None
        self.Species = None
        self.Configuration = None
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
            raise TimeoutError('No xml file for BABF...')
        
    def check(self)->None:
        """Internal check of parameters
        """
        if self.has_potential and self.has_species:
            self.check_output_location()
            if self.check_configuration() : 
                return True
            else : 
                return False
        return False
    
    def ready(self)->bool:
        """Check if all parameters are set

        Returns
        -------
        bool
        """
        if not self.has_potential:
            raise TypeError("\n\nNo Potential set!\n")
        if not self.has_species:
            raise TypeError("\n\nNo Species set!\n")
        if not self.has_out:
            raise TypeError("\n\nNo output location set!\n")
        if not self.check_configuration() :
            raise TypeError("\n\nIntial configuration file does not exist!\n")

        return True
            
    def check_output_location(self)->None:
        """
            Ensure working directory exists
        """
        # dump data
        df = self.parameters["WorkingDirectory"]
        self.found_output_dir = os.path.isdir(df)
        if not self.found_output_dir:
            try:
                os.mkdir(df)
                self.has_out = True
            except Exception as e:
                print(f"Working directory {df} cannot be made!",e)
                self.has_out = False
        
    def check_configuration(self) -> bool : 
        """
            Ensure configuration file exists
        """        
        configuration = self.parameters["Configuration"]
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
        self.parameters["CoresPerWorker"] = 1
        self.parameters["Verbose"] = 0
        self.parameters["Temperature"] = 10.0 
        self.parameters["LambdaGrid"] = {"Min":0.0,"Max":1.0,"Number":100,"Rbuffer":0.0,"NumberBuffer":0}        
        self.parameters["ThermalisationStep"] = 1000
        self.parameters["StochasticSteps"] = 10000
        self.parameters["WritingStep"] = 2000
        self.parameters["GatherStep"] = 1000
        self.parameters["WorkingDirectory"] = './mab_work'
        self.parameters["Configuration"] = 'to_set'

        self.parameters["Species"] = 'Fe'
        self.parameters["Dynamic"] = "OverdampedLangevin"
        self.parameters["DetlaT"] = 0.1
        self.parameters["Friction"] = 0.05
        self.parameters["ReferenceModel"] = {"model":'Einstein','omega':1.0}
        self.parameters["ConstrainedPotential"] = {"model":'Standard','C':1.0,'delta':1.0}
        self.parameters["Block"] = False

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
            elif var.tag == "Species" : 
                species = var.text.strip()

        self.set_species(species)
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
                if isinstance(o,int):
                    self.parameters[tag] = int(n)
                elif isinstance(o,float):
                    self.parameters[tag] = float(n)
                elif isinstance(o,bool):
                    self.parameters[tag] = boolean_converter(n)
                elif isinstance(o,str):
                    self.parameters[tag] = n
                elif isinstance(o,list):
                    self.parameters[tag] = n
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
        if not species is None:
            self.set_species(species)
        self.check()

    def set_species(self,species:str|List[str])->None:
        """Set element list
            Must be in correct order for LAMMPS to read !
            TODO: check this? 

        Parameters
        ----------
        species : str | List[str]
            string or list of species
        """
        if isinstance(species,str):
            species = [species]
        assert min([isinstance(s,str) for s in species])
        self.Species = species
        self.has_species = True
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
            run 0
            thermo 10
            run 0
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
                
    
    def as_Element(self) -> ET.Element:
        """Cast all parameters as xml.etree.ElementTreeElement

        Returns
        -------
        xml.etree.ElementTree.Element
            Data as xml.etree.ElementTree
        """
        xmlET = ET.Element("PAFI")
        def add_branch(key:str,data:Any):
            """Add Axes, Parameter and Script data

            Parameters
            ----------
            key : str
                key
            data : Any
                data
            """
            branch = ET.Element(key)
            for key, val in data.items():
                child = ET.Element(key)
                if isinstance(val,list):
                    child.text = ""
                    for v in val:
                        child.text += str(v)+" "
                elif isinstance(val,np.ndarray):
                    child.text = ""
                    for v in val:
                        child.text += "%3.3g " % float(v)
                else:
                    child.text = str(val)
                branch.append(child)
            xmlET.append(branch)
        
        add_branch('Parameters',self.parameters)
        add_branch('Scripts',self.scripts)
        
        """
            Add PathwayConfiguration data
        """
        branch = ET.Element("MABCalculation")
        
        branch_branch = ET.Element("Potential")
        branch_branch.text = self.PotentialLocation
        branch.append(branch_branch)

        branch_branch = ET.Element("Species")
        branch_branch.text = " ".join(self.Species)
        branch.append(branch_branch)


        
        
        return xmlET

    def to_string(self) -> str:
        """
            Return all paramaters as XML string
        """
        Element = self.as_Element()
        ET.indent(Element, '  ')
        return str(ET.tostring(Element)).replace('\\n','\n')
    
    def to_xml_file(self,xml_file:None|str=None)->None:
        """Write all paramaters as XML file
        
        Parameters
        ----------
        xml_file : str, optional
            path to XML file, default None
        """
        if xml_file is None:
            xml_file = self.xml_file
        assert not xml_file is None
        print(f"""
                Writing PAFI config file {xml_file}
                Will write PAFI data to {self.csv_file}
                """)
            
        Element = self.as_Element()
        ElementTree = ET.ElementTree(Element)
        ET.indent(ElementTree, '  ')
        ElementTree.write(xml_file,encoding="utf-8", xml_declaration=True)
        
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
                     arguments:None|dict|ResultsBABF=None) -> str:
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
        elif isinstance(arguments,ResultsBABF):
            _args = arguments.data.copy()
        else:
            _args = arguments.copy()
        _args["Configuration"] = self.Configuration
        if not self.PotentialLocation is None:
            _args["Potential"] = self.PotentialLocation
        if not self.Species is None:
            _args["Species"] = " ".join(self.Species)
        for key,value in _args.items():
            script = self.replace(script,key,value)
        return script
    
    
    def welcome_message(self):
        """
            A friendly welcome message :)
        """

        welcome = f"""        
                ┳┳┓  ┓ 
            ┏┓┓┏┃┃┃┏┓┣┓
            ┣┛┗┫┛ ┗┗┻┗┛
            ┛  ┛                                                               
        
        copyleft CEA by ...                                        
        ... C. Lapointe, T.D.S Swinburne ,M. Athenes, M.-C. Marinica     
        other contributions                                        
        L. Cao, G. Stoltz, T. Lelievre                             
        email:mihai-cosmin.marinica@cea.fr, clovis.lapointe@cea.fr
        

        Working Directory:
            {self.parameters['WorkingDirectory']}
        Configuration:
            {self.Configuration}
        
        """
        
        welcome += """
        Scripts:
        """
        for key,val in self.scripts.items():
            welcome += f"""
                {key}: 
                    {val}"""
        
        welcome += """
        Parameters:
        """
        for key,val in self.parameters.items():
            if isinstance(self.parameters[key],dict) : 
                dictionnary = self.parameters[key]
                val = ''.join(['{} = {}'.format(k,dictionnary[k]) for k in dictionnary.keys()])
            welcome += f"""
                {key}: {val}"""
        return welcome
    

        