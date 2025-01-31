from __future__ import annotations
import numpy as np
import os, shutil
import xml.etree.ElementTree as ET
import numpy as np
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