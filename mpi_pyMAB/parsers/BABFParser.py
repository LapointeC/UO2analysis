from __future__ import annotations
import numpy as np
import os
import numpy as np
from typing import Union,Any
ScriptArg = Union[int,float,str]

from .BaseParser import BaseParser

class BABFParser(BaseParser):
    """Default reader of BABF XML configuration file
    
    Parameters
    ----------
    
    xml_path : os.PathLike[str]
        path to XML file, default None
    
    postprocessing: bool, optional
        are we just looking in postprocess?, default False
    
    Methods
    ----------
    
    __call__
    find_suffix_and_write
    

    Raises
    ------
    
    IOError
        If path is not found
    """
    def __init__(self,
            xml_path:None|os.PathLike[str]=None,
            rank:int=0) -> None:
        super().__init__(xml_path,rank)

        # initial seed, but must be different across workers...
        self.seeded = False
               
        
    def set(self,key:str,value:Any,create:bool=False)->None:
        """Set a parameter

        Parameters
        ----------
        
        key : str
            key of parameter. Must already exist if create is `False`
        
        value : Any
            value for entry
        
        create : bool, optional
            Create new entry if true    
        """
        if not create:
            assert key in self.parameters.keys()
        self.parameters[key] = value
    
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
    
    
    def info(self)->str:
        """Return all parameters as formatted string

        Returns
        -------
        
        str
            the parameter string
        """
        result = """
            BABF parameters information:
                    
            Scripts:"""
        for k,v in self.scripts.items():
            result += f"""
                {k} : {self.parse_script(v)}"""
        result += f"""

            Wildcard Potential:
                {self.PotentialLocation}
            
            Wildcard Elements:
                {self.Species}

            Pathway:
                Working Directory:
                    {self.parameters["WorkingDirectory"]}
                Configuration:
                    {self.Configuration}
                """
        
        result += """


            Parameters"""
        for k,v in self.parameters.items():
            #DO SOMETHING FOR DIRECTORY !
            if isinstance(self.parameters[k],dict) : 
                dictionnary = self.parameters[k]
                v = ''.join(['{} = {}'.format(k,dictionnary[k]) for k in dictionnary.keys()])
            result += f"""
                {k} : {v}"""
        
        result += f"""
                seeded : {self.seeded}"""
        result += f"""

        """
        return result
    
    def to_dict(self) -> dict:
        """Export axes and parameters as a nested dictionary

        Returns
        -------
        
        dict
            Dictionary-of-dictionaries with two keys 'axes' and 'parameters'
        """
        return super().to_dict()
    
    

        

            


