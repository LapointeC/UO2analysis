import os
import warnings
import shutil
import re
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple, Dict

from ase import Atoms
import numpy as np

class DBDictionnaryBuilder : 

    def __init__(self) : 
        self.dic : Dict[str,List[Atoms]] = {}

    def _builder(self, list_atoms : List[Atoms], list_sub_class : List[str] ) -> dict : 
        for id_at, atoms in enumerate(list_atoms) :
            self._update(atoms, list_sub_class[id_at])
        
        return self._generate_dictionnary()

    def _update(self, atoms : Atoms, sub_class : str) -> None : 
        if sub_class not in self.dic.keys() : 
            self.dic[sub_class] = [atoms]
        else : 
            self.dic[sub_class].append(atoms)     

    def _generate_dictionnary(self) -> dict :
        dictionnary = {}
        for sub_class in self.dic.keys() : 
            for id, atoms in enumerate(self.dic[sub_class]) : 
                name_poscar = '{:}_{:}'.format(sub_class,str(1000000+id+1)[1:])
                dictionnary[name_poscar] = {'atoms':atoms,'energy':None,'energy':None,'forces':None,'stress':None}

        return dictionnary


class GenerateMiladyInput:

    def __init__(self, Optimiser : dict, Regressor : dict, Descritpor : dict ,  Restart=None) : 
        self.optimiser_param = Optimiser
        self.regression_param = Regressor
        self.descriptor = Descritpor

    def check_descriptor(self) : 
        if self.descriptor is None :
            raise ValueError("Descriptors setup is not defined !")
    
    def convert_problematic_values(self, dict : dict) -> dict : 
        
        dict_copy = dict.copy()
        for key in dict.keys() : 
            if isinstance(dict[key],list) :
                str2keep = '' 
                for el in dict[key] :
                    if isinstance(el,str) : 
                        str2keep += '%s '%(el)
                    elif isinstance(el,int) : 
                        str2keep += '%2d '%(el)
                    elif isinstance(el,float) : 
                        str2keep += '%3.8f '%(el)

                dict_copy[key] = str2keep

        return dict_copy
        #To do !   

    def copy_standard_ml_files(self, directory : str = '', label : str = 'milady') -> None :
        label = os.path.basename(label) 
        template_dir = os.path.join(os.path.dirname(__file__),'templates')
        shutil.copy(os.path.join(template_dir,'eamtab.potin'),directory)
        shutil.copy(os.path.join(template_dir,'config.din'),os.path.join(directory,'%s.din'%(label)))
        shutil.copy(os.path.join(template_dir,'config.gin'),os.path.join(directory,'%s.gin'%(label)))
        
        with open(os.path.join(directory,'name.in'),'w') as f : 
            f.write('%s \n'%(label))
        
        return None

    def write_ml_file(self, directory : str = './', name_file : str = 'milady') -> None : 

        def boolean_converter(bool : bool) -> str : 
            if bool : 
                return '.true.'
            else :
                return '.false.'

        def float_format_converter(float : float) -> str :
            return re.sub('e','E','%1.9e'%(float))

        name_file = os.path.basename(name_file)  
        
        with open('%s/%s.ml'%(directory,name_file), 'w') as ml_file : 
            ml_file.write('& input_ml\n')


            all_dictionnaries = [self.optimiser_param, self.regression_param, self.descriptor]
            corresponding_text_line = ['!OPTIMISER \n', '!REGRESSION \n', '!DESCRIPTOR \n']

            for id_dict, dictionnary in enumerate(all_dictionnaries) : 
                dictionnary = self.convert_problematic_values(dictionnary)
                ml_file.write(corresponding_text_line[id_dict])
                ml_file.write('\n')
                for key in dictionnary.keys() :
                    if isinstance(dictionnary[key],bool) : 
                        txt = '%s=%s \n'%(key,boolean_converter(dictionnary[key]))
                    elif isinstance(dictionnary[key],int) :
                        txt = '%s=%4d \n'%(key,dictionnary[key])
                    elif isinstance(dictionnary[key],float) : 
                        txt = '%s=%s \n'%(key,float_format_converter(dictionnary[key])) 
                    elif isinstance(dictionnary[key],str) : 
                        txt = '%s="%s" \n'%(key,dictionnary[key])

                    ml_file.write(txt)

                ml_file.write('\n')

            ml_file.write('\n')
            ml_file.write('&end')