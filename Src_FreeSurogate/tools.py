import os
from typing import List
from difflib import SequenceMatcher
import numpy as np

from ase import Atoms
from typing import Tuple

def BuildingFunction(path_dir : str) -> List[str] :
    """Build the list of directories from an initial directory

    Parameters 
    ----------

    Path_dir : str
        Path to the initial directory

    
    Returns 
    -------

    List[str]
        List of child directory from initial directory

    
    """
    return [dir for dir in os.listdir(path_dir) if os.path.isdir('%s/%s'%(path_dir,dir)) ]


def RecursiveCheck(path : str, list : List[str], file2find : str = 'OUTCAR') -> None :
    """ This function builds recursively the list of path containing file2find
    
    Parameters 
    ----------

    path : str
        Initial path to check

    list : List[str]
        List which is built recursively to contain the list of path containing file2find
    
    file2find : str 
        Name of the file to find
    
    """   
    if os.path.exists('%s/%s'%(path,file2find)) and (BuildingFunction(path) == []):
        list.append(path)

    else :
        add_path = BuildingFunction(path)
        for el in add_path :
            new_path = '%s/%s'%(path,el)
            RecursiveCheck(new_path,list,file2find)

def RecursiveBuilder(root_dir_check : str, file2find : str = 'OUTCAR') -> List[str] :
    """Build recursively the list of all file2find paths
    
    Parameters 
    ----------

    root_dir_check : str
        Root of all paths to check

    file2find : str
        Name of the file to find   

    Returns 
    -------

    List[str]
        List of all file2find paths

    """
    list_path_outcar = []
    RecursiveCheck(root_dir_check,list_path_outcar,file2find)

    return list_path_outcar


def nearest_mode(list_mode,mode,nb_poss):
    list_score = []
    for mode_implemented_i in list_mode :
        score_i = 0.0
        compt = 0
        if len(mode.split('_')) > 1 :
            words_mode = mode.split('_')
            words = mode_implemented_i.split('_')
            for word_mode in words_mode :
                for word in words :
                    compt += 1
                    score_i += SequenceMatcher(None,word,word_mode).ratio()

        else :
            words = mode_implemented_i.split('_')
            for word in words :
                compt += 1
                score_i += SequenceMatcher(None,word,mode).ratio()

        score_i *= 1.0/compt
        list_score.append(score_i)

    """sorting"""
    for id_1,score_1 in enumerate(list_score) :
        for id_2, score_2 in enumerate(list_score) :
            if score_1 > score_2 :
                list_score[id_1], list_score[id_2] = list_score[id_2], list_score[id_1]
                list_mode[id_1], list_mode[id_2] = list_mode[id_2], list_mode[id_1]

    return list_mode[:nb_poss]

def compute_ratio_formation(atoms_dfct : Atoms) -> Tuple[float, float] :
    nb_atom_dfct = len(atoms_dfct)
    nb_atom_bulk = 2.0*np.power(int(np.power(nb_atom_dfct/2.0,1.0/3.0)),3)
    return nb_atom_dfct, nb_atom_bulk