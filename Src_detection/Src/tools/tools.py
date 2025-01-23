import os
import numpy as np

from typing import List, Dict
from difflib import SequenceMatcher

def BuildingFunction(path_dir : os.PathLike[str]) -> List[os.PathLike[str]] :
    """Build the list of directories from an initial directory

    Parameters 
    ----------

    Path_dir : os.PathLike[str]
        Path to the initial directory

    
    Returns 
    -------

    List[os.PathLike[str]]
        List of child directory from initial directory

    
    """
    return [dir for dir in os.listdir(path_dir) if os.path.isdir('%s/%s'%(path_dir,dir)) ]


def RecursiveCheck(path : os.PathLike[str], list : List[os.PathLike[str]], file2find : str = 'OUTCAR') -> None :
    """ This function builds recursively the list of path containing file2find
    
    Parameters 
    ----------

    path : os.PathLike[str]
        Initial path to check

    list : List[os.PathLike[str]]
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

def RecursiveBuilder(root_dir_check : os.PathLike[str], file2find : str = 'OUTCAR') -> List[str] :
    """Build recursively the list of all file2find paths
    
    Parameters 
    ----------

    root_dir_check : os.PathLike[str]
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


def nearest_mode(list_mode : List[str],
                 mode : str,
                 nb_poss : int) -> List[str]:
    """Naive function based on string metric to find nearest mode to a reference subset
    
    Parameters
    ----------

    list_mode : List[str]
        List of reference mode

    mode : str
        Mode to compare with references

    nb_poss : int 
        Number of closest mode in reference mode to print

    Returns 
    -------

    List[str]
        List of nearest modes found based on string metric
        
    """
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

def merge_dict_(dict : Dict[int,List[int]]) -> Dict[int,List[int]] :
    """Merge aggregation dictionnary for ```DislocationObject```
    
    Parameters
    ----------

    dict : Dict[int,List[int]]
        Dictionnary of aggregation containing redundant informations

    Returns
    -------

    Dict[int,List[int]]
        New aggreation dictionnary with redundancy 
    
    """
    key2del = []
    
    dict_m = dict.copy()
    for key1, val1 in dict_m.items() : 
        for key2, val2 in dict_m.items() : 
            if key1 < key2 and key1 not in key2del : 
                if key1 in val2 : 
                    key2del.append(key1)
                    val2 += val1

                else : 
                    for v in val1 : 
                        if v in val2 : 
                            key2del.append(key1)
                            val2 += val1 
                            break 

            else :
                continue
    for k2d in key2del : 
        del dict_m[k2d]

    for key in dict_m.keys() : 
        u_list = np.unique(dict_m[key]).tolist()
        dict_m[key] = u_list

    return dict_m