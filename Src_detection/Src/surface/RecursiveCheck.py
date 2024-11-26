import os
from .ExtractAllFromVASP import *

from typing import List

def WritingLog(log : str, name_log : str) -> None :
    """Writing the log file 
    
    Parameters
    ----------

    log : str
        Line to write in the log file 

    name_log : str
        Path to the log file 
    
    """
    if log == 'ini':
        log_f = open(name_log,'a')
        log_f.write('Here is the log file of checking calculation \n')
        log_f.write('\n')
        log_f.close()

    else :
        log_f = open(name_log,'a')
        log_f.write('%s \n'%(log))
        log_f.close()

    return


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



def RecursiveChecker(root_dir_check : str, log_file : str, file2find : str = 'OUTCAR') -> List[str] :
    """Recursively check the convergence of all calculations directory and return the list of unconverged
    directory
    
    Parameters 
    ----------

    root_dir_check : str
        Root of all paths to check

    log_file : str
        Path to the log file

    file2find : str
        Name of the file to find 

    Returns 
    -------

    List[str]
        List of unconverged directories

    """
    if os.path.exists(log_file):
        os.system('rm %s'%(log_file))

    WritingLog('ini',log_file)
    list_path_outcar = []
    list_non_converged_outcar = []
    RecursiveCheck(root_dir_check,list_path_outcar,file2find)

    for path_outcar in list_path_outcar :
        log_text = 'Directory %s ==> '%(path_outcar)
        path_poscar = '%s/POSCAR'%(path_outcar)
        path_outcar += '/OUTCAR'
        bool_acc, bool_struc_min = CheckConvergence(path_outcar)
        if bool_struc_min :
            log_text += 'accuracy is reached ! '

            #Extracting everything from outcar
            try :
                e = GetEnergyFromVasp(path_outcar)
                list_el, nb_el = GetElementsFromVasp(path_poscar)
                log_text += 'elements <=>'  
                for k, el in enumerate(list_el) : 
                    log_text += '%s %2d, '%(el,nb_el[k])
                log_text += '<=> VASP energy >=> %5.8f eV 1'%(e)
                WritingLog(log_text,log_file)
            except :
                log_text += 'Energy value error 0'
                WritingLog(log_text,'check_conv.log')
                list_non_converged_outcar.append(path_outcar)

        else :
            log_text += 'accuracy is not reached ... 0'
            WritingLog(log_text,log_file)
            list_non_converged_outcar.append(path_outcar)

    return list_non_converged_outcar


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

def FindPattern(file : str, pattern : str) -> bool :
    """Find a pattern of str in a file, if the pattern is found bool=True
    otherwise bool=False
    
    Parameters
    ----------

    file : str
        Path to the file read

    pattern : str
        Pattern to find 

    Returns 
    -------

    bool 
        True if the pattern is found and False otherwise

    """
    bool = False
    r = open(file,'r').readlines()
    for l in r :
        if pattern in l :
            bool = True
            break

    return bool

def CheckProblems(list_path : str) -> None :
    """Here is a little function to check if there some issues with calculation by looking at 
    output files. ! dic_pattern have to be extended when you find new possible problems and corresponding
    pattern in output file :) !
    
    Parameters
    ----------

    List of paths to check

    """
    dic_pattern = {'Sub-Space-Matrix is not hermitian':'!NO HERMITIAN MATIX ==> Memory problem ?'}
    for path in list_path :
        path = os.path.dirname(path)
        file_out = [ f for f in os.listdir(path) if '.out' in f ]
        if len(file_out) == 0 :
            print('No file out for %s'%(path))
            continue

        for pat in dic_pattern.keys() :
            bool_pat = FindPattern('%s/%s'%(path,file_out[0]),pat)
            if bool_pat :
                print('Problem with %s : %s'%(path,dic_pattern[pat]))

    return