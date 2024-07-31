import glob
import numpy as np
import os
import pickle
import warnings
import itertools
import struct

from typing import Dict, TypedDict, List, Tuple, Any
from scipy.io import FortranFile, _fortran
from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read

import h5py
from h5py import Group

def add_or_update_dynamical(hdf5_group : Group, dyn_index : str, dynamical_matrix : np.ndarray, positions : np.ndarray, cell : np.ndarray, elements : str) -> None :
    dyn_group_name = str(dyn_index)

    if dyn_group_name not in hdf5_group:
        # Create group for the dynamical matrix if it doesn't exist
        dyn_group = hdf5_group.create_group(dyn_group_name)
        dyn_group.create_dataset("dynamical_matrix", data=dynamical_matrix, compression="gzip", compression_opts=9)
        dyn_group.create_dataset("positions", data=positions, compression="gzip", compression_opts=9)
        dyn_group.create_dataset("cell", data=cell, compression="gzip", compression_opts=9)
        #dyn_group.create_dataset("elements", data=elements_array, compression="gzip", compression_opts=9, shape=(len(elements_array),))

    else : 
        hdf5_group[dyn_index].create_dataset("dynamical_matrix", data=dynamical_matrix, compression="gzip", compression_opts=9)
        hdf5_group[dyn_index].create_dataset("positions", data=positions, compression="gzip", compression_opts=9)
        hdf5_group[dyn_index].create_dataset("cell", data=cell, compression="gzip", compression_opts=9)
        #hdf5_group[dyn_index].create_dataset("elements", data=elements_array, compression="gzip", compression_opts=9, shape=(len(elements_array),))
    
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


class Data(TypedDict) : 
    dynamical_matrix : np.ndarray
    atoms : Atoms 
    inputs_lammps : str

class DataPhondy : 
    def __init__(self, root_dir : os.PathLike[str]) -> None : 
        self.Data : Dict[str,Data] = {}
        self.root_dir = root_dir
    def GenerateDataParallel(self, hdf5 : Group, njob : int = 1) -> None : 
        list_all_calculations = RecursiveBuilder(self.root_dir, file2find='in.lmp')
        for path_calculation in list_all_calculations : 
            print('Extracting data : {:s}'.format(path_calculation))
            #try : 
            self.UpdateDataParallel(path_calculation, hdf5, njob = njob) 
            #except : 
            #    print('Something wrong with : {:s}'.format(path_calculation))       

    def GenerateData(self) : 
        list_all_calculations = RecursiveBuilder(self.root_dir, file2find='in.lmp')
        for path_calculation in list_all_calculations : 
            try : 
                self.UpdateData(path_calculation)
            except : 
                print('Something wrong with : {:s}'.format(path_calculation))

    def GatherDynamicalMatrix(self, directory : os.PathLike[str], nproc : int = 1, name_matrix : str = '100000.', name_gather_matrix : str = 'dyna') -> None : 
        u, v, m = self.read_phondy_matrix_multi_proc_draft('{:s}/{:s}'.format(directory,name_matrix), njob=nproc)
        self.write_binary_file('{:s}.u'.format(name_gather_matrix), 'i4', u)
        self.write_binary_file('{:s}.v'.format(name_gather_matrix), 'i4', v)
        self.write_binary_file('{:s}.m'.format(name_gather_matrix), 'f8', m)
        return 

    def WritePickle(self, path2write : str = './') -> None :
        if os.path.exists('{:s}/phondy.pickle'.format(path2write)) : 
            os.remove('{:s}/phondy.pickle'.format(path2write))

        if self.Data is None : 
            warnings.warn('Data dictionnary is empty !')
        pickle.dump(self.Data, open('{:s}/phondy.pickle'.format(path2write),'wb'))

    def UpdateData(self, path : str, hdf5 : Group, name_matrix : str = '100000.', name_lmp : str = 'in.lmp', inputs_lammps : str = 'in.lammps') -> None :
        dynamical_matrix = self.read_phondy_matrix('{:s}/{:s}'.format(path,name_matrix))
        atoms = read('{:s}/{:s}'.format(path,name_lmp),format='lammps-data',style='atomic')
        inputs_lammps_file = self.read_inputs_lammps('{:s}/{:s}'.format(path,inputs_lammps))
        self.Data[os.path.dirname(path)] = {'atoms':atoms,'dynamical_matrix':dynamical_matrix,'inputs_lammps':inputs_lammps_file}
        add_or_update_dynamical(hdf5, os.path.basename(path), dynamical_matrix, atoms.positions, atoms.cell[:], atoms.symbols)

    def UpdateDataParallel(self, path : str, hdf5 : Group, name_lmp : str = 'in.lmp', inputs_lammps : str = 'in.lammps', njob : int = 1) -> None :
        dynamical_matrix = self.read_phondy_matrix_multi_proc(path, njob=njob)
        atoms = read('{:s}/{:s}'.format(path,name_lmp),format='lammps-data',style='atomic')
        atoms = self.change_all_symbols(atoms, 'Fe')
        inputs_lammps_file = self.read_inputs_lammps('{:s}/{:s}'.format(path,inputs_lammps))
        self.Data[os.path.dirname(path)] = {'atoms':atoms,'dynamical_matrix':dynamical_matrix,'inputs_lammps':inputs_lammps_file}
        add_or_update_dynamical(hdf5, os.path.basename(path), dynamical_matrix, atoms.positions, atoms.cell[:], atoms.get_chemical_formula())

    def change_all_symbols(self, atoms : Atoms, symbol : str) -> Atoms : 
        atoms.set_chemical_symbols([symbol] * len(atoms))
        return atoms

    def read_inputs_lammps(self, path : str) -> str : 
        return open(path,'r').read()

    def read_bin(self, file : str , ext : str) -> np.ndarray : 
        fmat = FortranFile(file,'r')
        list_data = []

        #fucking first index ...
        if ext == 'm' :
            fmat.read_ints()

        while True :
            if ext == 'u' or ext == 'v' : 
                try : 
                    list_data.append( fmat.read_ints(dtype='int32')[0] )
                except _fortran.FortranEOFError :
                    break
            else :
                try : 
                    list_data.append(fmat.read_reals(dtype='float64')[0] )
                except _fortran.FortranEOFError : 
                    break
        return np.array(list_data)

    def write_binary_file(self, name_file : os.PathLike[str], encoding : str, data : np.ndarray) -> None : 
        binary_data = struct.pack(encoding,data)
        with open(name_file,'wb') as w :
            w.write(binary_data)
        return 

    def read_phondy_matrix(self, path : str) -> np.ndarray :   
        matrix_list = glob.glob('{:s}*'.format(path))

        dic_matrix : Dict[str,np.ndarray] = {}
        for mat in matrix_list :    
            dic_matrix[mat[-1]] = self.read_bin(mat,mat[-1])

        number_of_mode = int(np.amax(dic_matrix['u'])) 
        dynamical_matrix = np.zeros( (number_of_mode,number_of_mode), dtype=np.float64)
        for id in range(len(dic_matrix['u'])) : 
            i = dic_matrix['u'][id] - 1
            j = dic_matrix['v'][id] - 1 
            dij = dic_matrix['m'][id]
            dynamical_matrix[i,j] = dij

        return dynamical_matrix
    
    
    def read_bin_multi_proc(self, file : str, idx : int) -> np.ndarray : 
        ext = file[-1]
        f = open(file, 'rb')
        if ext == 'm' : 
            dt_float = np.dtype('f8')
            np.fromfile(f, dtype=dt_float, offset=4,count=1)
            return np.fromfile(f, dtype=dt_float, offset=4,count=idx)
        if ext == 'u' or ext == 'v' :
            dt_int = np.dtype('i4')
            np.fromfile(f, dtype=dt_int,count=1)
            return np.fromfile(f, dtype=dt_int,count=idx)

    def read_imax(self, file_m : str) -> int : 
        with open(file_m, 'rb') as file:
            np.fromfile(file, dtype=np.int32, count=1)
            return np.fromfile(file, dtype=np.int32, count=1)[0]

    def read_phondy_matrix_multi_proc(self, path : str, njob : int = 1) -> np.ndarray :  
        matrix_list_m = glob.glob('{:s}/*.m'.format(path))
        index_list = Parallel(n_jobs=njob)(delayed(self.read_imax)(mat) for mat in matrix_list_m)
        matrix_m = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)(mat,idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_u = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}u'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_v = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}v'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))

        matrix_u = list(itertools.chain.from_iterable(matrix_u))
        matrix_v = list(itertools.chain.from_iterable(matrix_v))
        matrix_m = list(itertools.chain.from_iterable(matrix_m))

        matrix_size = int(np.amax(matrix_u))
        dynamical_matrix = np.zeros( (matrix_size,matrix_size), dtype=np.float64)
        
        array_u = np.array(matrix_u) - 1
        array_v = np.array(matrix_v) - 1
        array_m = np.array(matrix_m) 
        dynamical_matrix[array_u,array_v] = array_m

        print(np.linalg.norm(dynamical_matrix.T - dynamical_matrix))

        return dynamical_matrix
    
    def read_phondy_matrix_multi_proc_draft(self, path : str, njob : int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :  
        matrix_list_m = glob.glob('{:s}/*.m'.format(path))
        index_list = Parallel(n_jobs=njob)(delayed(self.read_imax)(mat) for mat in matrix_list_m)
        matrix_m = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)(mat,idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_u = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}u'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_v = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}v'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))

        matrix_u = list(itertools.chain.from_iterable(matrix_u))
        matrix_v = list(itertools.chain.from_iterable(matrix_v))
        matrix_m = list(itertools.chain.from_iterable(matrix_m))

        return np.array(matrix_u), np.array(matrix_v), np.array(matrix_m)

#root_dir = '/home/lapointe/WorkML/GenerateThermalConfig/phondy_matrix/para_mat'
#root_dir = '/home/lapointe/WorkML/GenerateThermalConfig/phondy_matrix/stream_mat'
root_dir = '/home/lapointe/WorkML/GenerateThermalConfig/phondy_matrix/mat_JZ'
file = h5py.File('dynamical.h5', 'w')
# Create a group for potentials
dynamical_group = file.create_group('dynamical')


obj_phondy = DataPhondy(root_dir)
obj_phondy.GenerateDataParallel(dynamical_group,njob=2)
#obj_phondy.WritePickle(path2write=os.getcwd())

#with h5py.File('dynamical.h5', 'r') as file:
#    dynamical_group = file['dynamical']
#    for key in dynamical_group.keys() : 
#        for quantity in dynamical_group[key].keys() : 
#            print(dynamical_group[key][quantity][:])