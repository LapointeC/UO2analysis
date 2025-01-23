import glob
import numpy as np
import os
import pickle
import warnings
import itertools
import struct

from typing import Dict, TypedDict, Tuple
from joblib import Parallel, delayed
from .tools import RecursiveBuilder

from ase import Atoms
from ase.io import read

import h5py
from h5py import Group

def add_or_update_dynamical(hdf5_group : Group, 
                            dyn_index : str, 
                            dynamical_matrix : np.ndarray, 
                            positions : np.ndarray, 
                            cell : np.ndarray, elements : str) -> None :
    """Update method for dynamical matrix h5 files
    
    Parameters
    ----------

    hdf5_group : ```Group```
        hd5 group to update 

    dyn_index : str
        Name of the index to add/update in h5

    dynamical_matrix : np.ndarray
        Dynamical matrix of the system

    positions : np.ndarray 
        Positions array of the system 

    cell : np.ndarray 
        Cell associated to the system

    elements : str 
        Elements in the system => to be delete or think about different storage
    """
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

class Data(TypedDict) : 
    dynamical_matrix : np.ndarray
    atoms : Atoms 
    inputs_lammps : str

class DataPhondy : 
    def __init__(self, root_dir : os.PathLike[str]) -> None : 
        """Init method for ```DataPhondy``` object 
        
        Parameters
        ----------

        root_dir : os.PathLike[str]
            Root directory to find all binary files from phondy
            Path for each binary file will be extracted recursively

        """
        self.Data : Dict[str,Data] = {}
        self.root_dir = root_dir
    
    def GenerateDataParallel(self, hdf5 : Group, njob : int = 1) -> None : 
        """Fill the hdf5 group parallely by reading binary files from phondy
        
        Parameters
        ----------

        hdf5 : ```Group```
            hdf5 group to update 

        njob : int 
            Number of parallel jobs for storage
        """
        list_all_calculations = RecursiveBuilder(self.root_dir, file2find='in.lmp')
        for path_calculation in list_all_calculations : 
            print('Extracting data : {:s}'.format(path_calculation))
            self.UpdateDataParallel(path_calculation, hdf5, njob = njob) 
       
    def GenerateData(self) : 
        """Fill the hdf5 group serialy by reading binary files from phondy"""
        list_all_calculations = RecursiveBuilder(self.root_dir, file2find='in.lmp')
        for path_calculation in list_all_calculations : 
            try : 
                self.UpdateData(path_calculation)
            except : 
                print('Something wrong with : {:s}'.format(path_calculation))

    def GatherDynamicalMatrix(self, directory : os.PathLike[str], 
                              nproc : int = 1, 
                              name_matrix : str = '100000.', 
                              name_gather_matrix : str = 'dyna') -> None : 
        """Gathering method for phondy matrices
        Binary files containing sub matrix from parallel phondy calculations will be gather as one

        Parameters
        ----------

        directory : os.PathLike[str]
            Directory path of initial binary files 

        nproc : int
            Number of processor for parallel reading 

        name_matrix : str 
            Name of initial bianry matrix files 

        name_gather_matrix : str 
            New name for the gathered matrices
        """

        u, v, m, idx = self.read_phondy_matrix_multi_proc_draft('{:s}/{:s}'.format(directory,name_matrix), njob=nproc)
        self.write_binary_file('{:s}.u'.format(name_gather_matrix), 'i4', idx, u)
        self.write_binary_file('{:s}.v'.format(name_gather_matrix), 'i4', idx, v)
        self.write_binary_file('{:s}.m'.format(name_gather_matrix), 'f8', idx, m)
        return 

    def WritePickle(self, path2write : os.PathLike[str] = './') -> None :
        """Writing method, data are stored in .pkl file
        
        Parameters
        ----------

        path2write : os.PathLike[str]
            Path to write the pickle file 
        """
        if os.path.exists('{:s}/phondy.pickle'.format(path2write)) : 
            os.remove('{:s}/phondy.pickle'.format(path2write))

        if self.Data is None : 
            warnings.warn('Data dictionnary is empty !')
        pickle.dump(self.Data, open('{:s}/phondy.pickle'.format(path2write),'wb'))
        return 

    def UpdateData(self, path : os.PathLike[str], 
                   hdf5 : Group, 
                   name_matrix : str = '100000.', 
                   name_lmp : str = 'in.lmp', 
                   inputs_lammps : str = 'in.lammps') -> None :
        """Serial updating data by reading phondy binaires
        
        Parameters
        ----------

        path : os.PathLike[str]
            Directory path of original binary files 

        name_matrix : str
            Name of the binary files

        name_lmp : str 
            Name of the geomtry file for lammps 

        inputs_lammps : str
            Name of the inputs file for lammps
        """
        dynamical_matrix = self.read_phondy_matrix('{:s}/{:s}'.format(path,name_matrix))
        atoms = read('{:s}/{:s}'.format(path,name_lmp),format='lammps-data',style='atomic')
        inputs_lammps_file = self.read_inputs_lammps('{:s}/{:s}'.format(path,inputs_lammps))
        self.Data[os.path.dirname(path)] = {'atoms':atoms,'dynamical_matrix':dynamical_matrix,'inputs_lammps':inputs_lammps_file}
        add_or_update_dynamical(hdf5, os.path.basename(path), dynamical_matrix, atoms.positions, atoms.cell[:], atoms.symbols)
        return 

    def UpdateDataParallel(self, path : os.PathLike[str],
                           hdf5 : Group, 
                           name_lmp : str = 'in.lmp', 
                           inputs_lammps : str = 'in.lammps', 
                           njob : int = 1) -> None :
        """Parallel updating data by reading phondy binaires
        
        Parameters
        ----------

        path : os.PathLike[str]
            Directory path of original binary files 

        name_matrix : str
            Name of the binary files

        name_lmp : str 
            Name of the geomtry file for lammps 

        inputs_lammps : str
            Name of the inputs file for lammps

        njob : int 
            Number of processors for parallel reading
        """
        dynamical_matrix = self.read_phondy_matrix_multi_proc(path, njob=njob)
        atoms = read('{:s}/{:s}'.format(path,name_lmp),format='lammps-data',style='atomic')
        atoms = self.change_all_symbols(atoms, 'Fe')
        inputs_lammps_file = self.read_inputs_lammps('{:s}/{:s}'.format(path,inputs_lammps))
        self.Data[os.path.dirname(path)] = {'atoms':atoms,'dynamical_matrix':dynamical_matrix,'inputs_lammps':inputs_lammps_file}
        add_or_update_dynamical(hdf5, os.path.basename(path), dynamical_matrix, atoms.positions, atoms.cell[:], atoms.get_chemical_formula())
        return 

    def change_all_symbols(self, atoms : Atoms, symbol : str) -> Atoms : 
        """Little method to change species in ```Atoms``` object
        
        Parameters
        ----------

        atoms : ```Atoms```
            ```Atoms``` object to update

        symbol : str 
            New species to set

        Returns
        -------

        ```Atoms```
            Updated ```Atoms``` object

        """
        atoms.set_chemical_symbols([symbol] * len(atoms))
        return atoms

    def read_inputs_lammps(self, path : os.PathLike[str]) -> str : 
        """Read the input file from lammps
        
        Parameters
        ----------

        path : os.PathLike[str]
            Path to the lammps input file 

        Returns
        -------

        str 
            Inputs file convert into string
        """
        return open(path,'r').read()

    def write_binary_file(self, name_file : os.PathLike[str], 
                          encoding : str, 
                          index : int, 
                          data : np.ndarray) -> None : 
        """Write a binary file from data 
        
        Parameters
        ----------

        name_file : os.PathLike[str]
            Path to write the binary file

        encoding : str 
            Enconding type of data (i4, f8...)
        
        index : int
            Index number to write at the beginning of the bianary file 

        data : np.ndarray 
            Data to binarise
        """
        binary_idx = struct.pack('i4',index)
        binary_data = struct.pack(encoding,data)
        with open(name_file,'wb') as w :
            w.write(binary_idx)
            w.write(binary_data)
        return 

    def read_phondy_matrix(self, path : os.PathLike[str]) -> np.ndarray :   
        """Serial reading for phondy matrices 
        
        Parameters 
        ----------

        path : os.PathLike[str]
            Directory path the original binary files

        Returns
        -------

        np.ndarray 
            Reconstructed dynamical matrix
        """
        matrix_list = glob.glob('{:s}*'.format(path))

        dic_matrix : Dict[str,np.ndarray] = {}
        for mat in matrix_list :    
            index = self.read_imax(mat)
            dic_matrix[mat[-1]] = self.read_bin_multi_proc(mat,index)

        number_of_mode = int(np.amax(dic_matrix['u'])) 
        dynamical_matrix = np.zeros( (number_of_mode,number_of_mode), dtype=np.float64)
        for id in range(len(dic_matrix['u'])) : 
            i = dic_matrix['u'][id] - 1
            j = dic_matrix['v'][id] - 1 
            dij = dic_matrix['m'][id]
            dynamical_matrix[i,j] = dij

        return dynamical_matrix
    
    def read_bin_multi_proc(self, file : os.PathLike[str], idx : int) -> np.ndarray : 
        """Binary reader for matrix files from phondy
        
        Parameters
        ----------

        file : os.PathLike[str]
            Path the binary file to read

        idx : int 
            Integer header to read the binary file

        Returns 
        -------

        np.ndarray 
            Array containing data from tthe binary file

        """
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

    def read_imax(self, file_m : os.PathLike[str]) -> int : 
        """Read the integer header needed to extract data from binary
        
        Parameters
        ----------

        file_m : os.PathLike[str]
            Path to the binary file 

        Returns
        -------

        int 
            Interger header
        """
        with open(file_m, 'rb') as file:
            np.fromfile(file, dtype=np.int32, count=1)
            return np.fromfile(file, dtype=np.int32, count=1)[0]

    def read_phondy_matrix_multi_proc(self, path : str, njob : int = 1) -> np.ndarray :
        """Parallel reading for phondy matrices 
        
        Parameters 
        ----------

        path : os.PathLike[str]
            Directory path the original binary files

        njob : int 
            Number of processors for parallel reading
            
        Returns
        -------

        np.ndarray 
            Reconstructed dynamical matrix
        """  
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

        #print(np.linalg.norm(dynamical_matrix.T - dynamical_matrix))

        return dynamical_matrix
    
    def read_phondy_matrix_multi_proc_draft(self, path : str, njob : int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray] :  
        """Draft method only used for debug ...."""
        matrix_list_m = glob.glob('{:s}/*.m'.format(path))
        index_list = Parallel(n_jobs=njob)(delayed(self.read_imax)(mat) for mat in matrix_list_m)
        matrix_m = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)(mat,idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_u = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}u'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))
        matrix_v = Parallel(n_jobs=njob)(delayed(self.read_bin_multi_proc)('{:s}v'.format(mat[:-1]),idx) for idx,mat in list(zip(index_list,matrix_list_m)))

        matrix_u = list(itertools.chain.from_iterable(matrix_u))
        matrix_v = list(itertools.chain.from_iterable(matrix_v))
        matrix_m = list(itertools.chain.from_iterable(matrix_m))

        return  np.array(matrix_u), np.array(matrix_v), np.array(matrix_m), np.sum(index_list, dtype=np.int32)