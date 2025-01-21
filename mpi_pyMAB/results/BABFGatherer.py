from __future__ import annotations
from mpi4py import MPI
import os
import numpy as np
from typing import List
from ..parsers.BABFParser import BABFParser
from .ResultsBABF import ResultsBABF
import h5py 

from h5py import Group, File

class BABFGatherer:
    def __init__(self,params:BABFParser,
                 nWorkers:int,
                 rank:int,
                 comm_world:MPI.Intracomm,
                 ensemble_comm:MPI.Intracomm,
                 roots:List[int])->None:
        """Basic gatherer of PAFI simulation data

        Parameters
        ----------
        
        params : BABFParser
            Custom or predefined PAFIParser object
        
        nWorkers : int
            total number of PAFI workers
        
        rank : int
            global MPI rank
        
        ensemble_comm : MPI.Intracomm
            MPI communicator to gather ensemble data
        
        roots : List[int]
            list of root ranks for each worker. Only collate data here
            This should be depreceated asap
        """
        self.params = params
        self.nWorkers = nWorkers
        self.rank = rank
        self.comm_world = comm_world
        self.comm = ensemble_comm
        self.roots = roots
        self.epoch_data = None # for each cycle
        self.epoch_data_b = None
        self.all_data = None # for total simulation
        self.last_data = None # for print out
        self.last_data_b = None
        self.h5file : File = None

    def Parallel_biais_update(self,results : ResultsBABF, key_data : str, block : bool = False) -> ResultsBABF: 
        """Perform MPI reduce for interest quantities of ResultsBABF object (sum_w_AxU_dl, sum_w_A)...
        and broadcast on all processors

        Parameters
        ----------
        
        results : ResultsBABF
            results object to parallely update
        
        key_data : str 
            quantity to reduce and then broadcast : sum_w_AxU_dl, sum_w_A...
        
        block : bool 
            Key word for constrained BABF method

        Returns 
        -------
        
        ResultsBABF
            Updated resutls object with data of all simulations

        """
        
        sum_data = self.comm.reduce(results.data_babf[key_data], op=MPI.SUM, root=0)
        if self.rank == 0 : 
            data2broadcast = sum_data
        else : 
            data2broadcast = None
        data2broadcast = self.comm_world.bcast(data2broadcast,root=0)
        results.data_babf[key_data] = data2broadcast

        if block : 
            sum_data = self.comm.reduce(results.data_babf_block[key_data], op=MPI.SUM, root=0)
            if self.rank == 0 : 
                data2broadcast = sum_data
            else : 
                data2broadcast = None
            data2broadcast = self.comm_world.bcast(data2broadcast,root=0)
            results.data_babf_block[key_data] = data2broadcast
            return results               

    def Parallel_biais_full_update(self,results : ResultsBABF, key_data_list : List[str], block : bool = False) -> ResultsBABF: 
        """Perform MPI reduce for interest quantities of ResultsBABF object (sum_w_AxU_dl, sum_w_A)...
        and broadcast on all processors

        Parameters
        ----------
        
        results : ResultsBABF
            results object to parallely update
        
        key_data : List[str] 
            list of quantities to reduce and then broadcast : sum_w_AxU_dl, sum_w_A...
        
        block : bool 
            Key word for constrained BABF method

        Returns 
        -------
        
        ResultsBABF
            Updated results object with data of all simulations

        """
        for key_word in key_data_list : 
            if block : 
                results = self.Parallel_biais_update(results,key_word,block=block)
            results = self.Parallel_biais_update(results,key_word,block=False)

        return results

                
    def gather(self,data:dict|ResultsBABF,block : bool = False)->None:
        """Gather results from a simulation epoch,
        local to each worker. Here, very simple,
        is overwritten by each call of gather()

        Parameters
        ----------
        
        data : dict or ResutlsBABF
            Simulation data, extracted as dictionary from ResultsHolder
        """
        if self.rank in self.roots:
            if isinstance(data,ResultsBABF):
                self.last_data = data.extract_estimator(block=block)
                if block : 
                    self.epoch_data = data.data_babf.copy()
                    self.epoch_data_b = data.data_babf_block.copy()                
                else : 
                    self.epoch_data = data.data_babf.copy()
            else:
                self.epoch_data = data.copy()
       
    
    def get_line(self,fields:List[str])->List[str]|None:
        """Return output data to print out on root node

        Parameters
        ----------
        
        fields : List[str]
            fields to extract
        
        Returns
        -------
        
        List[str]|None
            if root process, return list of lines to print, else return `None`
        """
        if self.rank != 0:
            return None
        
        line = []
        for f in fields:
            if (not self.last_data is None) and (f in self.last_data.keys()):
                d = self.epoch_data[f]
                line += [ d ]
            else:
                line += ["n/a"]
            
        return line
    
    def get_dict(self,fields:List[str])->dict|None:
        """Return output data to print out on root node

        Parameters
        ----------
        
        fields : List[str]
            fields to extract
        
        Returns
        -------
        
        dict|None
            if root process, return dict of fields to print, else return `None`
        """
        line = self.get_line(fields)
        if not line is None:
            return {kv[0]:kv[1] for kv in zip(fields,line)}

    def concatenate_babf_dictionnary(self) -> dict : 
        """Build dictionnary for block BABF"""
        concatenated_dict = self.epoch_data.copy()
        for key in self.epoch_data_b.keys() : 
            concatenated_dict['%s_block'%(key)] = self.epoch_data_b[key]
        return concatenated_dict

    def init_h5file(self, path : os.PathLike[str], parameters : dict) -> None :
        """Initialise hdf5 file to store data 
        
        Parameters:
        -----------

        path : os.PathLike[str] 
            Path to hdf5 file 

        parameters : dict 
            Dictionnarty containing all parameters for the simulation
        
        """
        self.h5file = h5py.File(path, 'w')
        self.h5file.create_group('MAB') 
        hdf5_group = self.h5file['MAB']
        init_group = hdf5_group.create_group('Parameters')
        for key, values in parameters.items() : 
            init_group.attrs[key] = values
        return 

    def close_h5file(self) -> None : 
        """Close the hdf5 file"""
        self.h5file.close()
        return

    def update_h5file(self, hdf5_group : Group, iteration : int, dictonnary_data : dict = None):
        """Update dat in hdf5 file 
        
        Parameters:
        -----------

        hdf5_group : Group 
            Hdf5 group for MAB

        iteration : int 
            Index of iteration 

        dictionnary_data : 
            dictionnary containing data to store

        """
        iteration_group = f'Step{iteration}'
        if iteration_group not in hdf5_group:
            # Create group for the potential if it doesn't exist
            it_group = hdf5_group.create_group(iteration_group) 
            if dictonnary_data is not None :
                for key, value in dictonnary_data.items():
                    it_group.create_dataset(key, data=value, compression="gzip", compression_opts=9)

    def write_hdf5(self, step : int = 0, block : bool = False)->None:
        """Write data as pandas dataframe

        Parameters
        ----------
        
        step : int 
            step number of BABF procedure
        
        block : bool 
            key word for constrained BABF method

        """

        if self.rank==0:
            if self.epoch_data is None:
                print("No data to write! Exiting!")
            else:
                if block :
                    concatenate_dict = self.concatenate_babf_dictionnary()
                    self.update_h5file(self.h5file['MAB'], step, concatenate_dict)
                else : 
                    self.update_h5file(self.h5file['MAB'], step, self.epoch_data)
        
        return
    
            
                
