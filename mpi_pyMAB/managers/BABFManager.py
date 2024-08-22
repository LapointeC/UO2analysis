from __future__ import annotations
import itertools
from typing import List,Dict
import numpy as np
import os
from mpi4py import MPI
from ..results.ResultsBABF import ResultsBABF
from .BaseManager import BaseManager
from ..parsers.BABFParser import BABFParser
from ..workers.BABFWorker import BABFWorker
from ..results.Gatherer import Gatherer

class BABFManager(BaseManager):
    def __init__(self, world: MPI.Intracomm, 
                 xml_path:None|os.PathLike[str]=None,
                 parameters:None|BABFParser=None,
                 Worker:BABFWorker=BABFWorker,
                 Gatherer:Gatherer=Gatherer) -> None:
        """Default manager of MAB, child of BaseManager

        Parameters
        ----------
        world : MPI.Intracomm
            MPI communicator
        xml_path : None or os.PathLike[str], optional
            path to XML configuration file, default None
        parameters : None or BABFParser object, optional
            preloaded BABFParser object, default None
        restart_data : None or os.PathLike[str], optional
            path to CSV data file. Will read and skip already sampled parameters
        Worker : BABFWorker, optional,
            Can be overwritten by child class, by default BABFWorker
        Gatherer : Gatherer, optional
            Can be overwritten by child class, by default Gatherer
        """
        
        
        assert (not parameters is None) or (not xml_path is None)
        
        if parameters is None:
            # TODO have standalone check for suffix in config_[suffix].xml?
            # not the best solution currently if we use BaseManager alone....
            parameters = BABFParser(xml_path=xml_path,rank=world.Get_rank())
        
        super().__init__(world, parameters, Worker, Gatherer)
        
    
    
    def run(self,print_fields:List[str]|None=None,
            width:int=10,
            precision:int=5)->None:
        """Basic parallel MAB sampling
        Parameters
        ----------
        print_fields : List[str] or None
            Fields to print to screen, default None. 
            character count of field printout, default 10
        precision : int
            precision of field printout, default 4
        """
        assert self.parameters.ready()

        if print_fields is None:
            # to do !
            print_fields = \
                ["Temperature","","<d_lambda F(1)>","A(1)","<A(1)^2> - <A(1)>^2"]
        
        for f in print_fields:
            width = max(width,len(f))
        
        def line(data:List[float|int|str]|Dict[str,float|int|str])->str:
            """Format list of results to print to screen
            """
            if len(data) == 0:
                return ""
            format_string = ("{: >%d} "%width)*len(data)
            if isinstance(data,dict):
                _fields = []
                for f in print_fields:
                    val = data[f]
                    _fields += [val]
            else: 
                _fields = data
            
            fields = []
            for f in _fields:
                isstr = isinstance(f,str)
                fields += [f if isstr else np.round(f,precision)]
            return format_string.format(*fields)

        
        if self.rank==0:
            screen_out = f"""
            Initialized {self.nWorkers} workers with {self.CoresPerWorker} cores
            <> == time averages,  av/err over ensemble
            """
            print(screen_out)
            print(line(print_fields))
            
            # return value
            average_results = {k:[] for k in print_fields}
            parameters_dict = self.parameters.to_dict()
            self.Gatherer.init_h5file( f"{self.parameters['WorkingDirectory']}/mab.h5",parameters_dict)

        results = ResultsBABF(self.parameters['LambdaGrid'],
                              block=self.parameters['Block'])

        results = self.Worker.thermalisation_steps(results, 
                                                   block = self.parameters['Block'])
        for it_lang in self.parameters["StochasticSteps"] : 
            results = self.Worker.update_step_BABF(results,
                                                   block=self.parameters['Block'])  

            if it_lang%self.parameters["WritingStep"] : 
                screen_out = self.Gatherer.get_dict(print_fields)
                if self.rank == 0:
                    for k in screen_out.keys():
                        average_results[k].append(screen_out[k])
                    
                    print(line(screen_out))

                self.Gatherer.write_hdf5(f"{self.parameters['WorkingDirectory']}/mab.csv",
                                           it_lang,
                                           block=self.parameters['Block'])
                self.world.Barrier()

            if it_lang%self.parameters["GatherStep"] : 
                key_to_update = ['sum_w_AxU_dl', 'sum_w_AxU2_dl', 'sum_w_A']
                results = self.Gatherer.Parallel_biais_full_update(results,
                                                                   key_to_update,
                                                                   block=self.parameters['Block'])
                self.world.Barrier()

        if self.rank == 0 :
            self.Gatherer.close_h5file()

        return