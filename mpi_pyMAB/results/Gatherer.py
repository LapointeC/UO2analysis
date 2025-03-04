import numpy as np
from mpi4py import MPI
from typing import List
from ..parsers.BABFParser import BABFParser
from .BABFGatherer import BABFGatherer

class Gatherer(BABFGatherer):
    """Gatherer of PAFI simulation data

        Parameters
        ----------
        
        params : PAFIParser
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
    def __init__(self, params: BABFParser, nWorkers: int, rank: int, 
                 ensemble_comm: MPI.Intracomm, roots: List[int]) -> None:
        super().__init__(params, nWorkers, rank, ensemble_comm, roots)