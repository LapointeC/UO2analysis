from mpi4py import MPI
from .BaseParser import BaseParser
from .LAMMPSWorker import LAMMPSWorker

class BaseManager:
    """Base class for MC manager

        Parameters
        ----------
        world : MPI.Intracomm
            MPI communicator
        parser : BaseParser object 
            A BaseParser or inherited class instance
        Worker : Worker class
            a predefined or custom Worker classes, default BaseWorker
    """
    def __init__(self,world:MPI.Intracomm,parameters:BaseParser,
                 Worker=LAMMPSWorker)->None:
        self.world = world
        self.rank = world.Get_rank()
        self.nProcs = world.Get_size()
        # Read in configuration file
        self.parameters = parameters
        self.CoresPerWorker = int(self.parameters.parameters["CoresPerWorker"])
        if self.nProcs%self.CoresPerWorker!=0:
            if self.rank==0:
                print(f"""
                    CoresPerWorker={self.CoresPerWorker} must factorize nProcs={self.nProcs}!!
                """)
                exit(-1)
        # Establish Workers
        # worker_comm : Worker communicator for e.g. LAMMPS
        self.nWorkers = self.nProcs // self.CoresPerWorker
        
        # Create worker communicator 
        self.worker_rank = self.rank // self.CoresPerWorker
        self.worker_comm = world.Split(self.worker_rank,0)
        
        # ensemble_comm: Global communicator for averaging
        self.roots = [i*self.CoresPerWorker for i in range(self.nWorkers)]
        self.ensemble_comm = world.Create(world.group.Incl(self.roots))
        

        # set up and seed each worker
        self.parameters.seed(self.worker_rank)
        self.Worker = Worker(self.worker_comm,
                             self.parameters,
                             self.worker_rank,
                             self.rank,
                             self.roots)
        
        
    def close(self)->None:
        """Close Manager
            closes Worker
        """
        self.Worker.close()
