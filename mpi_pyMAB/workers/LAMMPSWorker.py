from __future__ import annotations
import numpy as np
import os
from typing import Any, List
from ..parsers.BaseParser import BaseParser
from ..results.ResultsBABF import ResultsBABF
from mpi4py import MPI
from lammps import lammps,LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,LMP_TYPE_SCALAR
from .BaseWorker import BaseWorker

class LAMMPSWorker(BaseWorker):
    """LAMMPS worker for PAFI, inheriting BaseWorker

        Initialization routine:
        - runs "Input" script, loading first configuration
        - extracts cell data
        - defines various functions wrapping computes etc.
        - defines initialize_hyperplane function to set plane

        Parameters
        ----------
        
        comm : MPI.Intracomm
            MPI communicator
        
        parameters : BABFParser
            Predefined or custom BABFParser object
        
        worker_instance : int
            unique worker rank
        """
    def __init__(self, comm: MPI.Intracomm, 
                 parameters: BaseParser, worker_instance: int,
                 rank: int, roots: List[int]) -> None:
        super().__init__(comm, parameters, worker_instance, rank, roots)
        
        self.name = "LAMMPSWorker"
        self.last_error_message = ""
        self.start_lammps()
        if self.has_errors:
            print("ERROR STARTING LAMMPS!",self.last_error_message)
            return
        start_config = self.parameters.Configuration
        # TODO abstract
        self.run_script("Input")
        if self.has_errors:
            print("ERROR RUNNING INPUT SCRIPT!",self.last_error_message)
            return
        
        self.has_cell_data = False
        self.get_cell_data()

        
    def start_lammps(self)->None:
        """Initialize LAMMPS instance

            Optionally 

        """
        if self.parameters("LogLammps"):
            logfile = 'log.lammps.%d' % self.worker_instance 
        else:
            logfile = 'none'
        try:
            cmdargs = ['-screen','none','-log',logfile]
            self.L = lammps(comm=self.comm,cmdargs=cmdargs)
        except Exception as ae:
            print("Couldn't load LAMMPS!",ae)
            self.has_errors = True
            
    def run_script(self,key:str,arguments:None|dict|ResultsBABF=None)->None:
        """Run a script defined in the XML
            Important to replace any %wildcards% if they are there!
            
        Parameters
        ----------
        
        key : str
            script key from XML
        
        arguments : None | dict | ResultsBABF, optional
            will be used to replace wildcards, by default None
        """
        if key in self.parameters.scripts:
            if self.parameters("Verbose")>0 and self.rank==0:
                print(f"RUNNING SCRIPT {key}")
        
            script = self.parameters.parse_script(key,arguments=arguments)
            self.run_commands(script)

    def run_commands(self,cmds : str | List[str]) -> bool:
        """
            Run LAMMPS commands line by line, checking for errors
        """
        cmd_list = cmds.splitlines() if isinstance(cmds,str) else cmds
        for cmd in cmd_list:
            try:
                if self.parameters("Verbose")>0 and self.rank==0:
                    print(f"TRYING COMMAND {cmd}")
                self.L.command(cmd)
            except Exception as ae:
                if self.local_rank==0:
                    message = f"LAMMPS ERROR: {cmd} {ae}"
                else:
                    message = None
                self.last_error_message = ae
                raise SyntaxError(message)
    
    def gather(self,name:str,type:None|int=None,count:None|int=None)->np.ndarray:
        """Wrapper of LAMMPS gather()
           

        Parameters
        ----------
        
        name : str
            name of data
        
        type : None | int, optional
            type of array, 0:integer or 1:double, by default None.
            If None, an attempt will be made to determine autonomously.
        
        count : None | int, optional
            number of data per atom, by default None. 
            If None, an attempt will be made to determine autonomously.

        Returns
        -------
        
        np.ndarray
            the LAMMPS data

        Raises
        ------
        
        ValueError
            if name not found
        """
        if name in ['x','f','v'] or 'f_' in name:
            if type is None:
                type = 1
            if count is None:
                count = 3
        elif name in ['id','type','image']:
            if type is None:
                type = 0
            if count is None:
                count = 1
        if type is None or count is None:
            raise ValueError("Error in gather: type or count is None")
        
        try:
            res = self.L.gather(name,type,count)
        except Exception as ae:
            if self.local_rank==0:
                print("Error in gather:",ae)
            self.last_error_message = ae
        return np.ctypeslib.as_array(res).reshape((-1,count))

    def scatter(self,name:str,data:np.ndarray)->None:
        """Scatter data to LAMMPS
            Assume ordered with ID

        Parameters
        ----------
        
        name : str
            name of array
        
        data : np.ndarray
            numpy array of data. Will be flattened.
        """
        if np.issubdtype(data.dtype,int):
            type = 0
        elif np.issubdtype(data.dtype,float):
            type = 1
        count = data.shape[1] if len(data.shape)>1 else 1
        try:
            self.L.scatter(name,type,count,
                           np.ctypeslib.as_ctypes(data.flatten()))
        except Exception as ae:
            if self.local_rank==0:
                print("Error in scatter:",ae)
            self.last_error_message = ae
    
    def get_natoms(self)->int:
        """Get the atom count

        Returns
        -------
        
        int
            the atom count
        """
        return self.L.get_natoms()
    
    def get_cell_data(self)->None:
        """
            Extract supercell information
        """
        boxlo,boxhi,xy,yz,xz,pbc,box_change = self.L.extract_box()
        self.Periodicity = np.array([bool(pbc[i]) for i in range(3)],bool)
        self.Cell = np.zeros((3,3))
        for cell_j in range(3):
            self.Cell[cell_j][cell_j] = boxhi[cell_j]-boxlo[cell_j]
        self.Cell[0][1] = xy
        self.Cell[0][2] = xz
        self.Cell[1][2] = yz
        self.invCell = np.linalg.inv(self.Cell)
        self.has_cell_data = True
        return None

    def load_config(self,file_path:os.PathLike[str])->np.ndarray:
        """Load a LAMMPS data file with read_data() and return a numpy array of positions

        Parameters
        ----------
        
        file_path : os.PathLike[str]
            Path to the LAMMPS .dat file.

        Returns
        -------
        
        np.ndarray, shape (N,3)
            the positions
        """
        self.run_commands(f"""
            delete_atoms group all
            read_data {file_path} add merge
        """)

        self.x = self.gather("x",type=1,count=3)
        self.x_reference = self.x.copy()
        self.v = np.zeros(self.x.shape)
        self.force = np.zeros(self.x.shape)

        #return self.gather("x",type=1,count=3)

    def extract_compute(self,id:str,vector:bool=True)->float|np.ndarray:
        """    Extract compute from LAMMPS
            
            id : str
        Parameters
        ----------
        
        id : str
            compute id
        
        vector : bool, optional
            is the return value a vector, by default True

        Returns
        -------
        
        np.ndarray or float
           return data
        """
        style = LMP_STYLE_GLOBAL
        type = LMP_TYPE_VECTOR if vector else LMP_TYPE_SCALAR
        assert hasattr(self.L,"numpy")
        try:
            res = self.L.numpy.extract_compute(id,style,type) 
            return np.array(res)
        except Exception as e:
            if self.local_rank==0:
                print("FAIL EXTRACT COMPUTE",e)
            self.close()
            return None
        
    def get_energy(self)->float:
        """Extract the potential energy
        
        Returns
        -------
        
        float
            The potential energy (pe)
        """
        return self.extract_compute("thermo_pe",vector=False)


    def evaluate_lammps_forces(self) -> np.ndarray : 
        """Evaluate the lammps forces
        
        Returns
        -------
        
        np.ndarray
            Lammps forces ! 
        """
        self.run_commands('run 0')
        return self.gather('f',1,3)


    def close(self) -> None:
        """
            Close down. TODO Memory management??
        """
        super().close()
        self.L.close()
