from __future__ import annotations
import numpy as np
import os
from typing import List
from mpi4py import MPI
from .BaseParser import BaseParser

class BaseWorker:
    """Basic PAFI Worker

        Parameters
        ----------
        comm : MPI.Intracomm
            MPI communicator
        parameters : BaseParser
            Predefined or custom  BaseParser object
        worker_instance : int
            unique worker rank
        rank : int
            global rank (for MPI safety)
        roots : List[int]
            list of master ranks  (for MPI safety)
        """
    def __init__(self, comm : MPI.Intracomm,
                 parameters:BaseParser,
                 worker_instance:int,
                 rank:int,
                 roots:List[int]) -> None:
        self.worker_instance = worker_instance
        self.comm = comm
        self.local_rank = comm.Get_rank()
        self.roots = roots
        self.rank = rank
        self.parameters = parameters
        self.error_count = 0
        self.scale = np.ones(3)
        self.out_width=16
        self.natoms=0
        self.nlocal=0
        self.offset=0
        self.x : np.ndarray = None
        self.x_reference : np.ndarray = None
        self.v : np.ndarray = None
        self.force : np.ndarray = None
        self.name="BaseWorker"
        self.has_errors = False
        self.has_cell_data=False
        self.Cell = None
        self.Periodicity = None
        self.invCell = None
        self.parameters.seed(worker_instance)
        self.kB = 8.617e-5
        self.mass_array : np.ndarray = None
        self.mass_dictionary = {"H":1.008,"He":4.003,"Li":6.941,"Be":9.012,"B":10.811,
                                "C":12.011,"N ":14.007,"O":15.999,"F":18.998,"Ne":20.180,
                                "Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.086,"P":30.974,
                                "S":32.065,"Cl":35.453,"Ar":39.948,"K":39.098,"Ca":40.078,
                                "Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,
                                "Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.390,
                                "Ga":69.723,"Ge":72.640,"As":74.922,"Se":78.960,"Br":79.904,
                                "Kr":83.800,"Rb":85.468,"Sr":87.620,"Y":88.906,"Zr":91.224,
                                "Nb":92.906,"Mo":95.940,"Tc":98.000,"Ru":101.070,"Rh":102.906,
                                "Pd":106.420,"Ag":107.868,"Cd":112.411,"In":114.818,"Sn":118.710,
                                "Sb":121.760,"Te":127.600,"I":126.905,"Xe":131.293,"Cs":132.906,
                                "Ba":137.327,"La":138.906,"Ce":140.116,"Pr":140.908,"Nd":144.240,
                                "Pm":145.000,"Sm":150.360,"Eu":151.964,"Gd":157.250,"Tb":158.925,
                                "Dy":162.500,"Ho":164.930,"Er":167.259,"Tm":168.934,"Yb":173.040,
                                "Lu":174.967,"Hf":178.490,"Ta":180.948,"W":183.840,"Re":186.207,
                                "Os":190.230,"Ir":192.217,"Pt":195.078,"Au":196.967,"Hg":200.590,
                                "Tl":204.383,"Pb":207.200,"Bi":208.980,"Po":209.000,"At":210.000,
                                "Rn":222.000,"Fr":223.000,"Ra":226.000,"Ac":227.000,"Th":232.038,
                                "Pa":231.036,"U ":238.029,"Np":237.000,"Pu":244.000,"Am":243.000,
                                "Cm":247.000,"Bk":247.000,"Cf":251.000,"Es":252.000,"Fm":257.000,
                                "Md":258.000,"No":259.000,"Lr":262.000,"Rf":261.000,"Db":262.000,
                                "Sg":266.000,"Bh":264.000,"Hs":277.000,"Mt":268.000}

    def load_config(self,file_path:os.PathLike[str]) -> np.ndarray:
        """Placeholder function to load in file and return configuration
        Overwritten by LAMMPSWorker
        
        Parameters
        ----------
        file_path : os.PathLike[str]
            path to file

        Returns
        -------
        np.ndarray, shape (natoms,3)
            configuration vector
        """
        assert os.path.exists(file_path)
        return np.loadtxt(file_path)

    def pbc(self,X:np.ndarray,central:bool=True)->np.ndarray:
        """Minimum image convention, using cell data
            
        Parameters
        ----------
        X : np.ndarray, shape (natoms,3)
            configuration vector
        central : bool, optional
            map scaled coordinates to [-.5,.5] if True, else [0,1], by default True

        Returns
        -------
        np.ndarray
            wrapped vector
        """
        if not self.has_cell_data:
            return X
        else:
            sX = X.reshape((-1,3))@self.invCell
            sX -= np.floor(sX+0.5*int(central))@np.diag(self.Periodicity)
            return (sX@self.Cell).reshape((X.shape))

    def get_mass_center(self) -> np.ndarray :
        """Compute the mass center of positions x

        Returns 
        -------
        np.ndarray
            x center of mass
        """
        return np.mean(self.mass_array@self.x,axis=0)/np.trace(self.mass_array)

    def centering_system(self) -> None :
        """Recenter system / reference system on center of mass"""
        mass_center = self.get_mass_center()
        self.x += - mass_center
        self.x_reference += - mass_center
        return 

    def max_displacement(self) -> float : 
        """Evaluate the maximum atomic displacement wrt the reference lattice
        
        Returns
        -------
        float 
            maximum atomic displacement
        """
        return np.amax(np.linalg.norm(self.x - self.x_reference, axis=1))


    def close(self)->None:
        pass
            

