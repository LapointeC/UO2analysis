from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic, Diamond, SimpleCubic
from ase.lattice.tetragonal import SimpleTetragonal
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked
from ase.lattice.cubic import FaceCenteredCubicFactory, SimpleCubicFactory
from ..tools.neighbour import get_N_neighbour_Cosmin

import warnings
import numpy as np
from ase import Atoms
from typing import List, Tuple, Dict

class A15Factory(SimpleCubicFactory):
        """A factory for A15 lattices."""
        xtal_name = 'a15'
        bravais_basis = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.0, 0.5],[0.75,0.0,0.5],[0.5,0.25,0.0],[0.5,0.75,0.0],[0.0,0.5,0.25],[0.0,0.5,0.75]]
A15 = A15Factory()


class C15Factory(FaceCenteredCubicFactory):
        """A factory for C15 lattices."""
        xtal_name = 'c15'
        bravais_basis = [[1./8., 1./8., 1./8.], [7./8., 7./8., 7./8.], [1./2., 1./2., 1./2.],[1./4.,1./4.,1./2.],[1./4.,1./2.,1./4.],[1./2.,1./4.,1./4.]]
C15 = C15Factory()

class SolidAse(object):
    """Builder written by MCM look at unseen.git"""
    def __init__(self, size : List[int], symbol : str, lattice_cube : float | List[float]) -> None:
        if size is None:
            self.size = 4
        else:
            self.size=size

        if symbol is None:
            self.symbol='Fe'
        else:
            self.symbol=symbol

        if lattice_cube is None:
            self.lattice_cube = 2.8553
        else:
            self.lattice_cube = lattice_cube


    def structure(self, argument : str) -> Atoms:
        """ Dispatch the method name """
        method_name = str(argument)
        method = getattr(self, method_name, lambda: "not_yet")
        def isalambda(v):
            LAMBDA = lambda:0
            return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
        if (isalambda(method)):
            raise NotImplementedError("This structure {:s} is not implemented ....:".format(argument))
        else:
            return method()


    def BCC(self) -> Atoms:
        """Build BCC structure"""
        atoms = BodyCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)
        return atoms

    def FCC(self) -> Atoms:
        """Build FCC structure"""
        atoms = FaceCenteredCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)
        return atoms

    def HCP(self) -> Atoms: 
        """Build HCP structure"""
        # test lattice constant
        if isinstance(self.lattice_cube,float) : 
            warnings.warn('Missing c parameter for hcp lattice !')
            print('Ratio c/a will be set to {:1.3f} by default ...'.format(1.63))
            self.lattice_cube = [self.lattice_cube, 1.63*self.lattice_cube]

        atoms = HexagonalClosedPacked(directions=[[1,0,-1,0] ,[1,1,-2,0], [0,0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)
        return atoms

    def Diamond(self) -> Atoms:
        """Build perfect diamond structure"""
        atoms = Diamond(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)

        return atoms


    def SimpleCubic(self) -> Atoms:
        """Build Simple cubic structure"""
        atoms = SimpleCubic(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)
        return atoms

    def A15(self) -> Atoms:
        """Build A15 structure"""
        atoms = A15(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)

        return atoms

    def C15(self) -> Atoms:
        """Build C15 structure"""
        atoms = C15(directions=[[1,0,0], [0,1,0], [0,0,1]], size=tuple(self.size),
                                  symbol=self.symbol, pbc=(1,1,1),
                                  latticeconstant=self.lattice_cube)

        return atoms

class NeighboursCoordinates : 
    def __init__(self, structure : str = 'BCC',
                       rcut : float = 4.0,
                       size : Tuple[int] = (5,5,5)) -> None :
        """Little class to generate neighbour coordinate (in unit vector) for usual cirstallographic structure
        This class can be used to parametrise the Nye tensor calculations for dislocation analysis

        Parameters
        ----------

        structure : str 
            Cristallographic structure to consider

        rcut : float 
            Cutoff radius for neighbour calculations

        size : Tuple[int] 
            Size of the system to consider for neighbours construction
        """
        self.structure = structure
        self.rcut = rcut

        solide_ase = SolidAse(size, 'Kr',1.0)
        self.lattice_structure = solide_ase.structure(self.structure)

    def BuildNeighboursCoordinates(self, Nneigh : int = 100,
                                   shell : int = 3,
                                   threshold : float = 0.1) ->  Tuple[Dict[str, np.ndarray], np.ndarray] :
        """Build the array of neighbour coordinates
        
        Parameters
        ----------

        Nneigh : int 
            Total numbers of neighbours for shell constructions

        shell : int 
            Number of shell build

        threshold : float 
            Threshold in |d_ij|^2 for increment a new shell


        Returns 
        -------

        Dict[str, np.ndarray]
            Dictionnary of shells {shell i: corresponding neighour array (Nshell,3) in shell i}

        np.ndarray 
            Concatenate neigbour coordinates array (sum_{i=1}^shell Nshell_i,3)
        """

        array_neigh, _ = get_N_neighbour_Cosmin(self.lattice_structure,
                                                self.lattice_structure,
                                                self.lattice_structure.cell[:],
                                                self.rcut,
                                                Nneigh)
        dic_shell = {}
        full_neigh = None 

        local_neigh = array_neigh[0,:]
        local_shell_compt = 1
        for neigh in local_neigh : 
            if len(dic_shell) == 0 : 
                dic_shell[f'shell{local_shell_compt}'] = neigh.reshape(1,-1)
                full_neigh = neigh.reshape(1,-1)
            
            else : 
                neigh_shell = dic_shell[f'shell{local_shell_compt}'] 
                neigh = neigh.reshape(1,-1)
                array_distance_shell = np.sum(neigh_shell**2, axis=1)
                distance = np.sum(neigh**2, axis=1)
                
                # append case !
                if (np.abs(array_distance_shell - distance) < threshold).all() : 
                    dic_shell[f'shell{local_shell_compt}'] = np.concatenate( (dic_shell[f'shell{local_shell_compt}'], neigh), axis=0 )

                # add new shell
                else : 
                    local_shell_compt += 1
                    if local_shell_compt > shell :
                        break
                    else : 
                        dic_shell[f'shell{local_shell_compt}'] = neigh

                full_neigh = np.concatenate((full_neigh, neigh), axis=0)

        return dic_shell, full_neigh