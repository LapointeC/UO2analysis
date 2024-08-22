from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic, Diamond, SimpleCubic
from ase.lattice.tetragonal import SimpleTetragonal
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import Hexagonal, HexagonalClosedPacked
from ase.lattice.cubic import FaceCenteredCubicFactory, SimpleCubicFactory

import warnings
from ase import Atoms
from typing import List

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
