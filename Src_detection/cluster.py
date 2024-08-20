import numpy as np
from ase import Atom, Atoms
from typing import Dict, List, Any

#######################################################
## Defect class
#######################################################
class Cluster : 
    """Cluster class which contains all data about atomic defects found in a given configuration
    This class contains the present list of methods : 
        - append : update defect cluster with new atom object 
        - update_extension : update the spatial extension of the cluster
        - get_volume : return the volume of the cluster
        - estimation_dfct_number : return an estimation of the number of defect inside the cluster (working for point defects...)

    """
    def __init__(self, atom : Atom, rcut : float, array_property : Dict[str,Any] = {}) -> None : 
        """Init method cluster class 
        
        Parameters:
        -----------

        atom : Atom 
            First atom object indentifies to be part of the cluster 

        rcut : float 
            Initial size of the cluster 
        
        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        """
        self.array_property = array_property
        self.atoms_dfct = Atoms()
        self.atoms_dfct.append(atom)
        self.rcut = rcut
        self.size = rcut
        self.elliptic = (1.0/(self.size**2))*np.eye(3)
        self.backup_elliptic = (1.0/(self.size**2))*np.eye(3)
        self.center = self.atoms_dfct.positions

    def append(self, atom : Atom, array_property : Dict[str,Any] = {}, elliptic : str ='iso') -> None : 
        """Append new atom in the cluster
        
        Parameters:
        -----------

        atom : Atom 
            New atom to put in the cluster 
        
        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        """       
        self.atoms_dfct.append(atom)
        self.center = self.atoms_dfct.get_center_of_mass()
        if elliptic == 'iso' :
            self.size = self.update_extension()
            self._isotropic_extension()
        if elliptic == 'aniso' : 
            self._anistropic_extension()
        
        for prop in array_property.keys() : 
            self.array_property[prop] += array_property[prop]

    def update_extension(self) -> float : 
        """Update the spatial extension of the cluster 
        Should be changed for non isotropic defects ..."""
        return max([self.rcut, np.amax( [np.linalg.norm(pos - self.center) for pos in self.atoms_dfct.positions ] )])

    def _isotropic_extension(self) -> None : 
        """Distance covariance estimatior for isotropic clusters"""
        self.elliptic = (1.0/(self.size**2))*np.eye(3)
    
    def _anistropic_extension(self, regularisation : float = 0.0) -> None : 
        """Distance covariance estimatior for anisotropic clusters"""
        covariance = (self.atoms_dfct.positions - self.center).T@(self.atoms_dfct.positions - self.center)/(self.atoms_dfct.positions.shape[0] - 1)
        self.elliptic = np.linalg.pinv(covariance + regularisation*np.eye(covariance.shape[0]))

        # check the minimal isotropic raduis for each direction
        for i in range(self.elliptic.shape[0]) : 
            if self.elliptic[i,i] > 1.0/(self.rcut**2) :
                self.elliptic[i,:] = 0.0
                self.elliptic[:,i] = 0.0
                self.elliptic[i,i] = 1.0/(self.rcut**2)

    def get_elliptic_distance(self, atom : Atom) -> float :
        """Compute distance to the elliptic covariance distances envelop
        
        Parameters:
        -----------

        atom : Atom
            Atom object to compute the elliptic distance

        Returns:
        --------

        float : Elliptic distance
        """
        return np.sqrt((atom.position.flatten()-self.center.flatten())@self.elliptic@(atom.position.flatten()- self.center.flatten()).T)

    def get_volume(self) -> float : 
        """Get an estimation of the cluster volume 
        
        Returns:
        --------

        float : cluster volume 
        """
        return np.sum(self.array_property['atomic-volume'])
    
    def estimate_dfct_number(self, mean_atomic_volume : float) -> float :
        """Get an estimation of number of atom inside the cluster
        
        Returns:
        --------

        float : number of atom estimation
        """
        return len(self.atoms_dfct) - self.get_volume()/mean_atomic_volume


class LocalLine : 
    def __init__(self, positions : np.ndarray, species : str) -> None : 
        self.center = positions
        self.species = species
        self.local_burger = None
        self.local_normal = None
        self.next = None
        self.norm_normal = None

    def update_center(self, center : np.ndarray) -> None : 
        self.center = center
        return 

    def update_burger(self, burger : np.ndarray) -> None : 
        self.local_burger = burger
        return

    def update_normal(self, normal : np.ndarray) -> None : 
        self.local_normal = normal
        return 
    
    def update_norm_normal(self, norm : float) -> None : 
        self.norm_normal = norm
        return 

    def update_next(self, next_id : int ) -> None : 
        self.next = next_id 
        return 

class ClusterDislo(Cluster) : 
    def __init__(self, local_line_init : LocalLine, id_line_init : int, rcut_cluster : float = 4.5) -> None :
        self.local_lines = {id_line_init:local_line_init}
        self.center = local_line_init.center
        self.positions = local_line_init.center
        
        self.order_line = []
        self.starting_point = None
        self.smooth_local_lines : Dict[int,LocalLine] = {}
        self.smooth_order_line = [] 
        super().__init__(Atom(local_line_init.species,local_line_init.center), rcut_cluster)

    def append(self, local_line : LocalLine, id_line : int) -> None : 
        """Append new atom in the cluster
        
        Parameters:
        -----------

        atom : Atom 
            New atom to put in the cluster 
        
        array_property : Dict[str,Any]
            Dictionnnary which contains additional data about atom in the cluster (atomic volume, mcd distance ...)

        """       
        self.local_lines[id_line] = local_line
        self.atoms_dfct.append( Atom(local_line.species, local_line.center) )
        self.center = self.atoms_dfct.get_center_of_mass().reshape((1,3))
        self._anistropic_extension()
        return 
    