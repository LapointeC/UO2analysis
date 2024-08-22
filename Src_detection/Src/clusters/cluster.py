import numpy as np
from ase import Atom, Atoms
from typing import Dict, List, Any, Tuple

#######################################################
## Defect class
#######################################################
class Cluster : 
    """Cluster class which contains all data about atomic defects found in a given configuration.
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
        """Distance covariance estimatior for anisotropic clusters
        
        Parameters:
        -----------

        regularisation : float 
            regularisation parameter for covariance matrix pseudo inversion
        """
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
        """Get an estimation of number interstitial atom inside the cluster
        
        Returns:
        --------

        float : number of atom estimation
        """
        return len(self.atoms_dfct) - self.get_volume()/mean_atomic_volume


class LocalLine : 
    """LocalLine object which contains all local properties of the dislocation like 
    ````local_normal``` and ```local_burger``` ..."""
    def __init__(self, positions : np.ndarray, species : str) -> None : 
        """Init method for LocalLine
        
        Parameters:
        -----------

        positions : np.ndarray 
            Inital position for the ```LocalLine```

        species : str 
            Species associated to the ```LocalLine```
        """
        self.center = positions
        self.species = species
        self.local_burger = None
        self.local_normal = None
        self.next = None
        self.norm_normal = None

    def update_center(self, center : np.ndarray) -> None : 
        """Update the center of ```LocalLine``` 
        
        Parameters:
        -----------

        center : np.ndarray 
            new center of the ```LocalLine```
        """
        self.center = center
        return 

    def update_burger(self, burger : np.ndarray) -> None : 
        """Update the burger vector of ```LocalLine``` 
        
        Parameters:
        -----------

        burger : np.ndarray 
            local burger vector of the ```LocalLine```
        """
        self.local_burger = burger
        return

    def update_normal(self, normal : np.ndarray) -> None : 
        """Update the normal of ```LocalLine``` 
        
        Parameters:
        -----------

        normal : np.ndarray 
            local normal vector of the ```LocalLine```
        """
        self.local_normal = normal
        return 
    
    def update_norm_normal(self, norm : float) -> None : 
        """Update the norm of normal vector of ```LocalLine``` 
        
        Parameters:
        -----------

        norm : float
            norm of normal vector of ```LocalLine```
        """
        self.norm_normal = norm
        return 

    def update_next(self, next_id : int) -> None : 
        """Update the next point the ```LocalLine``` 
        
        Parameters:
        -----------

        next_id : int
            next point id to build the whole dislocation line
        """
        self.next = next_id 
        return 

    def get_local_caracter(self) -> float : 
        """Compute the local dislocation caracter
        
        Returns:
        --------

        float 
            Dislocation caracter ( b \cdot n / \Vert b \Vert )
        """
        return np.dot(self.local_burger,self.local_normal)/np.linalg.norm(self.local_burger)

class ClusterDislo(Cluster) : 
    """```ClusterDislo``` class which contains all data about dislocation found in a given configuration. This class 
    inherit from ```Cluster``` class. 

    This class contains the present list of methods : 
        - append : update defect cluster with new atom object 
        - update_extension : update the spatial extension of the cluster
        !TO DO : new method to have size of line ...!
        - get_lenght_line : return the lenght of the dislocation line 
        - average_burger_vector_line : return the average burger vector along the dislocation line
    """
    def __init__(self, local_line_init : LocalLine, id_line_init : int, rcut_cluster : float = 4.5) -> None :
        """Init method for ```ClusterDislo``` 
        
        Parameters:
        -----------

        local_line_init : LocalLine 
            Associated first LocalLine object of the cluster 

        id_line_init : int 
            Id of the atom associated to the first LocalLine object

        rcut_cluster : float 
            Initial size of the cluster (in AA)
        """
        
        self.local_lines = {id_line_init:local_line_init}
        self.center = local_line_init.center
        self.positions = local_line_init.center
        
        self.order_line = []
        self.starting_point = None
        self.smooth_local_lines : Dict[int,LocalLine] = {}
        self.smooth_order_line = [] 
        super().__init__(Atom(local_line_init.species,local_line_init.center), rcut_cluster)

    def append_dislo(self, local_line : LocalLine, id_line : int) -> None : 
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
    
    def get_lenght_line(self) -> Tuple[float, np.ndarray] :
        """Compute the total lenght of the dislocation associated to ```ClusterDislo``` 
        Also extract the whole dislocation line vectors 


        Returns:
        --------

        float 
            Total lenght of the dislocation line (in AA)
        
        np.ndarray 
            Array of non normalised local normal vector 
        """
        return np.sum([ line.norm_normal for _, line in self.local_lines.items() ]), np.array([ line.local_normal*line.norm_normal for _, line in self.local_lines.items() ])

    def get_burger_vector_line(self) -> Tuple[float,np.ndarray] : 
        """Compute the average norm of burger vector of the dislocation associated to ```ClusterDislo``` 
        Also extract the whole burger vector along dislocation line

        Returns:
        --------

        float 
            Average burger vector along dislocation line (in AA)
        
        np.ndarray 
            Array of burger vector along dislocation line  
        """
        return np.mean([ np.linalg.norm(line.local_burger) for _, line in self.local_lines.items() ]), np.array([ np.linalg.norm(line.local_burger) for _, line in self.local_lines.items() ])

    def get_caracter_line(self) -> Tuple[float,np.ndarray] : 
        """Compute the average caracter of the dislocation associated to ```ClusterDislo``` 
        Also extract the whole local caracters along dislocation line

        Returns:
        --------

        float 
            Average caracter along dislocation line
        
        np.ndarray 
            Array of caracter along dislocation line
        """
        return np.mean([ line.get_local_caracter() for _, line in self.local_lines.items() ]), np.array([ line.get_local_caracter() for _, line in self.local_lines.items() ])