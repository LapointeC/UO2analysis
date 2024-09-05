import os
import numpy as np
from .lattice import SolidAse

from scipy.linalg import expm
from ase.io import write
from ase import Atoms

from typing import List, Tuple, TypedDict

class InputsDictDislocationBuilder(TypedDict) :
    structure : str
    a0 : float
    size_loop : float
    scale_loop : float
    orientation_loop : np.ndarray
    element : str

class DislocationsBuilder :
    def __init__(self, inputs_dict : InputsDictDislocationBuilder) -> None :
        self.param_dict = inputs_dict
        self.implemented_lattice = ['BCC', 'FCC']
        self.possible_plane = self.recursive_unit_vectors()
        self.normal_vector = self.param_dict['orientation_loop']/np.linalg.norm(self.param_dict['orientation_loop'])
        
        self.center : np.ndarray = None
        self.burger_norm : float = None 
        self.ase_system : Atoms = None
        
        if self.param_dict['structure'] not in self.implemented_lattice : 
            raise NotImplementedError(f'{self.param_dict["structure"]} : structure is not implemented ...')

        self.cubic_supercell = self.build_cubic_supercell()  
        print(f'... Initial number of atoms in supercell is {len(self.cubic_supercell)}')

        self.burger_norm = self.find_burger_vector_norm(threshold=1e-3)
        print(f'... Norm of Burger vector is {self.burger_norm} AA')

    def compute_cell_size(self) -> List[int] : 
        """Compute the size of the cubic supercell to put the dislocation
        
        Returns
        -------

        List[int]
            Replication list for ```SolidAse```
        """
        rbox = self.param_dict['size_loop']*self.param_dict['scale_loop']
        return [int(np.ceil(rbox/self.param_dict['a0'])) for _ in range(3)]
    
    def build_cubic_supercell(self) -> Atoms :
        """Build the initial ```Atoms``` object """
        size_list = self.compute_cell_size()
        print(f'... Cubic supercell : {size_list}')
        solid_ase_obj = SolidAse(size_list, 
                        self.param_dict['element'], 
                        self.param_dict['a0'])
        return solid_ase_obj.structure(self.param_dict['structure'])

    def find_burger_vector_norm(self, threshold = 1e-3) -> float : 
        """Find the norm of burger vector
        
        Parameters
        ----------

        threshold : float 
            Threshold for dot product
        """
        fake_solid_ase = SolidAse([2,2,2], 
                        self.param_dict['element'], 
                        self.param_dict['a0'])
        fake_lattice = fake_solid_ase.structure(self.param_dict['structure'])
        positions_fake_lattice = fake_lattice.positions
        
        list_rij = []
        for i, pos_i in enumerate(positions_fake_lattice) :
            for j, pos_j in enumerate(positions_fake_lattice) :
                if i < j : 
                    list_rij.append(pos_i-pos_j)

        array_rij = np.array(list_rij)
        normalised_array_rij = array_rij/np.linalg.norm(array_rij, axis = 1).reshape((array_rij.shape[0],1))        
        mask = np.abs(normalised_array_rij@self.normal_vector.T) > 1.0 - threshold

        oriented_rij = array_rij[mask]
        idx_min = np.argmin(np.linalg.norm(oriented_rij, axis=1))
        return np.linalg.norm(oriented_rij[idx_min])


    def recursive_unit_vectors(self, size : int = 3, possible_values : list[float] = [-1.0, 0.0, 1.0]) -> List[float] :
        """Build recursively all possible unit vectors for geometry ...
        
        Parameters
        ----------

        size : int
            size of each unit vector (generally 3)

        possible_values : list[float]
            Possible values for each component of the vector

        Returns 
        -------

        List[float]
            All possible unit vectors (possible_values^size)

        """
        if size == 1 : 
            return [[el] for el in possible_values]
        
        list_vector = []
        for previous_vector in self.recursive_unit_vectors(size=size-1) : 
            for el in possible_values : 
                list_vector += [previous_vector + [el]]

        return list_vector

    def get_orthogonal_vector(self) -> np.ndarray :
        """Get an orthogonal vector to the dislocation plane 
        
        Returns
        -------

        np.ndarray 
            Coordinate of the orthogonal vector
        """
        orientation = self.param_dict['orientation_loop']
        id_min = np.argmin( [np.abs(np.dot(orientation, np.array(el))) for el in self.possible_plane ] )
        return np.array(self.possible_plane[id_min])

    def rotation_matrix(self, axis : np.ndarray) :
        """build the rotation matrix method arrond a given axis"""
        return lambda theta : expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))

    def build_polygon_coordinates(self, nb_edge : int = 4, dict_shape : dict = {'kind':'circular',
                                                                                'r':1.0}) -> Tuple[List[np.ndarray], np.ndarray] : 

        def r(dict_shape : dict ,theta : float) -> float : 
            """Build raduis of ellipse depending on theta (from center)
            
            Parameters
            ----------

            dict_shape : dict 
                Shape dictionnary 

            theta : float 
                Angle 

            Returns
            -------

            float  
                r(theta)
            """
            exentricity = np.sqrt(1.0 - (dict_shape['b']**2)/(dict_shape['a']**2))
            return dict_shape['b']/np.sqrt(1 - exentricity**2*np.cos(theta))

        """Build polygon coordinates for non-circular dislocation"""
        in_plane_vector = self.get_orthogonal_vector()
        last_vector = np.cross(self.normal_vector,in_plane_vector)
        passage_matrix = np.concatenate((in_plane_vector,last_vector,self.normal_vector), axis=1)
        rotation_matrix_normal = self.rotation_matrix(self.normal_vector)

        list_theta = [ 2*np.pi*k/nb_edge for k in range(nb_edge) ]

        if dict_shape['kind'] == 'circular':
            list_vector = [dict_shape['r']*rotation_matrix_normal(theta)@in_plane_vector for theta in list_theta ]

        elif dict_shape['kind'] == 'elliptic' : 
            list_vector = [r(dict_shape,theta)*rotation_matrix_normal(theta)@in_plane_vector for theta in list_theta ]

        else :
            raise NotImplementedError(f'This kind of shape is not implemented : {dict_shape["kind"]}')

        return list_vector, passage_matrix

    def build_SIA(self, positions : np.ndarray, norm_burger_vector : float, orientation : np.ndarray) -> np.ndarray :
        """Build SIA for dislocation building 
        
        Parameters
        ----------

        positions : np.ndarray
            Position of the SIA center 
        
        norm_burger_vector : float 
            Norm of the Burger vector 

        orientation : str 
            Orientation of the Burger vector 

        Returns 
        -------

        np.ndarray
            Position of the SIA (2,3) matrix 
        
        """

        return np.array([positions - 0.5*norm_burger_vector*orientation, positions + 0.5*norm_burger_vector*orientation])

    def get_atoms_in_plane(self, tolerance : float = 1e-3) -> None : 
        """Find atoms in the dislocation loop plane and add ```in-plane``` property in 
        ```Atoms``` object
        
        Parameters
        ----------

        tolerance : float 
            Tolerance to be part of the plane
        """

        # We first find the closest atom to the center of the supercell
        idx_center = np.argmin(np.linalg.norm(self.cubic_supercell.positions - self.cubic_supercell.get_center_of_mass(), axis=1))
        self.center = self.cubic_supercell.positions[idx_center]

        # Compute hyperplane equation
        shift = np.dot(self.center,self.normal_vector)
        Natoms = len(self.cubic_supercell)

        in_plane_array = np.zeros(Natoms)
        positions = self.cubic_supercell.positions
        
        # compute atoms in plane
        metrics_array = np.abs(positions@self.normal_vector - shift)
        mask = metrics_array < tolerance

        #set new property ! 
        in_plane_array[mask] = 1.0
        self.cubic_supercell.set_array('in-plane', 
                                       #in_plane_array.reshape((Natoms,1)),
                                       in_plane_array,
                                       dtype=float)
        
        Nat_in_plane = len([el for el in in_plane_array if el > 0.0])
        print(f'... I found {Nat_in_plane} atoms in {self.param_dict["orientation_loop"]} plane')
        return 


    def build_circular_loop(self) -> Atoms :
        """Build circular loop !"""
        positions = self.cubic_supercell.positions
        Natoms = positions.shape[0]
        in_plane = self.cubic_supercell.get_array('in-plane')
        index_array = np.arange(Natoms).reshape(in_plane.shape)
        
        # build the index array to put SIA
        distance = np.linalg.norm(positions - self.center, axis=1).reshape((in_plane.shape))
        mask_distance = distance < 0.5*self.param_dict['size_loop'] 
        mask_plane = in_plane > 0.0

        in_loop_index_array = index_array[mask_distance & mask_plane]
        print(f'... I put {len(in_loop_index_array)} interstial atoms in the dislocation')

        #Build new atoms system to append !
        extra_atoms = Atoms()
        for idx in in_loop_index_array :
            positions_sia_idx = self.build_SIA(positions[idx],
                                               self.burger_norm,
                                               self.normal_vector)
            xtra = Atoms(2*[self.param_dict['element']],
                                     positions_sia_idx)
            extra_atoms += xtra

        # visualisation
        extra_atoms.set_array('defect',
                              np.ones((len(extra_atoms),)),
                              dtype=float)  

        dislocation_system = self.cubic_supercell.copy()
        del dislocation_system[[idx for idx in in_loop_index_array]]

        #visualisation
        dislocation_system.set_array('defect',
                                     np.zeros((len(dislocation_system),)),
                                     dtype=float)

        return dislocation_system + extra_atoms
    
    def build_polygonal_loop(self, nb_edge : int = 4, dict_shape : dict = {'kind':'elliptic',
                                                                                'a':1.0,
                                                                                'b':2.0}) -> Atoms : 
        positions = self.cubic_supercell.positions
        Natoms = positions.shape[0]
        in_plane = self.cubic_supercell.get_array('in-plane')
        
        # build the local basis and vectors for polygon
        list_vector, passage_matrix = self.build_polygon_coordinates(nb_edge=nb_edge,
                                                                     dict_shape=dict_shape)
        cartesian_vectors = np.array([ (passage_matrix@vect)/np.linalg.norm(vect) for vect in list_vector ])
        centered_positions = positions - self.center
        in_loop_index_array = []
        for idx, pos in enumerate(centered_positions) : 
            if (pos@cartesian_vectors.T < np.linalg.norm(list_vector, axis=1)).all() and in_plane[idx] > 0.0 :
                in_loop_index_array.append(idx)

        """
        dot_products = centered_positions @ cartesian_vectors.T

        # Get norms of list_vector
        list_vector_norms = np.linalg.norm(list_vector, axis=1)

        # Create a mask for the condition (pos @ cartesian_vectors.T < np.linalg.norm(list_vector, axis=1)).all()
        mask = np.all(dot_products < list_vector_norms, axis=1)

        # Combine the mask with the in_plane condition
        final_mask = mask & (in_plane > 0.0)

        # Get the indices that satisfy the condition
        in_loop_index_array = np.nonzero(final_mask)[0]"""


        print(f'... I put {len(in_loop_index_array)} interstial atoms in the dislocation')
        
        #Build new atoms system to append !
        extra_atoms = Atoms()
        for idx in in_loop_index_array :
            positions_sia_idx = self.build_SIA(positions[idx],
                                               self.burger_norm,
                                               self.normal_vector)
            xtra = Atoms(2*[self.param_dict['element']],
                                     positions_sia_idx)
            extra_atoms += xtra

        dislocation_system = self.cubic_supercell.copy()
        del dislocation_system[[idx for idx in in_loop_index_array]]

        return dislocation_system + extra_atoms      

    def write_dislocation(self, atoms : Atoms, path_writing : os.PathLike[str], format : str) -> None :
        """Write geometry file for dislocation loop 
        
        Parameters
        ----------

        atoms : Atoms 
            ```Atoms``` object containing the dislocation

        path_writing : os.PathLike[str]
            Path to the dislocation geometry file to write 

        format : str 
            Type geometry file to write (```lammps```, ```vasp``` ...)
        """
        write(path_writing, atoms, format=format)
        return 
    
    def BuildDislocation(self, 
                         writing_path : os.PathLike[str] = '/dislo.geom',
                         format : str = 'vasp',
                         dic_shape : dict = None,
                         ovito_mode : bool = False) -> None :
        
        self.get_atoms_in_plane()
        if dic_shape is None :
            dislocation = self.build_circular_loop()

        else : 
            restricted_dict_shape = {key:val for key, val in dic_shape.items() if key != 'nb_edge'}
            dislocation = self.build_polygonal_loop(dic_shape['nb_edge'],
                                                    dict_shape=restricted_dict_shape)

        lenght_scale = np.power(len(dislocation)/len(self.cubic_supercell), 0.3333)
        dislocation.set_cell(lenght_scale*dislocation.cell[:], scale_atoms=True)
        
        if not ovito_mode : 
            self.write_dislocation(dislocation,
                               writing_path,
                               format)

        else : 
            self.ase_system = dislocation

        return 
