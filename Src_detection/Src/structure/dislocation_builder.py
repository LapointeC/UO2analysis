import os
import numpy as np
from warnings import warn
from .lattice import SolidAse

from scipy.linalg import expm
from ase.io import write
from ase import Atoms

from typing import List, Tuple, TypedDict
from ..tools import timeit

class InputsDictDislocationBuilder(TypedDict) :
    """Little class for ```dislocation_builder``` inputs 
    Possible keys of the dictionnary are : 
    - ```structure``` : type of cristallographic strucutre (bcc, fcc ...)
    - ```a0``` : lattice parameter of the system in AA
    - ```size_loop``` : approximate loop size in AA
    - ```scale_loop``` : scaling factor to fix the supercell size (```size_loop*scale_loop```) in AA
    - ```orientation_loop``` : normal vector to the loop plane 
    - ```element``` : species of the system 
    """
    structure : str
    a0 : float
    size_loop : float
    scale_loop : float
    orientation_loop : np.ndarray
    element : str

class DislocationsBuilder :
    """Allows to build interstial dislocation loops working for bcc systems
    Based on M.C.M work : https://github.com/mcmarinica/Insert_Sias

    Parameters
    ----------

    inputs_dict : InputsDictDislocationBuilder
        Dictionnary of parameter to build dislocation loop

    see also : https://github.com/mcmarinica/Insert_Sias"""
    def __init__(self, inputs_dict : InputsDictDislocationBuilder) -> None :
        self.param_dict = inputs_dict
        self.implemented_lattice = ['BCC']
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

    def get_orthogonal_vector(self, tolerance : float = 1e-3) -> np.ndarray :
        """Get an orthogonal vector to the dislocation plane 
        
        Parameters
        ----------

        tolerance : float 
            tolerance of dot product for orthogonality

        Returns
        -------

        np.ndarray 
            Coordinate of the orthogonal vector
        """
        orientation = self.param_dict['orientation_loop']
        id_min = np.argmin( [np.abs(np.dot(orientation, np.array(el))) for el in self.possible_plane ] )

        dot_product = np.abs(np.dot(orientation, np.array(self.possible_plane[id_min])))
        if dot_product > tolerance : 
            warn(f'Could not find orthogonal vector to the plane n.v = {dot_product}')

        return np.array(self.possible_plane[id_min])

    def rotation_matrix(self, axis : np.ndarray) :
        """build the rotation matrix method arrond a given axis"""
        return lambda theta : expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))

    def build_polygon_coordinates(self, nb_edge : int = 4, dict_shape : dict = {'kind':'circular'}) -> Tuple[List[np.ndarray], np.ndarray] : 
        """Build the coordinates of regular polygon vertices in the dislocation plane coordinates
        
        Parameters
        ----------

        nb_edge : int 
            Number of polygon edges

        dic_shape : dict 
            Dictionnary which define the kink of shape for dislocation
            Example : ```{'nb_edge':4,'kind':'elliptic','b/a':2.0}``` or ```{'nb_edge':4,'kind':'circular'}```
        
        
        Returns
        -------

        List[np.ndarray]
            List of polygon vertices coordinates in the dislocation plane coordinates
        
        np.ndarray 
            Passage matrix between vertices coordinates and cartesian coordinates ```(V_{x_i 1}, V_{x_i 2} ,V_{x_i 3})```
        """


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
            exentricity = np.sqrt(1.0 - dict_shape['b/a']**2)
            scaled_a2 = 0.25*self.param_dict['size_loop']**2*(1.0 - exentricity**2)
            return np.sqrt(scaled_a2/(1.0 - exentricity**2*np.cos(theta)**2 ))

        """Build polygon coordinates for non-circular dislocation"""
        in_plane_vector = self.get_orthogonal_vector()
        last_vector = np.cross(self.normal_vector,in_plane_vector)
        passage_matrix = np.concatenate((in_plane_vector.reshape((3,1)) / np.linalg.norm(in_plane_vector),
                                        last_vector.reshape((3,1)) / np.linalg.norm(last_vector) ,
                                        self.normal_vector.reshape((3,1)) ), axis=1)

        rotation_matrix_normal = self.rotation_matrix(self.normal_vector)

        list_theta = [ 2*np.pi*k/nb_edge for k in range(nb_edge) ]

        if dict_shape['kind'] == 'circular':
            #in_plane_vector = np.array([1., 0., 0.])
            list_vector = [0.5*self.param_dict['size_loop']*rotation_matrix_normal(theta)@in_plane_vector for theta in list_theta ]

        elif dict_shape['kind'] == 'elliptic' : 
            
            #sanity check !
            if dict_shape['b/a'] > 1.0 :
                    raise TimeoutError(f'b/a shoud be less than 1.0 : {dict_shape["b/a"]}')
            if not 'b/a' in dict_shape.keys() :
                raise NotImplementedError(f'b/a key is missing to build elliptic loop...')
            
            #in_plane_vector = np.array([1.,0.,0.])
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
                                       in_plane_array,
                                       dtype=float)
        
        Nat_in_plane = len([el for el in in_plane_array if el > 0.0])
        print(f'... I found {Nat_in_plane} atoms in {self.param_dict["orientation_loop"]} plane')
        return 


    def build_circular_loop(self) -> Atoms :
        """Build circular loop and return ```Atoms``` object containing the whole system 
        
        Returns
        -------

        Atoms
            ```Atoms``` object containing the whole system (dislocation + bulk)
        """
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
                                                                                'b/a':2.0}) -> Atoms : 
        """Build polygonal loop and return ```Atoms``` object containing the whole system 
        
        Parameters
        ----------

        nb_edge : int 
            Number of polygon edges

        dic_shape : dict 
            Dictionnary which define the kink of shape for dislocation
            Example : ```{'nb_edge':4,'kind':'elliptic','b/a':2.0}``` or ```{'nb_edge':4,'kind':'circular'}```
        Returns
        -------

        Atoms
            ```Atoms``` object containing the whole system (dislocation + bulk)
        """

        positions = self.cubic_supercell.positions
        in_plane = self.cubic_supercell.get_array('in-plane')
        
        # build the local basis and vectors for polygon
        list_vector, _ = self.build_polygon_coordinates(nb_edge=nb_edge,
                                                                     dict_shape=dict_shape)
        centered_positions = positions - self.center
        
        cartesian_vectors = np.array(list_vector)/np.linalg.norm(list_vector, axis=1).reshape((len(list_vector),1))
        """in_loop_index_array = []
        for idx, pos in enumerate(centered_positions) : 
            if (pos@cartesian_vectors.T < np.linalg.norm(list_vector, axis=1)).all() and in_plane[idx] > 0.0 :
                in_loop_index_array.append(idx)"""

        
        dot_products = centered_positions @ cartesian_vectors.T
        list_vector_norms = np.linalg.norm(list_vector, axis=1)

        # Create a mask for the condition (pos @ cartesian_vectors.T < np.linalg.norm(list_vector, axis=1)).all()
        mask_dot = np.all(dot_products < list_vector_norms, axis=1)
        final_mask = mask_dot & (in_plane > 0.0)

        # Get the indices that satisfy the condition
        in_loop_index_array = np.nonzero(final_mask)[0]


        print(f'... I put {len(in_loop_index_array)} interstial atoms in the dislocation')
        
        #Build new atoms system to append !
        extra_atoms = Atoms()
        for idx in in_loop_index_array :
            positions_sia_idx = self.build_SIA(positions[idx],
                                               self.burger_norm,
                                               self.normal_vector)
            extra = Atoms(2*[self.param_dict['element']],
                                     positions_sia_idx)
            extra_atoms += extra

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
    
    @timeit
    def BuildDislocation(self, 
                         writing_path : os.PathLike[str] = '/dislo.geom',
                         format : str = 'vasp',
                         dic_shape : dict = None,
                         ovito_mode : bool = False) -> None :
        """Build dislocation from ```InputsDictDislocationBuilder``` dictionnary and 
        ```dic_shape``` dictionnary 
        
        Parameters
        ----------

        writing_path : os.Pathlike[str]
            Writing path for dislocation geometry file

        format : str 
            Type geometry file to write (```lammps```, ```vasp``` ...)
        
        dic_shape : dict 
            Dictionnary which define the kink of shape for dislocation if it sets to ```None``` circular loop is build otherwise polygonal loop is built :
            Example : ```{'nb_edge':4,'kind':'elliptic','b/a':2.0}``` or ```{'nb_edge':4,'kind':'circular'}```
        
        ovito_mode : bool 
            If bool is True there no geom file written
        """

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
