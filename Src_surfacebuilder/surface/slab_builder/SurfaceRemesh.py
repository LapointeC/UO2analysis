import numpy as np 
from ase import Atom
from typing import List, Tuple, TypedDict
from scipy.spatial import KDTree
from itertools import combinations

def crossProdNew(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    """Cross product function compatible with VS...
    
    Parameters 
    ----------

    a : np.ndarray 
        First 3D vector

    b : np.ndarray 
        Second 3D vector

    Returns 
    -------

    np.ndarray 
        a x b

    """
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return np.asarray(c)

"""derivated mesh types"""
class DerivatedMesh(TypedDict) :
    atoms: List[Atom]
    neigh: List[List[Atom]] 

class SurfaceRemesher : 
    """This one is good !"""
    def __init__(self, list_selected_atom : List[Atom], 
                 rcut : float) -> None : 
        """SurfaceRemesher allows to identify atoms which are part of rough surface terminations
        
        Parameters
        ----------

        list_selected_atom : List[Atom]
            List of Atom object which are used as a priori for the rough surface termination

        rcut : float
            Internal cut off raduis used during the procedure

        """
        self.prior_surface_atom_dic : DerivatedMesh = {'atoms':list_selected_atom,'neigh':[]} 
        self.rcut = rcut 
        self.build_neigh_atoms()

    def build_neigh_atoms(self) -> None : 
        """"Build the rcut neighborhood of each atom in the prio attribute"""
        full_positions = np.asarray([at.position for at in self.prior_surface_atom_dic['atoms'] ])
        tree_system = KDTree(full_positions)
        for at in self.prior_surface_atom_dic['atoms'] : 
            list_id = tree_system.query_ball_point(at.position,self.rcut)
            list_at_neigh_rcut = [self.prior_surface_atom_dic['atoms'][idx] for idx in list_id]
            self.prior_surface_atom_dic['neigh'].append(list_at_neigh_rcut)

        return 

    def build_local_normal_surface(self,sub_list_atom : List[Atom], 
                                   central_atom : Atom, 
                                   terminaison : str = 'haut',
                                   alpha : float = 2.0) -> np.ndarray :
        """Build the average normal vector on a central atom (i) for a given sublist of neighbour atoms
        
        n_i = sum_{1 < j neq i < k neq i < L} \frac{r_ij \time r_ik}{\Vert r_ij \time r_ik \Vert^\alpha} 

        Parameters 
        ----------

        sub_list_atom : List[Atom]
            Sublist of neighbour atoms

        central_atom : Atom 
            Central atom where the normal vector will be computed

        termination : str
            Type of termination for the surface 'haut' or 'bas'

        alpha : float 
            Power used for the radial decrease of normal vector computation


        Returns 
        -------

        np.ndarray 
            Local normal for the central atom
        """
        #idx_central = self.prior_surface_atom_dic['atoms'].index(central_atom)
        #neigh_central_inter_sub = [at for at in self.prior_surface_atom_dic['neigh'][idx_central] if at in sub_list_atom ]
        #average_normal_vect = np.array([0.0,0.0,0.0])
        #compt = 0
        #for at_i in neigh_central_inter_sub : 
        #    for at_j in neigh_central_inter_sub : 
        #        cross_vector = np.cross(central_atom.position-at_i.position,central_atom.position-at_j.position)
        #        cross_vector *= 1.0/np.power(np.linalg.norm(cross_vector),alpha)
        #        dic_statement = {'haut':np.dot(cross_vector,np.asarray([0.0,0.0,1.0])) > 0.0, 'bas':np.dot(cross_vector,np.asarray([0.0,0.0,1.0])) < 0.0} 
        #        if dic_statement[terminaison] : 
        #            average_normal_vect += cross_vector 
        #            compt += 1
        #if compt > 0 :
        #    average_normal_vect *= 1.0/compt
        #return average_normal_vect
        
        idx_central = self.prior_surface_atom_dic['atoms'].index(central_atom)
    
        # Filter neighbors efficiently using set intersection
        neigh_central = self.prior_surface_atom_dic['neigh'][idx_central]
        neigh_central_inter_sub = list(set(neigh_central) & set(sub_list_atom))

        if not neigh_central_inter_sub:
            return np.zeros(3)

        # Precompute values
        z_axis = np.array([0.0, 0.0, 1.0])
        average_normal_vect = np.zeros(3)

        # Convert to NumPy arrays for efficient operations
        central_pos = np.asarray(central_atom.position)
        positions = np.array([at.position for at in neigh_central_inter_sub])

        # Compute cross products using broadcasting
        delta_i = central_pos - positions[:, np.newaxis, :]
        delta_j = central_pos - positions[np.newaxis, :, :]
        cross_vectors = np.cross(delta_i, delta_j, axis=2)  # Shape: (N, N, 3)

        # Normalize cross vectors efficiently
        norms = np.linalg.norm(cross_vectors, axis=2, keepdims=True) ** alpha
        norms[norms == 0] = 1  # Avoid division by zero
        cross_vectors /= norms

        # Compute dot product with z-axis
        dot_products = np.dot(cross_vectors.reshape(-1, 3), z_axis)

        # Apply condition and sum
        if terminaison == 'haut':
            mask = dot_products > 0
        else:  # terminaison == 'bas'
            mask = dot_products < 0

        valid_cross_vectors = cross_vectors.reshape(-1, 3)[mask]
        if valid_cross_vectors.size > 0:
            average_normal_vect = valid_cross_vectors.sum(axis=0) / valid_cross_vectors.shape[0]

        return average_normal_vect

    def build_sub_list(self,list_ini : List[Atom]) -> List[List[Atom]] : 
        """Build all the sub list from a given list list_ini
        
        Parameters 
        ----------

        list_ini : List[Atom]
            List of Atom object 

        Returns
        -------

        List[List[Atom]] 
            Full set of sublist generated from list_ini
        
        """
        sub_list = []
        for i in range(len(list_ini)+1) : 
            tmp = [list(el) for el in combinations(list_ini,i) if len(list(el)) > 0]
            if len(tmp) > 0 : 
                sub_list += tmp
        return sub_list

    def remesh_surface(self,list_atom_surface : List[Atom], 
                       terminaison : str ='haut', 
                       beta : float = 2.0) -> Tuple[int,List[Atom]] : 
        """Find atom at the termination of a rough surface, cost function is detailed in Latex document ...
        
        Parameters 
        ----------

        list_atom_surface : List[Atom]
            Pior list of atoms which are considered to be part of the surface 

        termination : str 
            Type of termination 'haut' or 'bas'

        beta : float 
            Power used to regularised the loss function in Latex document
        
        Returns 
        -------

        int 
            Cardinal of the subset of atom which is found to be optimal 

        List[Atom]
            List of atom which are part of the surface after the remeshing procedure

        """
        #max_card = 0
        #loss_function = 1e-4
        #remesh_list_atom_surface = None
        #all_sublist_atom_surface : List[List[Atom]] = self.build_sub_list(list_atom_surface)
        #for sublist_k_atom_surface in all_sublist_atom_surface : 
        #    complementaire_k = [at for at in list_atom_surface if at not in sublist_k_atom_surface]
        #    bool_positivity = True
        #    projection_k_value = 0.0
        #    for atom_i in sublist_k_atom_surface :
        #        average_normal_vector_i = self.build_local_normal_surface(sublist_k_atom_surface,atom_i,terminaison=terminaison) 
        #        for atom_j in complementaire_k :
        #                norme_ave = np.amax([np.linalg.norm(average_normal_vector_i),1e-4])
        #                projection_ij = np.dot(atom_i.position-atom_j.position,average_normal_vector_i)/(np.linalg.norm(atom_i.position-atom_j.position)*norme_ave)
        #                
        #                if projection_ij < 0 :
        #                    bool_positivity = False
        #                    break 
        #                else : 
        #                    projection_k_value += projection_ij

        #        if not bool_positivity : 
        #            break 

        #    if bool_positivity : 
        #        loss_function_k = projection_k_value + np.power(len(complementaire_k),1.0/beta)*np.power(len(sublist_k_atom_surface),beta)
        #        if loss_function_k > loss_function : 
        #            loss_function = loss_function_k
        #            max_card = len(sublist_k_atom_surface)
        #            remesh_list_atom_surface = sublist_k_atom_surface

        max_card = 0
        loss_function = 1e-4
        remesh_list_atom_surface = None  
        all_sublist_atom_surface = self.build_sub_list(list_atom_surface)

        for sublist_k_atom_surface in all_sublist_atom_surface:
            complementaire_k = set(list_atom_surface) - set(sublist_k_atom_surface)  # Faster lookup with sets
            projection_k_value = 0.0
            bool_positivity = True

            # Precompute average normal vectors for all atoms in the sublist
            normal_vectors = {
                atom: self.build_local_normal_surface(sublist_k_atom_surface, atom, terminaison=terminaison)
                for atom in sublist_k_atom_surface
            }

            for atom_i in sublist_k_atom_surface:
                average_normal_vector_i = normal_vectors[atom_i]

                # Convert positions to NumPy arrays for faster operations
                pos_i = np.array(atom_i.position)
                complementaire_positions = np.array([atom_j.position for atom_j in complementaire_k])

                # Compute distances and dot products in one go
                diffs = pos_i - complementaire_positions  # Shape: (M, 3)
                norms = np.linalg.norm(diffs, axis=1)  # Shape: (M,)
                norme_ave = max(np.linalg.norm(average_normal_vector_i), 1e-4)

                projections = np.dot(diffs, average_normal_vector_i) / (norms * norme_ave)

                # If any projection is negative, stop early
                if np.any(projections < 0):
                    bool_positivity = False
                    break

                projection_k_value += np.sum(projections)

            if bool_positivity:
                loss_function_k = projection_k_value + (len(complementaire_k) ** (1.0 / beta)) * (len(sublist_k_atom_surface) ** beta)
                if loss_function_k > loss_function:
                    loss_function = loss_function_k
                    max_card = len(sublist_k_atom_surface)
                    remesh_list_atom_surface = sublist_k_atom_surface

        return max_card, remesh_list_atom_surface