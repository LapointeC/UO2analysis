import ase.neighborlist
import numpy as np

from ase import Atoms 
from typing import Optional, Tuple

def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None, 
    true_self_interaction=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ###########################################################################################
    # Neighborhood construction
    # Authors: Ilyes Batatia, Gregor Simm
    # This program is distributed under the MIT License (see MIT.md)
    ###########################################################################################
    """Build the neigbour list for a given positions vector (by including pbc)
    
    Parameters
    ----------

    positions : np.ndarray 
        Positions vector to build the neighbour list

    cutoff : float 
        Cutoff distance to build the neighborhood of each atom (in AA)

    pbc : Tuple[bool, bool, bool]
        Pbc tuple to build neighborhood

    cell : np.ndarray 
        Supercell associated to the positions vector, needed to take into account pbc ! 

    Return:
    -------

    np.ndarray 
        (M,2) array containing for each line i -> neigh(i) (sender -> receiver)

    np.ndarray 
        (M,3) array containing pbc shift (cartesian coordinates) for each line i -> shit(neigh(i)) 

    np.ndarray 
        (M,3) array containing pbc shift (cell coordinates) for each line i -> shit(neigh(i)) 

    np.ndarray 
        (M,3) array containing d_{i neig(i)} for each line i -> d(neigh(i))  
    """

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    sender, receiver, unit_shifts, distance = ase.neighborlist.primitive_neighbor_list(
        quantities="ijSD",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
        use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, distance


def get_N_neighbour(atoms : Atoms, 
                    extended_atoms : Atoms, 
                    full_atoms : Atoms, 
                    cutoff : float, 
                    N : int, 
                    pbc : Tuple[bool, bool, bool] = (True, True, True), 
                    threshold : float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    """Build the truncated neighborhood for a given system of atoms. Neighborhood is truncated to the N neighbour.
    Two neighborhood are computed : (i) for the system and (ii) for the extended system. These calculation are mandatory
    to evaluate Nye tensor of the system.

    Parameters
    ----------

    atoms : Atoms 
        System to compute the neighborhood

    extended_atoms : Atoms 
        Extended system to compute the neighborhood
    
    full_atoms : Atoms 
        Second overshell used to compute the neighborhood of extended_atoms system

    cutoff : float 
        Cutoff raduis to compute neighbours
    
    N : int 
        Truncation bound for number of neighbours of each atoms

    pbc : Tuple[bool, bool, bool]
        Pbc tuple to compute neighborhood

    threshold : float 
        Norm treshold to identify an atom of full_atoms system to be part of atoms or/and extended_atoms system
    
    
    Returns:
    --------

    np.ndarray 
        array of neighbours for ATOMS SYSTEM (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

    np.ndarray 
        array of index of neighbours for ATOMS SYSTEM (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}

    np.ndarray 
        array of neighbours for EXTENDED_ATOMS SYSTEM (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

    np.ndarray 
        array of index of neighbours for EXTENDED_ATOMS SYSTEM (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}
    """

    array_neighbour = np.zeros((len(atoms),N,3))
    index_neighbour = np.zeros((len(atoms),N))
    array_neighbour_extended = np.zeros((len(extended_atoms),N,3))
    index_neighbour_extended = np.zeros((len(extended_atoms),N))

    neighbors_ij, neighbors_shift, _, _ = get_neighborhood(full_atoms.positions,
                                                       cutoff,
                                                       pbc,
                                                       full_atoms.cell[:],
                                                       true_self_interaction=False)

    for full_id, full_at in enumerate(full_atoms) :
            #check if atom is in the extended system
            composite_id_distance = [(idx, np.linalg.norm(ext_at.position - full_at.position)) for idx, ext_at in enumerate(extended_atoms) ]
            k, min_dist =  sorted(composite_id_distance, key=lambda x: x[1])[0]
            if min_dist < threshold :
                mask_atom_k = neighbors_ij[0,:] == full_id
                r_kj_mask = full_atoms.positions[ neighbors_ij[1,:][mask_atom_k] ] - full_atoms.positions[ neighbors_ij[0,:][mask_atom_k] ] + neighbors_shift[mask_atom_k]

                if r_kj_mask.shape[0] < N :
                    raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {r_kj_mask.shape[0]})')

                sorted_index_kj = np.argsort(np.linalg.norm(r_kj_mask, axis=1))
                index_neighbour_extended[k,:] = neighbors_ij[1,:][mask_atom_k][sorted_index_kj][:N]
                array_neighbour_extended[k,:,:] = r_kj_mask[sorted_index_kj][:N]
                
            
            else : 
                continue
    
    neighbors_ij, neighbors_shift, _, _ = get_neighborhood(extended_atoms.positions,
                                                   cutoff,
                                                   pbc,
                                                   extended_atoms.cell[:],
                                                   true_self_interaction=False)

    for ext_id, ext_at in enumerate(extended_atoms) :
            #check if atom is in the extended system
            composite_id_distance = [(idx, np.linalg.norm(at.position - ext_at.position)) for idx, at in enumerate(atoms) ]
            k, min_dist =  sorted(composite_id_distance, key=lambda x: x[1])[0]
            if min_dist < threshold :
                mask_atom_k = neighbors_ij[0,:] == ext_id #k
                r_kj_mask = extended_atoms.positions[ neighbors_ij[1,:][mask_atom_k] ] - extended_atoms.positions[ neighbors_ij[0,:][mask_atom_k] ] + neighbors_shift[mask_atom_k]

                if r_kj_mask.shape[0] < N :
                    raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {r_kj_mask.shape[0]})')

                sorted_index_kj = np.argsort(np.linalg.norm(r_kj_mask, axis=1))
                index_neighbour[k,:] = neighbors_ij[1,:][mask_atom_k][sorted_index_kj][:N]
                array_neighbour[k,:,:] = r_kj_mask[sorted_index_kj][:N]
                
            
            else : 
                continue


    return array_neighbour, index_neighbour.astype(int), array_neighbour_extended, index_neighbour_extended.astype(int)
