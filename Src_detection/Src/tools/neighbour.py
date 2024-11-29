import ase.neighborlist
import numpy as np

from ase import Atoms 
from typing import Optional, Tuple, List

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


def get_N_neighbour_graph(atoms : Atoms, 
                    extended_atoms : Atoms, 
                    cutoff : float, 
                    N : int, 
                    pbc : Tuple[bool, bool, bool] = (True, True, True), 
                    threshold : float = 1e-2) -> Tuple[np.ndarray, np.ndarray] :
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
    """

    array_neighbour = np.zeros((len(atoms),N,3))
    index_neighbour = np.zeros((len(atoms),N))
   
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
                    raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {r_kj_mask.shape[0]} / {N})')

                sorted_index_kj = np.argsort(np.linalg.norm(r_kj_mask, axis=1))
                index_neighbour[k,:] = neighbors_ij[1,:][mask_atom_k][sorted_index_kj][:N]
                array_neighbour[k,:,:] = r_kj_mask[sorted_index_kj][:N]
                
            
            else : 
                continue


    return array_neighbour, index_neighbour.astype(int)

def build_extended_neigh_(system : Atoms,
                          list_idx: List[int], 
                          rcut_extended: float = 4.5, 
                          rcut_full: float = 7.0) -> Tuple[Atoms, Atoms, Atoms]:

    """Build double shelled neighborhood for Nye tensor estimation. ```extended_system``` (where displacements tensor is also calculated)
    and ```full_sytem``` will be returned.
    
    Parameters
    ----------

    system : Atoms 
        Initial Atoms object containing the whole system

    list_idx : List[int]
        Subset of indexes corresponding to the atoms guessed to be part of dislocations

    rcut_extended : float 
        First shell cut off raduis (defining ```extended_system```)

    rcut_full : float
        Second shell cut off raduis (defining ```full_system```)


    Returns
    -------

    Atoms 
        Atoms object containing the atoms guessed to be part of dislocations

    Atoms 
        Atoms object containing first shell atoms (```extended_system```)

    Atoms 
        Atoms object containing second shell atoms (```full_system```)
    """    

    if rcut_extended > rcut_full:
        raise ValueError(f'full rcut ({rcut_full} AA) is lower than extended rcut ({rcut_extended} AA)')
    
    dislo_system = system[list_idx]
    positions_d = dislo_system.positions
    positions_f = system.positions
    rcut_full_sq = rcut_full**2
    rcut_extended_sq = rcut_extended**2
    
    full_set = set()
    ext_set = set()
    
    for _, pos_d in enumerate(positions_d) :
        # Calculate the squared distances
        diff = positions_f - pos_d
        distances_sq = np.einsum('ij,ij->i', diff, diff)
    
        # Find indices where distance is within rcut_full
        within_full = np.where((distances_sq > 0) & (distances_sq < rcut_full_sq))[0]
        full_set.update(within_full)
    
        # Find indices where distance is within rcut_extended
        within_extended = np.where(distances_sq < rcut_extended_sq)[0]
        ext_set.update(within_extended)
    
    full_list = list(full_set)
    ext_list = list(ext_set)
    
    #organise lists for Nye tensor
    ext_list = list_idx + [el for el in ext_list if el not in list_idx]
    full_list = ext_list + [el for el in full_list if el not in ext_list]

    # Update the local, extended, and full dislocation data
    local_dislocation = system.copy()[list_idx]
    extended_dislocation = system.copy()[ext_list]
    full_dislocation = system.copy()[full_list]
    
    return local_dislocation, extended_dislocation, full_dislocation

def get_N_neighbour_huge(atoms : Atoms, 
                    extended_atoms : Atoms,
                    cutoff : float, 
                    N : int) -> Tuple[np.ndarray, np.ndarray] :
    """Build the truncated neighborhood for a given system of atoms. Neighborhood is truncated to the N neighbour.
    Two neighborhood are computed : (i) for the system and (ii) for the extended system. These calculation are mandatory
    to evaluate Nye tensor of the system. This version does not take into account the periodic bondary conditions but is very fast !

    Parameters
    ----------

    atoms : Atoms 
        System to compute the neighborhood

    extended_atoms : Atoms 
        Extended system to compute the neighborhood
    
    cutoff : float 
        Cutoff raduis to compute neighbours
    
    N : int 
        Truncation bound for number of neighbours of each atoms
    
    Returns:
    --------

    np.ndarray 
        array of neighbours for ATOMS SYSTEM (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

    np.ndarray 
        array of index of neighbours for ATOMS SYSTEM (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}

            """


    # building neighbour arrays
    array_neighbour = np.zeros((len(atoms),N,3))
    index_neighbour = np.zeros((len(atoms),N))

    # starting with extended neighbour 
    positions_e = extended_atoms.positions
    rcut_full_sq = cutoff**2

    # here is the dislocation system 
    positions_d = atoms.positions

    for id_d, pos_d in enumerate(positions_d) :
        # Calculate the squared distances
        diff = positions_e - pos_d
        distances_sq = np.einsum('ij,ij->i', diff, diff)
    
        # Find indices where distance is within rcut_full
        within_full = np.where((distances_sq > 0) & (distances_sq < rcut_full_sq))[0]
        
        # Building order neighbour list ! 
        if len(within_full) < N : 
            raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {len(within_full)}/{N})')
        sorted_index_kd = np.argsort(distances_sq[within_full])
        sorted_neigh_kd = within_full[sorted_index_kd][:N]

        # Update neighbours
        array_neighbour[id_d,:,:] = positions_e[sorted_neigh_kd] - pos_d
        index_neighbour[id_d,:] = sorted_neigh_kd


    return array_neighbour, index_neighbour.astype(int)

def get_N_neighbour_Cosmin(system : Atoms, 
                           extended_system : Atoms,
                           cell : np.ndarray, 
                           cutoff_distance : float, 
                           N : int) -> Tuple[np.ndarray, np.ndarray]:
    """Build the truncated neighborhood for a given system of atoms. Neighborhood is truncated to the N neighbour.
    Two neighborhood are computed : (i) for the system and (ii) for the extended system. These calculation are mandatory
    to evaluate Nye tensor of the system. Fast implementation from Cosmin that is taking into acount periodic boundary conditions

    Parameters
    ----------

    atoms : Atoms 
        System to compute the neighborhood

    extended_atoms : Atoms 
        Extended system to compute the neighborhood
    
    cell : np.ndarray 
        Supercell containing the whole system to take into account pbc

    cutoff : float 
        Cutoff raduis to compute neighbours
    
    N : int 
        Truncation bound for number of neighbours of each atoms
    
    Returns:
    --------

    np.ndarray 
        array of neighbours for ATOMS SYSTEM (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

    np.ndarray 
        array of index of neighbours for ATOMS SYSTEM (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}
    """

    # Build positions arrays
    positions_d = system.positions
    positions_e = extended_system.positions

    # building neighbour arrays
    array_neighbour = np.zeros((len(system),N,3))
    index_neighbour = np.zeros((len(system),N))

    # Find neighbors for each atom using neighboring cells
    for i_d in range(positions_d.shape[0]):
        dX = - positions_d[i_d] + positions_e
        dX -= np.round(np.dot(dX, np.linalg.inv(cell))) @ cell        
        distance = np.sum(dX**2, axis=1)

        within_full = np.where((distance < cutoff_distance**2) & (distance > 0))[0]

        if len(within_full) < N : 
            raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {len(within_full)}/{N})')
        sorted_index_kd = np.argsort(distance[within_full])
        sorted_neigh_kd = within_full[sorted_index_kd][:N]

        # Update neighbours
        array_neighbour[i_d,:,:] = dX[sorted_neigh_kd]
        index_neighbour[i_d,:] = sorted_neigh_kd

    return array_neighbour, index_neighbour.astype(int)


def get_N_neighbour(system : Atoms, 
                           extended_system : Atoms,
                           full_system : Atoms, 
                           cutoff_distance : float, 
                           N : int,
                           kind_neigh : str = 'graph-ase'):
    """Build the truncated neighborhood for a given system of atoms. Neighborhood is truncated to the N neighbour.
    Two neighborhood are computed : (i) for the system and (ii) for the extended system. These calculation are mandatory
    to evaluate Nye tensor of the system. Fast implementation from Cosmin that is taking into acount periodic boundary conditions

    Parameters
    ----------

    atoms : Atoms 
        System to compute the neighborhood

    extended_atoms : Atoms 
        Extended system to compute the neighborhood
    
    full_atoms : Atoms 
        Second overshell used to compute the neighborhood of extended_atoms system

    cell : np.ndarray 
        Supercell containing the whole system to take into account pbc

    cutoff : float 
        Cutoff raduis to compute neighbours
    
    N : int 
        Truncation bound for number of neighbours of each atoms
    
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

    list_implemented_kind = ['graph-ase', 'fast', 'fast-pbc']
    if kind_neigh not in list_implemented_kind :
        raise NotImplementedError(f'... Kind : {kind_neigh} is not implemented, possible kind are {list_implemented_kind} ...')

    if kind_neigh == 'graph-ase' : 
        pbc = (True,True,True)
        array_neighbour_extended, index_neighbour_extended = get_N_neighbour_graph(extended_system,
                                                                                   full_system,
                                                                                   cutoff_distance,
                                                                                   N,
                                                                                   pbc=pbc)
        array_neighbour, index_neighbour = get_N_neighbour_graph(system,
                                                                 extended_system,
                                                                 cutoff_distance,
                                                                 N,
                                                                 pbc=pbc)

    elif kind_neigh == 'fast' : 
        array_neighbour_extended, index_neighbour_extended = get_N_neighbour_huge(extended_system,
                                                                                  full_system,
                                                                                  cutoff_distance,
                                                                                  N)
        array_neighbour, index_neighbour = get_N_neighbour_huge(system,
                                                                extended_system,
                                                                cutoff_distance,
                                                                N)

    elif kind_neigh == 'fast-pbc' :
        cell = full_system.cell[:]
        array_neighbour_extended, index_neighbour_extended = get_N_neighbour_Cosmin(extended_system,
                                                                                    full_system,
                                                                                    cell,
                                                                                    cutoff_distance,
                                                                                    N)
        array_neighbour, index_neighbour = get_N_neighbour_Cosmin(system,
                                                                  extended_system,
                                                                  cell,
                                                                  cutoff_distance,
                                                                  N)

    return array_neighbour, index_neighbour.astype(int), array_neighbour_extended, index_neighbour_extended.astype(int)