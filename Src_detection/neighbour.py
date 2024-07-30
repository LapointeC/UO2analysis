import ase.neighborlist
import numpy as np

def sort_distance_composite( array_ij : np.ndarray ) -> np.ndarray :
    sorted_distance_array = sorted( array_ij, key=lambda x: x[1])
    return np.ndarray([ rij for rij, _ in  sorted_distance_array])

def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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


def get_N_neighbour(atoms : Atoms, cutoff : float, N : int, pbc : Tuple[bool, bool, bool] = (True, True, True)) -> Tuple[np.ndarray,np.ndarray] :
    neighbors_ij, neighbors_shift, _, d_ij = get_neighborhood(atoms.positions,
                                                       cutoff,
                                                       pbc,
                                                       atoms.cell[:],
                                                       true_self_interaction=False)

    array_neighbour = np.empty((len(atoms),N,3))
    index_neighbour = np.empty((len(atoms),N))
    for k in range(len(atoms)) :
        mask_atom_k = neighbors_ij[0,:] == k
        r_kj_mask = atoms.positions[ neighbors_ij[mask_atom_k][1,:] ] - atoms.positions[ neighbors_ij[mask_atom_k][0,:] ] + neighbors_shift[mask_atom_k]
        d_kj_mask = d_ij[mask_atom_k]

        if len(d_kj_mask) < N :
            raise ValueError(f'Number of neighbour is too small to continue, cutoff raduis should be increased (N found = {len(d_kj_mask)})')

        sorted_index_kj = np.argsort(d_kj_mask)
        index_neighbour[k,:] = neighbors_ij[mask_atom_k][1,:][sorted_index_kj][:N]
        array_neighbour[k,:,:] = r_kj_mask[sorted_index_kj][:N]


    return array_neighbour, index_neighbour
