import numpy as np
import ase.neighborlist

from ..mld import DBManager
from ..tools.neighbour import get_N_neighbour_Cosmin
from ase import Atoms

from typing import List, Dict, Tuple, TypedDict, Optional


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

class Configuration(TypedDict) :
    atoms : Atoms
    equiv_descriptor : np.ndarray

class FastEquivariantDescriptor : 
    def __init__(self, dbmodel : DBManager, rcut : float = 4.7) -> None :
        self.dbmodel = dbmodel
        self.configurations : Dict[str,Configuration] = {}
        self.rcut = rcut

    def _build_N_neighbour(self, atoms : Atoms, pbc : Tuple[bool, bool, bool] = (True, True, True)) -> np.ndarray:
        neighbors_ij, _, _, _ = get_neighborhood(atoms.positions,
                                                           self.rcut,
                                                           pbc,
                                                           atoms.cell[:],
                                                           true_self_interaction=False)
     
        _, count = np.unique(neighbors_ij[0,:], return_counts=True)
        size_neigh = np.amax(count)
        index_neighbour = np.empty((len(atoms),size_neigh), dtype=int)
        for k in range(len(atoms)) :
            mask_atom_k = neighbors_ij[0,:] == k
            index_neighbour[k,:] = np.resize(neighbors_ij[1,:][mask_atom_k], size_neigh)

        return index_neighbour

    def _build_N_neighbour_fast(self, atoms : Atoms, N : int) -> np.ndarray : 
        _, array_id = get_N_neighbour_Cosmin(atoms,
                                             atoms,
                                             atoms.cell[:],
                                             self.rcut,
                                             N)
        return array_id

    def _build_local_equivariant_desc(self, descriptor : np.ndarray, sub_set_index : np.ndarray, desc_dim : int = None) -> np.ndarray :
        sub_set_desc = descriptor[np.unique(sub_set_index),:]
        sub_covariance_matrix = sub_set_desc.T @ sub_set_desc / (sub_set_desc.shape[0] - 1)
        eig_val = np.linalg.eigvalsh(sub_covariance_matrix, UPLO='L')
        eig_val = -np.sort(-eig_val)
        if desc_dim is not None : 
            eig_val = eig_val[:desc_dim]

        print(eig_val, len([eig for eig in eig_val if abs(eig) > 1e-10]))
        return eig_val

    def _equivariant_desc_per_config(self, atoms : Atoms, descriptor : np.ndarray) -> np.ndarray :
        index_neighbour = self._build_N_neighbour(atoms, pbc = (True, True, True))
        equivariant_desc_config = np.empty((descriptor.shape))
        for id_k in range(len(index_neighbour)) : 
            equivariant_desc_config[id_k,:] = self._build_local_equivariant_desc(descriptor, index_neighbour[id_k,:], desc_dim=None)
        
        return equivariant_desc_config

    def _equivariant_desc_per_config_fast(self, atoms : Atoms, descriptor : np.ndarray, maxN : int = 30) -> np.ndarray :
        index_neighbour = self._build_N_neighbour_fast(atoms, maxN)
        equivariant_desc_config = np.empty((descriptor.shape))
        for id_k in range(len(index_neighbour)) : 
            equivariant_desc_config[id_k,:] = self._build_local_equivariant_desc(descriptor, index_neighbour[id_k,:], desc_dim=None)
        
        return equivariant_desc_config

    def BuildEquivariantDescriptors(self) -> None :
        for struc, sub_dic in self.dbmodel.model_init_dic.items() : 
            atoms_struc = sub_dic['atoms']
            descriptor = sub_dic['atoms'].get_array('milady-descriptors')

            self.configurations[struc] = {'atoms':None, 'equiv_descriptor':None}
            self.configurations[struc]['atoms'] = atoms_struc
            self.configurations[struc]['equiv_descriptor'] = self._equivariant_desc_per_config(atoms_struc, descriptor)

    def GetConfigurations(self) -> Dict[str,Configuration] : 
        return self.configurations