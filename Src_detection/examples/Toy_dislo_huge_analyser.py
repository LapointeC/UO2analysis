import os, sys  
import numpy as np

#Line needed to work with ovito graphical mode
sys.path.append(os.getcwd())

from ase.io import read 
from ase import Atoms, Atom
sys.path.insert(0,'../')
from Src import DislocationObject, DfctAnalysisObject, \
                  FrameOvito, NaiveOvitoModifier, \
                  timeit, get_N_neighbour

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, PythonSource, StaticSource
from ovito.io import *

from ovito.modifiers import VoronoiAnalysisModifier, CommonNeighborAnalysisModifier, CoordinationAnalysisModifier

from typing import Tuple, List
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication


class ToyDislocationAnalyser : 
    def __init__(self, path_dislo : os.PathLike[str], data_type : str = 'lammps-data', structure : str = 'bcc') -> None :
        self.dislocation = read(path_dislo, format=data_type)

        self.inv_dict_struct = {'other':0,
                                'fcc':1,
                                'hcp':2,
                                'bcc':3,
                                'ico':4}
        
        self.target_struct_str = structure
        self.target_struct = self.inv_dict_struct[structure]
        self.dislocation_object : DislocationObject = None

    def extend_dislocation(self, tuple_repeat = (1,1,1)) -> None : 
        self.dislocation *= tuple_repeat

    def get_outliers(self, rcut : float = 3.5) -> List[int] : 
        data_ovito = ase_to_ovito(self.dislocation)
        pipeline = Pipeline(source = StaticSource(data = data_ovito))
        cna = CommonNeighborAnalysisModifier(cutoff=rcut)

        #pipeline.modifiers.append(voro)
        pipeline.modifiers.append(cna)
        data = pipeline.compute()
        
        structure_type = data.particles['Structure Type'][:]
        id_atoms = np.array([k for k in range(len(structure_type))])

        #select only atoms identified as different from structure
        mask_struct = structure_type != self.target_struct 
        id_atoms_bar = id_atoms[mask_struct]

        return id_atoms_bar

    @timeit
    def build_extended_neigh_(self, list_idx: List[int], rcut_extended: float = 4.5, rcut_full: float = 7.0) -> Tuple[List[int], List[int]]:
        if rcut_extended > rcut_full:
            raise ValueError(f'full rcut ({rcut_full} AA) is lower than extended rcut ({rcut_extended} AA)')

        dislo_system = self.dislocation[list_idx]
        positions_d = dislo_system.positions
        positions_f = self.dislocation.positions

        rcut_full_sq = rcut_full**2
        rcut_extended_sq = rcut_extended**2

        full_set = set()
        ext_set = set()

        for id_d, pos_d in enumerate(positions_d) :
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

        # Update the local, extended, and full dislocation data
        self.local_dislocation = self.dislocation.copy()[list_idx]
        self.extended_dislocation = self.dislocation.copy()[ext_list]
        self.full_dislocation = self.dislocation.copy()[full_list]

        return ext_list, full_list

    @timeit
    def build_extended_neigh(self, list_idx: List[int], rcut_extended: float = 4.5, rcut_full: float = 7.0) -> Tuple[List[int], List[int]]:
        if rcut_extended > rcut_full:
            raise ValueError(f'full rcut ({rcut_full} AA) is lower than extended rcut ({rcut_extended} AA)')

        positions = self.dislocation.positions

        # Get the positions of the selected indices
        r_ids = positions[list_idx, :]

        # Calculate squared distances to avoid unnecessary square root computations
        dist_squared = np.sum((positions[:, np.newaxis, :] - r_ids[np.newaxis, :, :]) ** 2, axis=2)

        # Create boolean masks for the two cutoff distances
        ext_mask = dist_squared < rcut_extended ** 2
        full_mask = dist_squared < rcut_full ** 2

        # Find unique indices within the extended and full cutoff ranges
        ext_list = np.unique(np.where(ext_mask)[0])
        full_list = np.unique(np.where(full_mask)[0])

        # Update the local, extended, and full dislocation data
        self.local_dislocation = self.dislocation.copy()[list_idx]
        self.extended_dislocation = self.dislocation.copy()[ext_list]
        self.full_dislocation = self.dislocation.copy()[full_list]

        return ext_list, full_list


    @timeit
    def perform_dislocation_analysis(self, unit_cell : np.ndarray, 
                                     rcut_dislo = 5.0, 
                                     rcut_line = 4.5, 
                                     rcut_cluster = 5.0,
                                     scale_cluster = 1.5,
                                     nb_averaging_window=2,
                                     rcut_burger = 4.5) -> Tuple[List[int], Atoms] : 
        self.dislocation_object = DislocationObject(self.local_dislocation, 
                                                    self.extended_dislocation, 
                                                    self.full_dislocation, 
                                                    rcut_dislo, 
                                                    structure={'structure':self.target_struct_str,
                                                               'unit_cell':unit_cell},
                                                    kind_neigh='fast')
        print('... Starting Nye tensor computation ...')
        nye, array_neigh, index_neigh, array_neigh_ext, index_neigh_ext = self.dislocation_object.NyeTensor()
        print('... Starting building of sampling line ...')
        idx_list = self.dislocation_object.BuildSamplingLine(rcut_line=rcut_line, rcut_cluster=rcut_cluster, scale_cluster=scale_cluster) 
        self.dislocation_object.RefineSamplingLine(scale=scale_cluster)
        self.dislocation_object.StartingPointCluster()
        print('... Starting ordering line ...')
        _ = self.dislocation_object.BuildOrderingLine(array_neigh_ext,scale_cluster,descriptor=None,idx_neighbor=index_neigh_ext)
        tmp_atoms = self.dislocation_object.LineSmoothing(nb_averaging_window=nb_averaging_window)
        self.dislocation_object.ComputeBurgerOnLineSmooth(rcut_burger, nye, descriptor=None)
        
        return idx_list, tmp_atoms

path_dislo = '../data/CONTCAR_p3p00GPa.lmp.gz'
dislocation_finder = ToyDislocationAnalyser(path_dislo)
dislocation_finder.extend_dislocation((1,1,1))

print('... Building outlier ...')
outliers_idx = dislocation_finder.get_outliers(rcut=3.2)

print('... Building extended neighborhood')
extended_outlier, full_outlier = dislocation_finder.build_extended_neigh_(outliers_idx, rcut_extended=5.5, rcut_full=8.0)
print('... Starting analysis ...')
sample_id, line_atoms = dislocation_finder.perform_dislocation_analysis(3.1882*np.eye(3),
                                                                        rcut_dislo=4.5,
                                                                        rcut_line=4.5,
                                                                        rcut_cluster=6.0,
                                                                        scale_cluster=2.5,
                                                                        nb_averaging_window=4,
                                                                        rcut_burger=5.0)

frame_object = FrameOvito([dislocation_finder.dislocation])
naive_modifier = NaiveOvitoModifier(dict_color={'firebrick':extended_outlier,
                                                'blue':outliers_idx,
                                                'royalblue':outliers_idx[sample_id]},
                                    dict_transparency={0.9:extended_outlier,
                                                        0.7:outliers_idx,  
                                                        0.0:outliers_idx[sample_id]},
                                    array_line=[ line_atom.positions for line_atom in line_atoms ])

pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
pipeline_config.modifiers.append(naive_modifier.BuildArrays)

for frame in range(pipeline_config.source.num_frames) :
    data = pipeline_config.compute(frame)
pipeline_config.add_to_scene()
