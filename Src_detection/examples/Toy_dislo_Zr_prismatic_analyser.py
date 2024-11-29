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
    def __init__(self, path_dislo : os.PathLike[str], data_type : str = 'lammps-data', structure : str = 'hcp', avoid_structure : List[str] = ['fcc']) -> None :
        self.dislocation = read(path_dislo, format=data_type)

        self.inv_dict_struct = {'other':0,
                                'fcc':1,
                                'hcp':2,
                                'bcc':3,
                                'ico':4}
        
        self.target_struct_str = structure
        self.avoid_structure = avoid_structure
        self.target_struct = self.inv_dict_struct[structure]
        self.dislocation_object : DislocationObject = None

    def extend_dislocation(self, tuple_repeat = (1,1,1)) -> None : 
        self.dislocation *= tuple_repeat

    def get_outliers(self, nb_std : float = 0.5, rcut : float = 3.5) -> List[int] : 
        data_ovito = ase_to_ovito(self.dislocation)
        pipeline = Pipeline(source = StaticSource(data = data_ovito))
        voro = VoronoiAnalysisModifier(
            compute_indices = True,
            use_radii = False,
            edge_threshold = 0.1)
        cna = CommonNeighborAnalysisModifier(cutoff=rcut)

        pipeline.modifiers.append(voro)
        pipeline.modifiers.append(cna)
        data = pipeline.compute()
        
        structure_type = data.particles['Structure Type'][:]
        volume = data.particles['Atomic Volume'][:]
        id_atoms = np.array([k for k in range(len(structure_type))])

        mean_volume, std_volume = np.mean(volume), np.std(volume)

        #select only atoms identified as different from structure
        volume_bar = volume.copy()
        id_atoms_bar = id_atoms.copy()
        for strct in [self.target_struct] + self.avoid_structure : 
            mask_struct = structure_type != strct
            volume_bar = volume_bar[mask_struct]
            id_atoms_bar = id_atoms_bar[mask_struct]
            structure_type = structure_type[mask_struct]


        # selecty only atoms with low volume (to eliminate free surface)
        mask_volume = volume_bar < mean_volume + nb_std*std_volume
        return id_atoms_bar[mask_volume].tolist()

    @timeit
    def build_extended_neigh(self, list_idx: List[int], rcut_extended: float = 4.5, rcut_full: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
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

        #organise lists for Nye tensor
        ext_list = list_idx + [el for el in ext_list if el not in list_idx]
        full_list = ext_list + [el for el in full_list if el not in ext_list]

        # Update the local, extended, and full dislocation data
        self.local_dislocation = self.dislocation.copy()[list_idx]
        self.extended_dislocation = self.dislocation.copy()[ext_list]
        self.full_dislocation = self.dislocation.copy()[full_list]

        return np.array(ext_list), np.array(full_list)


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
                                                               'unit_cell':unit_cell})
        
        nye, array_neigh, index_neigh, array_neigh_ext, index_neigh_ext =self.dislocation_object.NyeTensor()
        idx_list = self.dislocation_object.BuildSamplingLine(rcut_line=rcut_line, rcut_cluster=rcut_cluster, scale_cluster=scale_cluster)
        self.dislocation_object.RefineSamplingLine(scale=scale_cluster)
        self.dislocation_object.StartingPointCluster()
        _ = self.dislocation_object.BuildOrderingLine(array_neigh_ext,scale_cluster,descriptor=None,idx_neighbor=index_neigh_ext)
        tmp_atoms = self.dislocation_object.LineSmoothing(nb_averaging_window=nb_averaging_window)
        self.dislocation_object.ComputeBurgerOnLineSmooth(rcut_burger, nye, descriptor=None)
        
        return idx_list, tmp_atoms

##############################
### INPUTS
##############################
path_dislo = '../data/zr_loop.79_n6_I1.cfg'
#path_dislo = '/home/lapointe/DisloEmmanuel/small_tests/zr_loop.305_n6_BB.cfg'
cell = 3.234*np.array([[1.0, 0.0, 0.0],
                       [-0.5, 0.8660254037844386468, 0.0],
                       [0.0, 0.0, 1.598021026592455164]])
##############################


dislocation_finder = ToyDislocationAnalyser(path_dislo, data_type='cfg')
dislocation_finder.extend_dislocation((1,1,1))
outliers_idx = dislocation_finder.get_outliers(nb_std=1.0,
                                rcut=3.5)
extended_outlier, full_outlier = dislocation_finder.build_extended_neigh(outliers_idx, rcut_extended=6.5, rcut_full=8.0)
sample_id, line_atoms = dislocation_finder.perform_dislocation_analysis(cell,
                                                                        rcut_dislo=6.0,
                                                                        rcut_line=3.5,
                                                                        rcut_cluster=5.5,
                                                                        scale_cluster=3.0,
                                                                        nb_averaging_window=4,
                                                                        rcut_burger=6.5)
# put into np.array for visualisation
outliers_idx = np.array(outliers_idx)

print(line_atoms)
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
