import numpy as np

from ase import Atoms
from typing import List

import matplotlib.pyplot as plt
from matplotlib import colors

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import PipelineSourceInterface
from ovito.io import *
from ovito.data import DataCollection, Lines

########################################################
## FRAME INTERFACE FOR OVITO
########################################################
class FrameOvito(PipelineSourceInterface) : 
    """Define the PipelineSourceInterface given to ovito pipeline to 
    plot frame by frame the confiugrations"""
    def __init__(self, list_atoms : List[Atoms]) :
        self.list_atoms = list_atoms
        self.lenght = len(list_atoms)
        
    def compute_trajectory_length(self, **kwargs):
        """Needed method for pipeline"""
        return self.lenght
            
    def create(self, data : DataCollection, *, frame : int, **kwargs) :
        """Creator for DataCollection object inside the ovito pipeline"""
        data_atoms : DataCollection = ase_to_ovito(self.list_atoms[frame])
        data.particles = data_atoms.particles
        data.cell = data_atoms.cell

class NaiveOvitoModifier : 
    """Naive ```OvitoModifier``` for debug ..."""
    def __init__(self, dict_transparency : dict = None, dict_color : dict = None, array_line : np.ndarray = None) -> None : 
        if dict_transparency is None or dict_color is None : 
            raise TimeoutError('Some dictionnaries are set to None ...')

        self.dict_transparency = dict_transparency
        self.dict_color = dict_color
        self.array_line = array_line

    def _build_rgb_dict(self) -> None : 
        self.dic_rgb = {}
        for key in self.dict_color.keys() :
            self.dic_rgb[key] = colors.to_rgb(self.dict_color[key])

    def BuildArrays(self, frame : int , data : DataCollection) -> None : 
        def builder(dict : dict, data : DataCollection, type = 'color') -> np.ndarray : 
            if type == 'color' : 
                init_array = np.ones((len(data.particles['Particle Type'][:]),3))
            else :
                init_array = np.full(len(data.particles['Particle Type'][:]),1.0)
            
            for key, val in dict.items() : 
                if type == 'color' : 
                    init_array[val] = colors.to_rgb(key)
                else : 
                    init_array[val] = key

            return init_array

        self.array_color = builder(self.dict_color, data, type ='color')
        self.array_transparency = builder(self.dict_transparency, data, type ='transparency')

        data.particles_.create_property('Color',data=self.array_color)
        data.particles_.create_property('Transparency',data=self.array_transparency)
        
        if self.array_line is not None : 
            lines = data.lines.create(identifier='myline', positions=self.array_line)
            lines.vis.color = colors.to_rgb('chartreuse')
            lines.vis.width = 0.5 #(color=colors.to_rgb('chartreuse'), width=1.0)
            #data.lines_.create_property('Color',data=self.array_color) 

class MCDModifier :
    """Ovito modifier function for visual logistic selection"""
    def __init__(self, init_transparency : float = 0.9, threshold_mcd : float = 0.5, color_map : str = 'viridis', dict_color : dict = None) :
        self.init_transparency = init_transparency
        self.threshold_mcd = threshold_mcd
        self.color_map = plt.cm.get_cmap(color_map)
        self.dict_color = dict_color

        if self.dict_color is not None : 
            self._build_rgb_dict()

    def _build_rgb_dict(self) -> None : 
        self.dic_rgb = {}
        for key in self.dict_color.keys() :
            self.dic_rgb[key] = colors.to_rgb(self.dict_color[key])

    def AssignColors(self, frame : int, data : DataCollection) :
        particules_type = data.particles['Particle Type'][:]
        color_array = np.array([self.dic_rgb[type] for _,type in enumerate(particules_type)])
        data.particles_.create_property('Color',data=color_array)


    def PerformMCDVolumeSelection(self, frame : int, data : DataCollection) :
        mcd_score = data.particles['mcd-distance'][:]
        atomic_volume = data.particles['atomic-volume'][:]

        mean_volume = np.mean(atomic_volume)
        max_mcd = np.amax(mcd_score)

        #color_array = np.empty((len(mcd_score),3))
        #array_transparency = np.empty((mcd_score.shape[0],))
        normalized_mcd = mcd_score / max_mcd
        mask = (normalized_mcd > self.threshold_mcd) & (atomic_volume < mean_volume)

        # Initialize color_array and array_transparency
        color_array = np.empty((len(mcd_score), 3))
        array_transparency = np.full(mcd_score.shape[0], self.init_transparency)

        # Apply conditions using boolean indexing
        array_transparency[mask] = 0.2 * (1.0 - normalized_mcd[mask])
        color_array[mask] = [colors.to_rgb(self.color_map(mcd)) for mcd in normalized_mcd[mask]]

        # Set default values for elements not meeting the condition
        color_array[~mask] = colors.to_rgb('grey')
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)

    def PerformMCDSelection(self, frame : int, data : DataCollection) :
        mcd_score = data.particles['mcd-distance'][:]
        max_mcd = np.amax(mcd_score)
        
        normalized_mcd = mcd_score / max_mcd
        mask = (normalized_mcd > self.threshold_mcd)

        # Initialize color_array and array_transparency
        color_array = np.empty((len(mcd_score), 3))
        array_transparency = np.full(mcd_score.shape[0], self.init_transparency)

        # Apply conditions using boolean indexing
        array_transparency[mask] = 0.2 * (1.0 - normalized_mcd[mask])
        color_array[mask] = [colors.to_rgb(self.color_map(mcd)) for mcd in normalized_mcd[mask]]
        color_array[~mask] = colors.to_rgb('grey')
        
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)
