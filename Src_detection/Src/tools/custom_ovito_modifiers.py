import numpy as np

from ase import Atoms
from typing import List, Dict, TypedDict
from warnings import warn

import ovito
import matplotlib.pyplot as plt
from matplotlib import colors

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import PipelineSourceInterface
from ovito.io import *
from ovito.data import DataCollection
from ovito.traits import Color
from matplotlib.rcsetup import _validators
from ovito.vis import ViewportOverlayInterface
from traits.api import Bool, Code, Enum, Range

########################################################
## FRAME INTERFACE FOR OVITO
########################################################
class FrameOvito(PipelineSourceInterface) : 
    """Define the PipelineSourceInterface given to ovito pipeline to 
    plot frame by frame the configurations"""
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

#######################################################
# LATEX OVERLAY
#######################################################
class TeXLabel(TypedDict) :
    """Subclass for LateX labelling"""
    label : str
    x_offset : float
    y_offset : float
    color : np.ndarray

class LaTeXTextOverlay(ViewportOverlayInterface):
    """Init method for Latex labeling in Ovito Viewport 
    
    Parameters
    ----------

    dictionnary_label : Dict[int,TexLabel]
        Dictionnary containing all data to draw label in viewport
    """
    def __init__(self, dictionnary_label : Dict[int,TeXLabel]) -> None :
        self.dictionnary_label = dictionnary_label

    text = Code(value=r"\exp^{i \pi} + 1 = 0", label="Text")
    font = Enum(_validators["mathtext.fontset"].valid.values(), label="Font")
    fontsize = Range(low=1, high=None, value=50, label="Font size")
    if ovito.version >= (3, 10, 3):
        text_color = Color(default=(0.0, 0.0, 0.0), label="Text color")
    else:
        text_color = (0.0, 0.0, 0.0)
    px = Range(low=0.0, high=1.0, value=0.5, label="X position")
    py = Range(low=0.0, high=1.0, value=0.5, label="Y position")
    if ovito.version >= (3, 10, 3):
        show_background = Bool(False, label="Show background")
        background_color = Color(default=(1.0, 0.5, 0.5), label="Background color")

    def render(self, 
               canvas: ViewportOverlayInterface.Canvas, 
               data: DataCollection, 
               **kwargs) -> None :
        
        if ovito.version >= (3, 10, 3) and self.show_background:
            bbox = dict(
                boxstyle="round",
                ec=self.background_color,
                fc=self.background_color,
            )
        else:
            bbox = None

        with canvas.mpl_figure(pos=(self.px - 0.5, 0.5 + self.py),
                               size=(1.0, 1.0),
                               anchor="north west",
                               alpha=0,
                               tight_layout=True) as fig:
            ax = fig.subplots()

            identifier_data : int = data.particles_['LabelLatex'][:]
            label = self.dictionnary_label[identifier_data]['label']
            x_offset = self.dictionnary_label[identifier_data]['x_offset']
            y_offset = self.dictionnary_label[identifier_data]['y_offset']
            color = self.dictionnary_label[identifier_data]['color']

            if self.text:
                ax.text(
                    x_offset,
                    y_offset,
                    r"%s"%(label),
                    fontsize=self.fontsize,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    math_fontfamily=self.font,
                    bbox=bbox,
                )
            ax.axis("off")

class NaiveOvitoModifier : 
    """Naive ```OvitoModifier``` for debug ..."""
    def __init__(self, dict_transparency : Dict[float, List[int]] = None, 
                 dict_color : Dict[str, List[int]] = None, 
                 array_line : List[np.ndarray] = None,
                 array_caracter : List[np.ndarray] = None) -> None : 
        """Init method for ```NaiveOvitoModifier``` which just assign colors and transparency depending on ids given in dictionnaries ...
        
        Parameters
        ----------

        dict_transparency : Dict[float, List[int]]
            Dictionnary containing as key transparency value and associated idx for atoms system
        
        dict_color : Dict[str, List[int]]
            Dictionnary containing as key color value and associated idx for atoms system

        array_line : np.ndarray 
            Coordinates of line in cartesian space
            
        """
        if dict_transparency is None or dict_color is None : 
            warn('Some dictionnaries are set to None ...')

        self.dict_transparency = dict_transparency
        self.dict_color = dict_color
        self.array_line = array_line
        self.array_caracter = array_caracter

    def ASEArrayModifier(self, dict_transparency : Dict[float, float],
                               dict_color : Dict[float, str],
                               name_array : str) -> None : 
        """Variant of dictionnary based on ```ase``` array
        
        Parameters
        ----------

        dict_transparency : Dict[float, float]
            Transparency dictionnary, key are value is ```ase``` array

        dict_color : Dict[float, str]
            Color dictionnary, key are value is ```ase``` array

        name_array : str
            Name of array in ```ase``` object

        """
        self.dict_transparency_ase = dict_transparency
        self.dict_color_ase = dict_color 
        self.ase_array = name_array
        return 

    def BuildArrays(self, frame : int , data : DataCollection) -> None :
        """Build the whole ```NaiveOvitoModifier```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        """

        def builder(dict : Dict[str | float, List[int] ], data : DataCollection, type = 'color') -> np.ndarray : 
            """Little function to build transparencies / colors arrays
            
            Parameters
            ----------

            dict : Dict[str | float, List[int]]
                Colors / transparencies dictionnaries

            data : DataCollection
                System to visualise 

            Returns:
            --------

            np.ndarray 
                Colors / transparencies array associated to the system
            """
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
            if self.array_caracter is None : 
                Cmap = plt.get_cmap('autumn')
                colors_line = [Cmap(i) for i in np.linspace(0.0,1.0,num=len(self.array_line)) ]
                for id,line in enumerate(self.array_line) :
                    lines = data.lines.create(identifier=f'myline{id}', positions=line)
                    lines.vis.color = tuple( [ c for c in colors_line[id]][:-1] ) #colors.to_rgb('chartreuse')
                    lines.vis.width = 0.5

            else : 
                Cmap = plt.get_cmap('seismic')
                compt = 0 
                for id_l, line in enumerate(self.array_line) :
                    for id_p in range(line.shape[0] - 1) :               
                        lines = data.lines.create(identifier=f'myline{compt}', 
                                                  positions=line[id_p:id_p+2,:])
                        

                        compt+= 1
                        caracter = self.array_caracter[id_l][id_p][0]
                        lines.vis.color = tuple([ c for c in Cmap(caracter)[:-1] ])
                        lines.vis.width = 1.2


    def BuildArraysAse(self, frame : int , data : DataCollection) -> None :
        """Build the whole ```NaiveOvitoModifier```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        """

        array_ase = data.particles[self.ase_array][:]
        array_transparency = np.empty(array_ase.shape)
        array_color = np.empty((len(array_ase),3))

        for id_data_ase, data_ase in enumerate(array_ase) : 
            array_transparency[id_data_ase] = self.dict_transparency_ase[data_ase]
            array_color[id_data_ase] = colors.to_rgb(self.dict_color_ase[data_ase])


        data.particles_.create_property('Color',data=array_color)
        data.particles_.create_property('Transparency',data=array_transparency)
        

class MCDModifier :
    """```OvitoModifier``` function for visual MCD selection"""
    def __init__(self, init_transparency : float = 0.9, 
                 threshold_mcd : float = 0.5, 
                 color_map : str = 'viridis', 
                 dict_color : dict = None) -> None :
        """Init method for ```MCDModifier```
        
        Parameters
        ----------

        init_transparency : float 
            Default value for transparency 

        threshold_mcd : float 
            percentage of max mcd distance to be selected 

        color_map : str 
            Color map used to visualise mcd distances 
        
        dict_color : dict
            Species color dictionnary
        """
        self.init_transparency = init_transparency
        self.threshold_mcd = threshold_mcd
        self.color_map = plt.cm.get_cmap(color_map)
        self.dict_color = dict_color

        if self.dict_color is not None : 
            self._build_rgb_dict()

    def _build_rgb_dict(self) -> None : 
        """Convert the species color dictionnary into rgb"""
        self.dic_rgb = {}
        for key in self.dict_color.keys() :
            self.dic_rgb[key] = colors.to_rgb(self.dict_color[key])

    def AssignColors(self, frame : int, data : DataCollection) -> None :
        """Build ```Color``` property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 

        """
        particules_type = data.particles['Particle Type'][:]
        color_array = np.array([self.dic_rgb[type] for _,type in enumerate(particules_type)])
        data.particles_.create_property('Color',data=color_array)


    def PerformMCDVolumeSelection(self, frame : int, data : DataCollection) -> None :
        """Build mcd color map property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        
        """
        mcd_score = data.particles['mcd-distance'][:]
        atomic_volume = data.particles['atomic-volume'][:]

        mean_volume = np.mean(atomic_volume)
        max_mcd = np.amax(mcd_score)

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

class LogisticModifier :
    """```OvitoModifier``` function for visual logistic selection"""
    def __init__(self, init_transparency : float = 0.9, 
                 threshold : float = 0.5, 
                 dict_color : Dict[int,str] = None) -> None :
        """Init method for ```LogisticModifier```
        
        Parameters
        ----------

        init_transparency : float 
            Default value for transparency 

        threshold : float 
            threshold of logistic score to be visualise
        
        dict_color : dict
            Species color dictionnary
        """
        self.init_transparency = init_transparency
        self.threshold = threshold
        self.dict_color = dict_color
        
        if self.dict_color is not None : 
            self._build_rgb_dict()

    def _build_rgb_dict(self) -> None : 
        """Convert the species color dictionnary into rgb"""
        self.dic_rgb = {}
        for key in self.dict_color.keys() : 
            self.dic_rgb[key] = colors.to_rgb(self.dict_color[key])

    def AssignColors(self, frame : int, data : DataCollection) -> None : 
        """Build ```Color``` property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 

        """
        particules_type = data.particles['Particle Type'][:]
        color_array = np.array([self.dic_rgb[type] for _,type in enumerate(particules_type)])
        data.particles_.create_property('Color',data=color_array)

    def PerformLogisticSelection(self, frame : int, data : DataCollection) -> None : 
        """Build logisitic color property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        
        """
        logistic_score = data.particles['logistic-score'][:]
        array_transparency = np.full(logistic_score.shape[0], self.init_transparency)
        mask = (logistic_score[:,1] > self.threshold)
        array_transparency[mask] = 1.0 - logistic_score[mask,1] 
        data.particles_.create_property('Transparency',data=array_transparency)