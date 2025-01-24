import os, sys, shutil
import numpy as np
from ase import Atoms
from sklearn.covariance import MinCovDet
import pickle

import matplotlib.pyplot as plt
from matplotlib import colors

from ovito.io.ase import ase_to_ovito
from ovito.data import DataCollection 
from ovito.pipeline import PipelineSourceInterface
from ovito.pipeline import Pipeline, PythonSource

from typing import List, Dict, TypedDict


from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from ovito.vis import Viewport, TachyonRenderer, PythonViewportOverlay, CoordinateTripodOverlay

sys.path.append(os.getcwd())
from latex_overlay import LaTeXTextOverlay

class MCDAnalysis : 
    def __init__(self, path_desc_bulk : os.PathLike[str],
                 path_writing : os.PathLike[str],
                 contamination : float = 0.05) -> None : 
        
        self.path_desc_bulk = path_desc_bulk
        self.path_writing = path_writing
        self.contamination = contamination

    def read_milady_descriptor(self, filename : str,
                               ext : str = 'eml') -> np.ndarray :
        if ext == 'eml' :
            return np.loadtxt(filename)[:,1:]


    
    def FillThermicData(self) -> np.ndarray : 
        array_desc = None 
        for desc_file in [f'{self.path_desc_bulk}/{f}' for f in os.listdir(self.path_desc_bulk)] :
            if array_desc is None :
                array_desc = self.read_milady_descriptor(desc_file, ext='eml')
            else : 
                array_desc = np.concatenate((array_desc, self.read_milady_descriptor(desc_file, ext='eml')), axis=0)
        
        return array_desc
    
    def MCDModel(self, array_desc : np.ndarray) -> None :
        mcd = MinCovDet(support_fraction=1.0-self.contamination)
        mcd.fit(array_desc)
        pickle.dump(mcd, open(f'{self.path_writing}/mcd_thermic.pickle','wb'))

        return 

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


class MCDModifier :
    """```OvitoModifier``` function for visual MCD selection"""
    def __init__(self, init_transparency : float = 0.9, 
                 threshold_mcd : float = 0.5, 
                 color_map : str = 'viridis', 
                 dict_color : dict = None,
                 raduis_at : float = 0.9) -> None :
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
        self.raduis_at = raduis_at

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


    def PerformMCDVolumeSelectionDraft(self, frame : int, data : DataCollection) -> None :
        """Build mcd color map property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        
        """
        mcd_score = data.particles['mcd-distance'][:]

        mask = (mcd_score > self.threshold_mcd)

        # Initialize color_array and array_transparency
        color_array = np.empty((len(mcd_score), 3))
        array_transparency = np.full(mcd_score.shape[0], self.init_transparency)

        # Apply conditions using boolean indexing
        array_transparency[mask] = self.init_transparency #0.4 * (1.0 - normalized_mcd[mask])
        color_array[mask] = [colors.to_rgb(self.color_map(mcd)) for mcd in mcd_score[mask]]

        # Set default values for elements not meeting the condition
        color_array[~mask] = colors.to_rgb('grey')
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)


    def PerformMCDVolumeSelection(self, frame : int, data : DataCollection) -> None :
        """Build mcd color map property for ```Ovito```
        
        frame : int 
            index of frame (needed by ```Ovito```)

        data : DataCollection 
            System to visualise 
        
        """
        mcd_score = data.particles['mcd-distance'][:]

        max_mcd = np.amax(mcd_score)

        normalized_mcd = mcd_score / max_mcd
        mask = (normalized_mcd > self.threshold_mcd)

        # Initialize color_array and array_transparency
        color_array = np.empty((len(mcd_score), 3))
        array_transparency = np.full(mcd_score.shape[0], self.init_transparency)

        # Apply conditions using boolean indexing
        array_transparency[mask] = 0.4 * (1.0 - normalized_mcd[mask])
        color_array[mask] = [colors.to_rgb(self.color_map(mcd)) for mcd in normalized_mcd[mask]]

        # Set default values for elements not meeting the condition
        color_array[~mask] = colors.to_rgb('grey')
        data.particles_.create_property('Transparency',data=array_transparency)
        data.particles_.create_property('Color',data=color_array)
        data.particles_.create_property('Radius',data=self.raduis_at*np.ones(mcd_score.shape))

        data.cell_.vis.line_width = 0.0

class Data(TypedDict) :
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    array_delta_FE : np.ndarray
    atoms : Atoms
    stress : np.ndarray
    volume : float
    energy : float
    Ff : np.ndarray

class ComputeMCDModifier : 
    def __init__(self, 
                 path_pkl_config : os.PathLike[str],
                 path_pkl_mcd : os.PathLike[str],
                 list2plot : List[str] = None) -> None : 
        
        self.data_obj : Dict[str,Data] = pickle.load(open(path_pkl_config,'rb'))
        self.model_mcd : MinCovDet = pickle.load(open(path_pkl_mcd,'rb'))
        self.list2plot = list2plot

    def ComputeMCDDistances(self) -> List[Atoms] :
        list_atoms = []
        for key, val in self.data_obj.items() : 
            atoms_obj = val['atoms']
            mcd_distances = self.model_mcd.mahalanobis(atoms_obj.get_array('milady-descriptors'))
            atoms_obj.set_array('mcd-distance', mcd_distances, dtype=float)
            
            if self.list2plot is not None : 
                if key in self.list2plot : 
                    list_atoms.append(atoms_obj.copy())
                    print(key)
                else :
                    continue
            
            else :
                list_atoms.append(atoms_obj.copy())
        
        return list_atoms
    
    def BuildOvitoPipeline(self) -> None : 
        list_atoms = self.ComputeMCDDistances()
        frame_obj = FrameOvito(list_atoms)
        mcd_modifier = MCDModifier(init_transparency=0.98,
                                   threshold_mcd=0.2,
                                   color_map='viridis')

        pipeline_config = Pipeline(source = PythonSource(delegate=frame_obj))
        pipeline_config.modifiers.append(mcd_modifier.PerformMCDVolumeSelection)
        for frame in range(pipeline_config.source.num_frames) :
            data = pipeline_config.compute(frame)

        pipeline_config.add_to_scene()
        if not os.path.exists(f'{os.getcwd()}/images_ovito') : 
            os.mkdir('images_ovito')
        else : 
            shutil.rmtree(f'{os.getcwd()}/images_ovito')
            os.mkdir('images_ovito')
            
 
        viewport = Viewport(type = Viewport.Type.Perspective, camera_dir=(-1,-1,-0.7))
        #viewport.overlays.append(PythonViewportOverlay(delegate=CoordinationPlotOverlay()))
        viewport.zoom_all(size=(800, 800))
        tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1,
                                  antialiasing=False)

        
        viewport.overlays.append(PythonViewportOverlay(delegate=LaTeXTextOverlay()))
        tripod = CoordinateTripodOverlay(font='Arial',
                                         axis1_color=(0,0,0),axis1_label='[100]',
                                         axis2_color=(0,0,0),axis2_label='[010]',
                                         axis3_color=(0,0,0),axis3_label='[001]',
                                         offset_x=0.1,
                                         offset_y=0.05)
        """
        viewport.overlays.append(tripod)

        for id_f in range(pipeline_config.source.num_frames) : 
            viewport.render_image(size=(1000,1000), 
                            filename=f"images_ovito/figure_{id_f}.png", 
                            background=(1,1,1), 
                            frame=id_f,
                            renderer=tachyon)"""

##########################
#### INPUTS
##########################
path_desc = '/home/lapointe/ToSave/Work_ML/Free_energy/outlier_I4/mcd_bulk/descDB'
path_pkl_mcd = '/home/lapointe/WorkML/FreeEnergySurrogate/Src/mcd_analysis'
path_pkl_config = '/home/lapointe/WorkML/FreeEnergySurrogate/data/mab_desc_j4_r6.pickle'
mode = 'ovito'
list2plot = ['10306', '10233', '10278', '10079', '11048', '11203', '10529', '10307', '10954', '11222', '10756', '10041', '10149', '10077', '10177', '10557', '11032', '10160', '10448', '10975', '10347', '10452', '10236', '10145', '10313', '11250']
##########################

if mode == 'mcd' : 
    obj_mcd = MCDAnalysis(path_desc,
                          path_pkl_mcd)
    design_array =  obj_mcd.FillThermicData()
    print(f'... Design array is filled : {design_array.shape}')
    obj_mcd.MCDModel(design_array)
    print('... MCD object has been filled')

if mode == 'ovito' : 
    obj_ovito = ComputeMCDModifier(path_pkl_config,
                                   f'{path_pkl_mcd}/mcd_thermic.pickle',
                                   list2plot=None)
    obj_ovito.BuildOvitoPipeline()