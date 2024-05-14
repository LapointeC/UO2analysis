import os, sys 
import numpy as np
import pickle
import shutil

sys.path.append(os.getcwd())
from matplotlib import colors

from ase import Atoms
from milady import * 
from create_inputs import *
from milady_writer import *
from my_cfg_reader import my_cfg_reader

from library_mcd import Dfct_analysis_object, custom_writer

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, PythonSource, PipelineSourceInterface
from ovito.io import *
from ovito.data import DataCollection

from ovito.modifiers import ExpressionSelectionModifier

from typing import List, Dict
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from ovito.vis import Viewport
# Create a global Qt application object - unless we are running inside the 'ovitos' interpreter,
# which automatically initializes a Qt application object.
if not QApplication.instance():
    app = QApplication(sys.argv)

########################################################
### INPUTS
########################################################
pickle_data_file = 'data_dfct_min.pickle'
pickle_logistic = 'logistic.pickle'
########################################################


print('... Starting from the previous configuration file ...')
previous_dbmodel : DBManager = pickle.load(open(pickle_data_file,'rb'))
print('... DBmanager data is stored ...')
print()
print('... Loading logistic classifier models ...')
logistic_model : Dfct_analysis_object = pickle.load(open(pickle_logistic,'rb'))
print('... Logistic models are stored ...')

if not os.path.exists('{:s}/ovito_min'.format(os.getcwd())) :
    os.mkdir('{:s}/ovito_min'.format(os.getcwd()))
else : 
    shutil.rmtree('{:s}/ovito_min'.format(os.getcwd()))
    os.mkdir('{:s}/ovito_min'.format(os.getcwd()))


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

class LogisticModifier :
    """Ovito modifier function for visual logistic selection"""
    def __init__(self, init_transparency : float = 0.9, threshold : float = 0.5, dict_color : Dict[int,str] = None) :
        self.init_transparency = init_transparency
        self.threshold = threshold
        self.dict_color = dict_color
        
        if self.dict_color is not None : 
            self._build_rgb_dict()

    def _build_rgb_dict(self) -> None : 
        self.dic_rgb = {}
        for key in self.dict_color.keys() : 
            self.dic_rgb[key] = colors.to_rgb(self.dict_color[key])

    def AssignColors(self, frame : int, data : DataCollection) : 
        particules_type = data.particles['Particle Type'][:]
        color_array = np.empty((len(particules_type),3))
        for id ,type in enumerate(particules_type) : 
            color_array[id, :] = self.dic_rgb[type] 
        data.particles_.create_property('Color',data=color_array)

    def PerformLogisticSelection(self, frame : int, data : DataCollection) : 
        logistic_score = data.particles['logistic-score'][:]
        array_transparency = np.empty((logistic_score.shape[0],))
        for id ,logi_score in enumerate(logistic_score) : 
            if logi_score[1] > self.threshold : 
                array_transparency[id] = (1.0 - logi_score[1])
            else : 
                array_transparency[id] = self.init_transparency 
        data.particles_.create_property('Transparency',data=array_transparency)


for id,key in enumerate(previous_dbmodel.model_init_dic.keys()) :
    print('... Analysis for : {:s} ...'.format(key))
    atom_config = logistic_model.one_the_fly_logistic_analysis(previous_dbmodel.model_init_dic[key]['atoms'])
    custom_writer(atom_config,
              '{:s}/ovito_min/{:s}.dump'.format(os.getcwd(),key),
              property='logistic-score',
              class_to_plot=1)

    if id == 0 :
        atoms_config : List[Atoms] = [atom_config]
    else : 
        atoms_config.append(atom_config)

frame_object = FrameOvito(atoms_config)
logistic_modifier = LogisticModifier(threshold=0.7,
                                     init_transparency=0.9,
                                     dict_color={1:'lightcoral',2:'chartreuse'})

pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
pipeline_config.modifiers.append(logistic_modifier.PerformLogisticSelection)
pipeline_config.modifiers.append(logistic_modifier.AssignColors)
for frame in range(pipeline_config.source.num_frames) : 
    data = pipeline_config.compute(frame)

pipeline_config.add_to_scene()

