import os, sys 
import numpy as np
import pickle
import shutil

sys.path.append(os.getcwd())

from ase import Atoms
from milady import * 
from create_inputs import *
from milady_writer import *
from my_cfg_reader import my_cfg_reader

from library_mcd import Dfct_analysis_object, custom_writer

from ovito.io.ase import ase_to_ovito
from ovito.pipeline import StaticSource, Pipeline, PythonSource, PipelineSourceInterface
from ovito.io import *
from ovito.data import DataCollection

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
    def __init__(self, list_atoms : List[Atoms]) : 
        self.list_atoms = list_atoms
        self.lenght = len(list_atoms)

    def compute_trajectory_length(self, **kwargs):
        return self.lenght
    
    def create(self, data : DataCollection, *, frame : int, **kwargs) : 
        data_atoms : DataCollection = ase_to_ovito(self.list_atoms[frame])
        data.particles = data_atoms.particles
        data.cell = data_atoms.cell

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
pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
for frame in range(pipeline_config.source.num_frames) : 
    data = pipeline_config.compute(frame)
    print(data.particles)

pipeline_config.add_to_scene()

