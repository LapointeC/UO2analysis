import os, sys 
import pickle
import shutil

sys.path.append(os.getcwd())

from ase import Atoms

sys.path.insert(0,'../')
from ..Src import DBManager, \
                  DfctAnalysisObject, \
                  my_cfg_reader, \
                  FrameOvito, LogisticModifier

from ovito.pipeline import Pipeline, PythonSource

from typing import List
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

# Create a global Qt application object - unless we are running inside the 'ovitos' interpreter,
# which automatically initializes a Qt application object.
if not QApplication.instance():
    app = QApplication(sys.argv)

########################################################
### INPUTS
########################################################
pickle_data_file = 'data_I2_min.pickle'
pickle_logistic = 'logistic.pickle'
########################################################


print('... Starting from the previous configuration file ...')
previous_dbmodel : DBManager = pickle.load(open(pickle_data_file,'rb'))
print('... DBmanager data is stored ...')
print()
print('... Loading logistic classifier models ...')
logistic_model : DfctAnalysisObject = pickle.load(open(pickle_logistic,'rb'))
print(logistic_model.meta_data_model)
print([key for key in logistic_model.logistic_model.keys()])
print([logistic_model.logistic_model[key].coef_ for key in logistic_model.logistic_model.keys()])
print('... Logistic models are stored ...')

if not os.path.exists('{:s}/ovito_min'.format(os.getcwd())) :
    os.mkdir('{:s}/ovito_min'.format(os.getcwd()))
else : 
    shutil.rmtree('{:s}/ovito_min'.format(os.getcwd()))
    os.mkdir('{:s}/ovito_min'.format(os.getcwd()))


for id,key in enumerate(previous_dbmodel.model_init_dic.keys()) :
    print('... Analysis for : {:s} ...'.format(key))
    atom_config = logistic_model.one_the_fly_logistic_analysis(previous_dbmodel.model_init_dic[key]['atoms'])

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