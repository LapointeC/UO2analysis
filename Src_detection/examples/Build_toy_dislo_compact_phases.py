import sys, os
import numpy as np

#Line needed to work with ovito graphical mode
sys.path.append(os.getcwd())

sys.path.insert(0,'../')
from Src import InputsDictDislocationBuilder, DislocationsBuilder, \
                  InputsCompactPhases, C15Builder, \
                  FrameOvito, NaiveOvitoModifier


from ovito.pipeline import Pipeline, PythonSource

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

####################################
### INPUTS DISLOCATION !
####################################
inputs_dict : InputsDictDislocationBuilder = {'structure':'BCC',
               'a0':2.8853,
               'size_loop':25.0,
               'scale_loop':3.0,
               'orientation_loop':np.array([0.0,0.0,1.0]),
               'element':'Fe'}
#####################################

###############################
##### INPUTS C15 !
###############################
dic_param : InputsCompactPhases = {'a0':2.853,
             'element':'Fe',
             'scale_factor':1.2}
path_xml = '../data/c15.xml'
###############################


print('... Building dislocation')
dislo_obj = DislocationsBuilder(inputs_dict)
dislo_obj.BuildDislocation(ovito_mode=True)
print()
print('... Building C15 ...')
C15obj = C15Builder(dic_param, path_inputs=path_xml)
C15obj.BuildC15Cluster(ovito_mode=True)

frame_object = FrameOvito([dislo_obj.ase_system, C15obj.C15_system])
naive_modifier = NaiveOvitoModifier()
naive_modifier.ASEArrayModifier({0.0:0.95,
                                 1.0:0.1},
                                 {0.0:'grey',
                                  1.0:'chartreuse'},
                                  'defect')

pipeline_config = Pipeline(source = PythonSource(delegate=frame_object))
pipeline_config.modifiers.append(naive_modifier.BuildArraysAse)

for frame in range(pipeline_config.source.num_frames) :
    data = pipeline_config.compute(frame)
pipeline_config.add_to_scene()