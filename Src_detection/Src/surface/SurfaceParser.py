import os
import xml.etree.ElementTree as ET

from xml.etree.ElementTree import Element
from typing import Dict
from ase.dft.kpoints import monkhorst_pack

class SurfaceParser : 
    def __init__(self, path_inputs : os.PathLike[str], mode : str) -> None : 
        self.parameters = {}
        
        self.init_params()
        self.parser_inputs_file(path_inputs)
        if not self.check_minimal_arg(mode) : 
            exit(0)

    def init_params(self) : 
        """Initialise the input dictionnary !"""
        
        """Geometry paths"""
        self.parameters['PathCif'] = 'Default'
        self.parameters['PathBulk'] = 'Default'
        self.parameters['PathSlabs'] = 'Default'

        """Orientation of slab for geometry"""
        self.parameters['Orientation'] = [0,0,1]

        """Cluster option ! """
        self.parameters['Cluster'] = 'Default'
        self.parameters['SlurmFile'] = 'Default'
        self.parameters['SlurmCommand'] = 'sbatch '
        self.parameters['NProcs'] = 10

        """VASP setup"""
        self.parameters['PathComput'] = 'Default'
        self.parameters['InputsVASP'] = 'Default'
        self.parameters['CalculationSpeed'] = 'normal'
        self.parameters['ReciprocalDensity'] = 1.0

        """Extraction settings"""
        self.parameters['ElToRemove'] = 'Default'
        self.parameters['HyperplaneData'] = 'Default'
        self.parameters['WritingSlabs'] = '%s/Slabs'%(os.getcwd())
        self.parameters['LevelLines'] = False
        self.parameters['MuDiscretisation'] = 100
        self.parameters['SlabsToPlot'] = ['Default']

        """Slab builder settings"""
        self.parameters['hklVectors'] = [[1,0,0]]
        self.parameters['PymatgenTolerance'] = 1e-6
        self.parameters['Vaccum'] = 15.0
        self.parameters['HeighPymatgen'] = 15.0
        self.parameters['Levels'] = 2
        self.parameters['HeighConstraint'] = 10.0
        self.parameters['ToleranceHeigh'] = 2.0
        self.parameters['ToleranceAtomSurface'] = 1e-2
        self.parameters['AlphaParameters'] = [1.0]
        self.parameters['ToleranceDescritpors'] = 1e-2

        """Optional parameters"""
        self.parameters['CheckSym'] = {'bool':False, 'replica': 1, 'rcut': 3.0}
        self.parameters['CheckDense'] = {'bool': False, 'tolerance': None, 'compacity': None}
        self.parameters['RemeshSurface'] = {'bool': False, 'rcut': 3.0, 'beta': 1.0} 

    def parser_inputs_file(self, path_inputs_xml : str) :
        """Fill parameters dictionnary for ART based on xml input file 
        
        Parameters 
        ----------

        path_inputs_xml : str
            path of xml input file 
        """

        def boolean_converter(str_xml : str) -> bool : 
            """Convert string chain into boolean 
            
            Parameters 
            ----------

            str_xml : str
                string to convert 

            Returns 
            -------
            
            bool
            """
            if str_xml in ['true' ,'True', '.TRUE.','.True.', 'Yes'] : 
                return True
            if str_xml in ['false', 'False', '.FALSE.', '.False.' ,'No'] : 
                return False

        def extra_argument_dict(xml : Element) -> None :
            """Fill extra arguments which need a dictionnary
            
            Parameters
            ----------

            xml : Element
                Xml element associated to the extra argument
            
            """
            for child in xml : 
                type_child = self.parameters[xml.tag][child.tag]
                if isinstance(type_child,bool):
                    self.parameters[xml.tag][child.tag] = boolean_converter(child.text)
                elif isinstance(type_child,int):
                    self.parameters[xml.tag][child.tag] = int(child.text)
                elif isinstance(type_child,float):
                    self.parameters[xml.tag][child.tag] = float(child.text)
                elif isinstance(type_child,str):
                    self.parameters[xml.tag][child.tag] = child.text

        xml_root = ET.parse(path_inputs_xml)
        for xml_param in xml_root.getroot() : 
            type = self.parameters[xml_param.tag]
            if xml_param.tag == 'hklVectors':
                self.parameters['hklVectors'] = [] 
                for hkl in xml_param : 
                    self.parameters['hklVectors'].append( [ float(el) for el in hkl.text.split() ] )

            elif xml_param.tag == 'AlphaParameters' :
                self.parameters['AlphaParameters'] = []
                for alpha in xml_param : 
                    self.parameters['AlphaParameters'].append(float(alpha.text))

            elif xml_param.tag == 'SlabsToPlot' : 
                self.parameters['SlabsToPlot'] = []
                for name_slab in xml_param : 
                    self.parameters['SlabsToPlot'].append(name_slab.text)
               
            elif xml_param.tag == 'Orientation' : 
                self.parameters[xml_param.tag] = [int(el) for el in xml_param.text]
            
            else :
                if isinstance(type,bool):
                    self.parameters[xml_param.tag] = boolean_converter(xml_param.text)
                elif isinstance(type,int):
                    self.parameters[xml_param.tag] = int(xml_param.text)
                elif isinstance(type,float):
                    self.parameters[xml_param.tag] = float(xml_param.text)
                elif isinstance(type,str):
                    self.parameters[xml_param.tag] = xml_param.text
                elif isinstance(type,dict):
                    extra_argument_dict(xml_param)

    def check_minimal_arg(self, mode : str) -> bool :
        """Check if the minimal arguments for Surface builder are defined ...
        
        Returns
        -------

        bool 
            Return False if there is problem with minimal argument, True otherwise
        """
        bool_min_arg = True

        dic_slab_builder = {'PathBulk':lambda path : os.path.exists(path),
                        'hklVectors': lambda list : len(list) > 1}
        
        dic_slab_energy_launcher = {'PathCif':lambda path : os.path.exists(path),
                                    'PathSlabs':lambda path : os.path.exists(path),
                                    'PathComput':lambda path : os.path.exists(path),
                                    'InputsVASP':lambda path : os.path.exists(path),
                                    'Cluster' : lambda el : el != 'Default', 
                                    'SlurmFile' : lambda el : el != 'Default'}
        
        dic_just_check = {'PathComput':lambda path : os.path.exists(path)}
        
        dic_check_and_relaunch = {'PathComput':lambda path : os.path.exists(path),
                                  'Cluster' : lambda el : el != 'Default',
                                  'SlurmFile' : lambda el : el != 'Default'}
        
        dic_extract_surface_energy = {'PathComput':lambda path : os.path.exists(path),
                                    'ElToRemove':lambda el : el != 'Default'}
        
        dic_plot_and_stability = {'HyperplaneData':lambda path : os.path.exists(path),
                                'PathSlabs':lambda path : os.path.exists(path),
                                'ElToRemove':lambda el : el != 'Default',
                                'SlabsToPlot':lambda list : list[0] != 'Default'}

        general_dict : Dict[str,Dict] = {'slab_builder':dic_slab_builder,
                        'slab_energy_launcher':dic_slab_energy_launcher,
                        'just_check':dic_just_check,
                        'check_and_relaunch':dic_check_and_relaunch,
                        'extract_surface_energy':dic_extract_surface_energy,
                        'plot_and_stability':dic_plot_and_stability}

        for key in self.parameters : 
                if key in general_dict[mode].keys() : 
                    if not general_dict[mode][key](self.parameters[key]) : 
                        print('I have some trouble with %s argument ...'%(key))
                        bool_min_arg = False
                else : 
                    continue
        
        return bool_min_arg
    
class AseVaspParser : 
    def __init__(self, path_inputs : os.PathLike[str]) -> None : 
        self.parameters = {}
        
        self.init_params()
        self.parser_inputs_file(path_inputs)

    def init_params(self) : 
        """Initialise the input dictionnary !"""
        
        self.parameters['xc'] = 'pbe'
        self.parameters['algo'] = 'Fast'
        self.parameters['isym'] = 0
        self.parameters['ediff'] = 1e-6
        self.parameters['ediffg'] = -1e-2
        self.parameters['ismear'] = 2
        self.parameters['sigma'] = 0.1
        self.parameters['prec'] = 'Normal'
        self.parameters['lreal'] = 'Auto'
        self.parameters['ispin'] = 2
        self.parameters['ibrion'] = 2
        self.parameters['isif'] = 2
        self.parameters['nsw'] = 0
        self.parameters['encut'] = 300.0
        self.parameters['ncore'] = 20
        self.parameters['nelm'] = 70
        self.parameters['lcharg'] = False
        self.parameters['lwave'] = False
        self.parameters['kpoints_grid'] = [1,1,1]
        self.parameters['restart'] = False

    def parser_inputs_file(self, path_inputs_xml : str) :
        """Fill parameters dictionnary for ART based on xml input file 
        
        Parameters 
        ----------

        path_inputs_xml : str
            path of xml input file 
        """

        def boolean_converter(str_xml : str) -> bool : 
            """Convert string chain into boolean 
            
            Parameters 
            ----------

            str_xml : str
                string to convert 

            Returns 
            -------
            
            bool
            """
            if str_xml in ['true' ,'True', '.TRUE.','.True.', 'Yes'] : 
                return True
            if str_xml in ['false', 'False', '.FALSE.', '.False.' ,'No'] : 
                return False

        xml_root = ET.parse(path_inputs_xml)
        for xml_param in xml_root.getroot() : 
            type = self.parameters[xml_param.tag]
            if xml_param.tag == 'kpoints_grid':
                self.parameters['kpoints_grid'] = monkhorst_pack([int(el) for el in xml_param.text.split()])
            
            else :
                if isinstance(type,bool):
                    self.parameters[xml_param.tag] = boolean_converter(xml_param.text)
                elif isinstance(type,int):
                    self.parameters[xml_param.tag] = int(xml_param.text)
                elif isinstance(type,float):
                    self.parameters[xml_param.tag] = float(xml_param.text)
                elif isinstance(type,str):
                    self.parameters[xml_param.tag] = xml_param.text