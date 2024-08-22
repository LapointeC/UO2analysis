from ase import Atoms
from typing import Dict, List
import numpy as np
import xml.etree.ElementTree as ET
import re, os

from ASEVasp import set_up_VaspCalculator

class ARTParser : 
    def __init__(self, path_inputs : str) : 
        self.parameters = {}
        self.dumps = {}
        self.pathways = {}
        
        self.init_params()
        self.parser_inputs_file(path_inputs)
        self.convert_molecule_format()
        self.calculator = set_up_VaspCalculator()
        self.check_restart()
        if not self.check_minimal_arg() : 
            exit(0)
        self.link_POTCAR_file()

    def init_params(self) : 
        """Initialise the input dictionnary !"""
        self.parameters['Rcut'] = 5.0 
        self.parameters['DeltaXi'] = 0.01
        self.parameters['ErrorHessian'] = 0.01
        self.parameters['NumberOfPaths'] = 100
        self.parameters['NumberDefStep'] = 10
        self.parameters['VerletIteration'] = 5
        self.parameters['NoiseARTn'] = True
        self.parameters['MaxDisplacementX'] = 0.5
        self.parameters['MaxDisplacementY'] = 0.5
        self.parameters['MaxDisplacementZ'] = 0.5
        self.parameters['AlphaRand'] = 0.0
        self.parameters['NumberMainStep'] = 10
        self.parameters['NumberRelaxStep'] = 100
        self.parameters['DeltaTRelax'] = 1.0 #have to check unit ...
        self.parameters['Dumping'] = 1.0 #have to check unit of dumping parameter...
        self.parameters['LambdaC'] = - 2*np.pi*1.0 # in (rad.THz)^2
        self.parameters['Mu'] = 0.1
        self.parameters['ToleranceForces'] = 1e-1 #shitty but I dont care !
        self.parameters['Molecule'] = 'C2H2'
        self.parameters['Debug'] = False

        self.dumps['Restart'] = False
        self.dumps['PickleFiles'] = 'GivenPath'     
        self.dumps['DumpsConfig'] = False
        self.dumps['DumpsPoscarStep'] = 1
        self.dumps['DumpsPickleStep'] = 10

        self.pathways['CalcPath'] = os.getcwd()
        self.pathways['Dumps'] = '%s/dumps'%(os.getcwd())

    def check_restart(self) : 
        """Check the restart option"""
        if self.dumps['Restart'] : 
            if not os.path.exists(self.dumps['PickleFiles']) :  
                print('RestartFiles path does not exist ...')
                exit(0)
        
        else : 
            self.dumps['PickleFiles'] = None

    def convert_molecule_format(self) : 
        """Convert the string molecule format into dictionnary"""
        split_molecule = re.split('(\d+)', self.parameters['Molecule'])
        split_molecule = [el for el in split_molecule if el != '']
        self.dic_molecule = {split_molecule[2*k]:int(split_molecule[2*k+1]) for k in range(int(len(split_molecule)/2))}

    def link_POTCAR_file(self) :
        """Link the potcar file in the Calc directory"""
        os.environ['VASP_PP_PATH'] = self.pathways['POTCARPath']

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
        for glob_attrib in ['ARTparameters', 'Pathways', 'DumpResults'] : 
            if xml_root.find(glob_attrib) is None :
                if glob_attrib == 'ARTparameters' or glob_attrib == 'DumpResults' : 
                    print(' %s will be set by default ...'%(glob_attrib))
                else : 
                    print(' Pathways have to be defined !')
                    exit(0)

            else : 
                xml_tree_glob_attrib = xml_root.find(glob_attrib)
                if glob_attrib == 'ARTparameters' : 
                    for xml_param in xml_tree_glob_attrib : 
                        type = self.parameters[xml_param.tag]
                        if isinstance(type,bool):
                            self.parameters[xml_param.tag] = boolean_converter(xml_param.text)
                        elif isinstance(type,int):
                            self.parameters[xml_param.tag] = int(xml_param.text)
                        elif isinstance(type,float):
                            self.parameters[xml_param.tag] = float(xml_param.text)
                        elif isinstance(type,str):
                            self.parameters[xml_param.tag] = xml_param.text
                
                if glob_attrib == 'Pathways' : 
                    for xml_param in xml_tree_glob_attrib : 
                        self.pathways[xml_param.tag] = str(xml_param.text)

                if glob_attrib == 'DumpResults' : 
                    for xml_param in xml_tree_glob_attrib : 
                        type = self.dumps[xml_param.tag]
                        if isinstance(type,bool):
                            self.dumps[xml_param.tag] = boolean_converter(xml_param.text)
                        elif isinstance(type,int):
                            self.dumps[xml_param.tag] = int(xml_param.text)
                        elif isinstance(type,float):
                            self.dumps[xml_param.tag] = float(xml_param.text)
                        elif isinstance(type,str):
                            self.dumps[xml_param.tag] = xml_param.text
    
    def check_minimal_arg(self) -> bool :
        """Check if the minimal arguments for ART method are defined ...
        
        Returns
        -------

        bool 
            Return False if there is problem with minimal argument, True otherwise
        """
        bool_min_arg = True
        dic_min_arg = {'InitialPath':lambda path : os.path.exists(path),'POTCARPath':lambda path : os.path.exists(path)}
        for dict in [self.parameters, self.dumps, self.pathways] : 
            for key in dic_min_arg :
                if key in dict.keys() : 
                    if not dic_min_arg[key](dict[key]) : 
                        print('I have some trouble with %s argument ...'%(key))
                        bool_min_arg = False
                else : 
                    continue
        
        return bool_min_arg
        