from __future__ import annotations

import os, sys
import numpy as np
import ase 
import pickle
from ase import Atom, Atoms
from ase.io import read
from ase.calculators.vasp import Vasp
from typing import List, Dict, Tuple

from ARTWorker import ARTWorker
from ARTResults import ARTResults, AllInstance
from ASEVasp import set_up_VaspCalculator

class ARTManager : 
    def __init__(self,dic_species : Dict[str,int], dic_param : Dict[str,float], dic_pathways : Dict[str,any], dic_dumps : Dict[str,any], vasp_calculator : Vasp, restart_file : str = None) -> None :  
        if restart_file is None : 
            self.init_system = read(dic_pathways['InitialPath'])
            self.restart_file = None
        else :
            self.init_system = Atoms()
            self.restart_file = restart_file 
        
        """All usefull dictionnaries..."""
        self.dic_param = dic_param
        self.dic_pathways = dic_pathways
        self.dic_dumps = dic_dumps
        self.calc_vasp = vasp_calculator

        if not self.build_id_ARTn(dic_species) : 
            print('Problem with building ARTn id ...')
            exit(0)

    def update_VASPcalc(self,dic_setting : Dict[str, float | str | bool]) : 
        """Update the settings of VASP calculator
        
        Parameters
        ----------

        dic_setting : Dict[str, float | str | bool]
            Dictionnary of new settings to update in VASP calculator
        """
        self.calc_vasp = set_up_VaspCalculator(new_setup=dic_setting)

    def build_id_ARTn(self, dic_species : Dict[str,int]) -> bool :
        """Build the list of id of interest for ARTn method
        
        Parameters
        ----------

        dic_species : Dict[str,int]
            Dictionnary containing specific species for ARTn method (atom from molecule generally)
            for each key the value associated value is the number of atom species

        Returns
        -------

        bool 
            Boolean to check if there the same number of atom in the dic_species and in the system
        
        """
        check_dic = {key:0 for key in dic_species.keys()}
        self.list_id_ARTn = []
        for idx, at in enumerate(self.init_system) :
            if at.symbol in dic_species.keys() : 
                self.list_id_ARTn.append(idx)
                check_dic[at.symbol] += 1

        if check_dic == dic_species : 
            return True 
        else : 
            return False
          
    def ARTn_deformation_procedure(self, ART_worker : ARTWorker,  system : Atoms) -> Tuple[bool,Atoms] :
        """Perform the deformation procedure for ARTn method. This step is considered to be finished 
        if the maximum number of iteration NumberDefStep is reached or if the lambda_c criterion is reached
        
        Parameters
        ----------

        ART_worker : ARTWorker
            ARTWorker object which performs the procedure
        
        system : Atoms 
            Initial system for deformation ARTn procedure


        Returns 
        -------

        bool 
            Boolean to check if convergence criterion is reached
        
        Atoms 
            Updated system after deformation ARTn procedure
        """
        
        for _ in range(self.dic_param['NumberDefStep']) : 
            bool_conv, system = ART_worker.run_deformation_step(system, noise = self.dic_param['NoiseARTn'], tolerance = self.dic_param['ErrorHessian'])        
            if bool_conv : 
                return bool_conv, system 

            else : 
                return False, system

    def ARTn_salmon_procedure(self, ART_worker : ARTWorker,  system : Atoms) -> Tuple[bool,Atoms,float | None, float | None, np.ndarray | None] :
        """Perform the main procedure for ARTn method. This step is considered to be finished 
        if the maximum number of iteration NumberMainStep is reached or if the force criterion at saddle point is reached
        
        Parameters
        ----------

        ART_worker : ARTWorker
            ARTWorker object which performs the procedure
        
        system : Atoms 
            Initial system for main ARTn procedure


        Returns 
        -------

        bool 
            Boolean to check if convergence criterion is reached
        
        Atoms 
            Updated system after main ARTn procedure

        float | None 
            Energy of the system at the saddle point if convergence is reached None otherwise 

        float | None
            Minimum eigenvalue if convergence is reached None otherwise
        
        np.ndarray | None 
            Associated eigenvector if convergence is reached None otherwise
        """
        
        for _ in range(self.dic_param['NumberMainStep']) : 
            bool_conv, system, min_eig_val, min_eig_vect = ART_worker.run_ARTn_step(system, tolerance = self.dic_param['ErrorHessian'])       
            if bool_conv : 
                if self.dic_param['Debug'] : 
                    system_energy = ART_worker.get_energy_toy(system)
                else : 
                    system_energy = ART_worker.get_energy_vasp(system)              
                return bool_conv, system, system_energy, min_eig_val, min_eig_vect

            else : 
                return False, system, None, None, None

    def ARTn_drive_minimum_procedure(self, ART_worker : ARTWorker, minimum_system : Atoms, saddle_system : Atoms, eig_val : float, eig_vect : np.ndarray) -> Tuple[bool,Atoms,float,np.ndarray] :
        """Perform the driving new minimum procedure procedure for ARTn method.
        Saddle point system is pushed out following the direction associated to the lowest eigen value at saddle point 
        Then minimisation Verlet schem is performed until it reachs a new minimum.
        This step is considered to be finished when the force criterion is reached for the new minimum
        
        
        Parameters
        ----------

        ART_worker : ARTWorker
            ARTWorker object which performs the procedure
        
        minimum_system : Atoms 
            Last minimum system contains in ASE object

        saddle_system : Atoms 
            Current saddle point system contains in ASE object

        eig_val : float 
            Minimum eigenvalue at saddle point 

        eig_vect : np.ndarray
            Associated eigenvector

        Returns 
        -------

        bool 
            Boolean to check if convergence criterion is reached
        
        Atoms 
            Updated system after main ARTn procedure

        float 
            Energy of the new minimum configuration 

        np.ndarray 
            Descriptor of the new minimum configuration
        """
        
        """Pushing out step to reach an explored bassin"""
        saddle_system = ART_worker.run_push_saddle_step(minimum_system,saddle_system,eig_val,eig_vect)
        for _ in range(self.dic_param['NumberRelaxStep']) : 
            bool_conv, saddle_system = ART_worker.run_relaxation_step(saddle_system)     
            if bool_conv : 
                if self.dic_param['Debug'] : 
                    system_energy = ART_worker.get_energy_toy(saddle_system)
                else :
                    system_energy = ART_worker.get_energy_vasp(saddle_system)
                _, inertia_tensor = ART_worker.compute_moment_inertia_tensor(saddle_system, ART_worker.center_calculator(saddle_system))
                return bool_conv, saddle_system, system_energy, inertia_tensor

            else : 
                return False, saddle_system, None, None

    def init_ARTResults(self, ART_worker : ARTWorker) -> Tuple[ARTResults, AllInstance, float, np.ndarray] : 
        """Initialise the results objects for ART 
        
        Parameters
        ----------

        ART_worker : ARTWorker 
            Worker object

        Returns
        -------

        ARTResults 
            ART results object which stores all the exploration with connectivity reduction

        AllInstance
            ART results object which stores simply all the exploration without connectivity reduction

        float 
            Energy of the first minimum configuration

        np.ndarray 
            Inertia moment tensor of the miminmum configuration
        """

        if self.dic_param['Debug'] :
            init_energy = ART_worker.get_energy_toy(self.init_system)
        else :
            init_energy = ART_worker.get_energy_vasp(self.init_system)
        _, init_inertia_tensor = ART_worker.compute_moment_inertia_tensor(self.init_system,ART_worker.center_calculator(self.init_system))
        
        return ARTResults(self.init_system, init_energy, init_inertia_tensor), AllInstance('00000',self.init_system, init_energy,'minimum',init_inertia_tensor), init_energy, init_inertia_tensor

    def read_pickle(self, path_pickles : str) -> ARTResults | AllInstance : 
        """Read pickle files from previous run 
        
        Parameters 
        ----------

        path_pickles : str 
            Path to pickle file to read

        Returns 
        -------

        ARTResults | AllInstance
            previous ARTResults
        """
        return pickle.load(open(path_pickles,'rb'))

    def init_ARTResults_restart(self) -> Tuple[ARTResults, AllInstance] : 
        """Initialise the results objects for ART from restart files

        Returns
        -------

        ARTResults 
            ART results object which stores all the exploration with connectivity reduction

        AllInstance
            ART results object which stores simply all the exploration without connectivity reduction

        float 
            Energy of the first minimum configuration

        np.ndarray 
            Inertia moment tensor of the miminmum configuration
        """
        pickle_list = ['%s/%s'%(self.restart_file,f) for f in os.listdir(self.restart_file) if f.split('.')[1] == 'sav' ]
        for pickle in pickle_list : 
            if 'AllInstance' in os.path.basename(pickle) : 
                pickle_all_instance = pickle
            
            if 'ARTn' in os.path.basename(pickle) :
                pickle_connectivity = pickle

        return self.read_pickle(pickle_connectivity), self.read_pickle(pickle_all_instance)    

    def run_ART(self) :
        """Have to write all the shitty explanation for this method ...""" 
        ART_worker = ARTWorker(self.init_system, 
                               self.list_id_ARTn,
                               self.calc_vasp,
                               self.dic_param['Rcut'],
                               self.dic_param['DeltaXi'],
                               [self.dic_param['MaxDisplacementX'],self.dic_param['MaxDisplacementY'],self.dic_param['MaxDisplacementZ']],
                               self.dic_param['AlphaRand'],
                               self.dic_param['DeltaTRelax'],
                               self.dic_param['Dumping'],
                               self.dic_param['LambdaC'],
                               self.dic_param['Mu'],
                               self.dic_param['ToleranceForces'],
                               self.dic_param['Debug']) 
        
        """Initialise tempory object for connectivity"""
        if not self.dic_dumps['Restart'] : 
            previous_system = self.init_system.copy()
            previous_saddle_system = self.init_system.copy()
            system = self.init_system.copy()
            """Results object initialisation"""       
            ART_results, ART_all_instance, energy, descriptor = self.init_ARTResults(ART_worker)
            previous_energy, previous_descriptor = energy, descriptor
            self.update_VASPcalc({'restart':True})

        else : 
            """Restarting exploration option from the previous exploration"""
            ART_results, ART_all_instance = self.init_ARTResults_restart()
            if ART_all_instance.all_instance[-1]['type'] == 'minimum' : 
                previous_energy, previous_descriptor = ART_all_instance.all_instance[-1]['energy'], ART_all_instance.all_instance[-1]['descriptor']
                try : 
                    previous_system, previous_saddle_system = ART_all_instance.all_instance[-1]['configuration'], ART_all_instance.all_instance[-2]['configuration']
                except : 
                    previous_system, previous_saddle_system = ART_all_instance.all_instance[-1]['configuration'], ART_all_instance.all_instance[-1]['configuration']
                    print('Problem with last saddle point configuration in restart !')
                    print('I put the last minimum...')

            else : 
                try : 
                    previous_energy, previous_descriptor = ART_all_instance.all_instance[-2]['energy'], ART_all_instance.all_instance[-2]['descriptor']
                    previous_system, previous_saddle_system = ART_all_instance.all_instance[-2]['configuration'], ART_all_instance.all_instance[-1]['configuration']
                except : 
                    print('I can not find a minimum configuration to restart ...')
                    print('Restarting wont be launched, be sure that your pickle files are correct ...')
                    exit(0)
            
            if self.dic_param['Debug'] : 
                ART_worker.get_energy_toy(previous_system)
            else :
                ART_worker.get_energy_vasp(previous_system)
        
            self.update_VASPcalc({'restart':True})

        for k in range(self.dic_param['NumberOfPaths']) : 
            """Deformation procedure..."""
            bool_def, system = self.ARTn_deformation_procedure(ART_worker, system)
            if bool_def : 
                """Main ART procedure"""
                bool_salmon, system, energy_saddle, eig_val, eig_vect = self.ARTn_salmon_procedure(ART_worker, system)
                if bool_salmon : 
                    """Time to fill the all instance object"""
                    descriptor = ART_worker.compute_moment_inertia_tensor(system,ART_worker.center_calculator(system))
                    previous_saddle_system = system.copy()
                    ART_all_instance.add_AllInstance('Roger',system,energy_saddle,'saddle',descriptor) #change the fucking Roger !
                    
                    """Pushing out + relaxation step"""
                    bool_relax, system, energy_minimum, descriptor = self.ARTn_drive_minimum_procedure(ART_worker, previous_system, previous_saddle_system, eig_val, eig_vect)
                    
                    """Little pickles writing"""
                    if k%self.dic_dumps['DumpsPickleStep'] == 0 :
                        ART_all_instance.dump_pickles_AllInstantce_object(self.dic_pathways['DumpPath'])
                    
                    if k%self.dic_dumps['DumpsPoscarStep'] == 0 and self.dic_dumps['DumpsConfig'] : 
                        ART_all_instance.dump_AllInstance_vasp_config(self.dic_pathways['DumpPath'])
                    
                    if bool_relax : 
                        """Update the All instance object"""
                        ART_all_instance.add_AllInstance('Roger',system,energy_minimum,'minimum',descriptor)

                        """Update ART results objects with connectivity"""
                        ART_results.add_event_linear('Roger', previous_system, previous_energy, energy_minimum, previous_saddle_system, energy_saddle, descriptor)
                        ART_results.add_event_graphs('Roger', previous_system, previous_energy, energy_minimum, previous_saddle_system, energy_saddle, previous_descriptor,descriptor) 
                        
                        """Updating previous variable ..."""
                        previous_system, previous_energy, previous_descriptor = system, energy_minimum, descriptor 
                        
                        ART_results.dump_log_file('Roger', energy_minimum, energy_saddle, descriptor, self.dic_pathways['DumpPath'])
                        if k%self.dic_dumps['DumpsPickleStep'] == 0 :
                            ART_results.dump_pickle(self.dic_pathways['DumpPath'])                       
                        

                    else : 
                        print('There is something wrong with pushing out / relaxation procedure...')
                        print('Maximum number for RelaxSteps has been reached without ensuring the forces criterion...')
                        print('ARTn will stop...')
                        exit(0)                      


                else : 
                    print('There is something wrong with salmon procedure...')
                    print('Maximum number for MainSteps has been reached without ensuring the forces criterion...')
                    print('ARTn will stop...')
                    exit(0)

            else : 
                print('There is something wrong with deformation procedure...')
                print('Maximum number for DeformationSteps has been reached without ensuring the lambda_c criterion...')
                print('ARTn will stop...')
                exit(0)

        print('NumberOfPaths is reached ARTn python will stop...')
        print('See you later :)')