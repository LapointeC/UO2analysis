from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import os, argparse, re
from mpi4py import MPI

#from BaseManager import BaseManager
#from BaseParser import BaseParser
#from LAMMPSWorker import LAMMPSWorker
#from SGC_MCObject import SGCMC, VC_SGCMC

from MC import BaseManager, BaseParser, LAMMPSWorker, SGCMC, VC_SGCMC

"""Algorithm from : Scalable parallel Monte Carlo algorithm for atomistic simulations of precipitation in alloys
https://link.aps.org/doi/10.1103/PhysRevB.85.184203

+ code clearly inspired from : https://github.com/tomswinburne/pafi
Unsupervised Calculation of Free Energy Barriers in Large Crystalline Systems
https://link.aps.org/doi/10.1103/PhysRevLett.120.135503 
"""

class MCManager(BaseManager):
    def __init__(self, world: MPI.Intracomm, 
                 xml_path:None|os.PathLike[str]=None,
                 parameters:None|BaseParser=None,
                 Worker:LAMMPSWorker=LAMMPSWorker) -> None:
        """Default manager of MC calculation, child of BaseManager

        Parameters
        ----------
        world : MPI.Intracomm
            MPI communicator
        xml_path : None or os.PathLike[str], optional
            path to XML configuration file, default None
        parameters : None or BABFParser object, optional
            preloaded BABFParser object, default None
        Worker : LammpsWorker, optional,
            Can be overwritten by child class, by default LammpsWorker
        """
        
        
        assert (not parameters is None) or (not xml_path is None)
        
        if parameters is None:
            # TODO have standalone check for suffix in config_[suffix].xml?
            # not the best solution currently if we use BaseManager alone....
            parameters = BaseParser(xml_path=xml_path,rank=world.Get_rank())
        
        super().__init__(world, parameters, Worker)

    

    def manage_restart(self, lammps_worker : LAMMPSWorker, dic_mu : dict = None, alpha : float = 0.1) -> dict : 

        def extract_numbers(s : str) -> List[float]:
            """"Extract float from string 
            
            Parameters
            ----------

            s : str 
                Original string

            Return
            ------

            List[float]
                List of float contains in string
            """
            # Regular expression pattern to match both integers and floating-point numbers
            number_pattern = r'-?\d+(?:\.\d+)?'
            number_strings = re.findall(number_pattern, s)
            return [float(num) for num in number_strings]

        if self.rank == 0 : 
            print('... You are using restart option ...')
            self.log('... You are using restart option ...')

        if self.parameters.parameters["Mode"] == 'SGC-MC' : 
            restart_file = max([ f'{self.parameters.parameters["WritingDirectory"]}/{f}' for f in os.listdir(self.parameters.parameters['WritingDirectory']) ],
                               key = os.path.getatime)
            last_mu = extract_numbers(os.path.basename(restart_file))[0]
            if last_mu > 0 :
                alpha = alpha + 1.0 
            else : 
                alpha = -alpha + 1.0
            new_dic_mu = {key: value for key, value in dic_mu.items() if key > alpha*last_mu}

            #update lammps worker 
            self.parameters.Configuration = restart_file
            lammps_worker.run_commands("clear")
            lammps_worker.run_script("Input")

            return new_dic_mu
        
        elif self.parameters.parameters["Mode"] == 'VC-SGC-MC' :
            restart_file = max([ f"{self.parameters.parameters['WritingDirectory']}/{f}" for f in os.listdir(self.parameters.parameters['WritingDirectory']) ],
                               key = os.path.getatime)
            self.parameters.Configuration = restart_file
            lammps_worker.run_script("Input")
            
            return None                    

        else : 
            raise NotImplementedError('... This mode is not implemented for restart ...')


    def log(self,string : str) -> None :
        """Writing log file with reuslts ...
        
        Parameters:
        -----------

        string : str 
            string to write in log file 

        """
        with open('mc.log','a') as w : 
            w.write(f'{string} \n')
        return

    
    def get_number_of_atoms_each_species(self) -> Tuple[np.ndarray,np.ndarray] : 
        """Get the number of atoms of each species in the system
        
        Returns:
        --------
        np.ndarray 
            Array of type id of species in lammps system 
        
        np.ndarray 
            Array of number of atom associated to each species in the system
        """
        type_array = self.Worker.gather('type')
        array_id, array_nb = np.unique(type_array.flatten(), return_counts=True)
        if np.amin(array_id) != 1 : 
            raise ValueError(f'Minimum id for array type is not 1 => {np.amin(array_id)}')
        else : 
            if (np.array([ i for i in range(1,len(array_id)+ 1)]) != array_id).all() : 
                raise ValueError(f'Types in lammps have bad ordering => {array_id}')

        return array_id, array_nb

    def create_npt_script(self, skim_script : str) -> str :    
        """Create the NPT script for MC sampling
        
        Parameters:
        -----------

        skim_script : str
            Skim script of NPT command in Lammps 

        Returns:
        --------

        str 
            Filled NPT command for Lammps 
        """
        temperature = self.parameters.parameters['Temperature']
        damping_temperature = self.parameters.parameters['DampingT']
        damping_barostat = self.parameters.parameters['DampingP']
        
        set_cmd = self.parameters.replace(skim_script,'TEMPERATURE',temperature)
        set_cmd = self.parameters.replace(set_cmd,'DAMPT',damping_temperature)
        return self.parameters.replace(set_cmd,'DAMPP',damping_barostat)
    
    def build_mu_dict(self,dic_mu : dict) -> Dict[float, np.ndarray] :
        """Build the dictionnary for mu grid
        
        Parameters:
        ----------

        dic_mu : dict
            Dictionnary to build mu grid in case of chemical potential estimation 

        Returns:
        --------

        Dict[float,np.ndarray]
            Dictionnary of chemical potential grid...
        """
        dic_mu_grid : Dict[float, np.ndarray] = {}
        mu_grid = np.linspace(dic_mu['Min'],dic_mu['Max'],num=dic_mu['NumberMu'])
        for mu in mu_grid :
            dic_mu_grid[mu] = np.array([0.0,mu])
        
        return dic_mu_grid

    def run(self) -> None:
        """Run MC sampling
        """
        assert self.parameters.ready()
        
        if self.rank==0:
            screen_out = f"""
            Initialized {self.nWorkers} workers with {self.CoresPerWorker} cores
            """
            print(screen_out)

        _, array_species = self.get_number_of_atoms_each_species()
        if self.parameters.parameters['Mode'] == 'SGC-MC' :
            """"HERE IS THE SEMI GRAND CANONICAL MC"""
            dic_mu = self.build_mu_dict(self.parameters.parameters['MuGrid'])
            if self.parameters.parameters["Restart"] : 
                dic_mu = self.manage_restart(self.Worker, dic_mu=dic_mu)
                _, array_species = self.get_number_of_atoms_each_species()

            if self.rank == 0 : 
                print('delta mu | <c0> | <c1> | <c^2> - <c>^2') 
                self.log('delta mu | <c0> | <c1> | <c^2> - <c>^2')

            npt_script = self.create_npt_script(self.parameters.scripts['NPTScript'])

            self.Worker.run_commands(npt_script)
            self.Worker.run_commands(f"run {self.parameters.parameters['ThermalisationSteps']}")

            for _, grid_mu in dic_mu.items() :  
                if len(grid_mu) != len(array_species) and self.rank == 0 : 
                    raise TimeoutError('Array of chemical potential and species in the system are inconsistant')
                
                # main program 
                MC_object = SGCMC(grid_mu, 
                                  array_species, 
                                  self.Worker, self.parameters.parameters["WritingDirectory"], 
                                  equiv_mu=self.parameters.parameters["EquivMu"])
                
                nb_main_step = int(self.parameters.parameters['NumberNPTSteps']/self.parameters.parameters['FrequencyMC'])
                for nb_it in range(nb_main_step) : 
                    MC_object.perform_lammps_script(f"run {self.parameters.parameters['FrequencyMC']}")
                    _ = MC_object.perform_SGCMC(self.parameters.parameters['FractionSwap'],self.parameters.parameters['Temperature'])

                if self.rank == 0 : 
                    average_c, variance_c = MC_object.compute_average()
                    delta_mu = grid_mu[1] - grid_mu[0]
                    print('{:1.3f} | {:1.3f} | {:1.3f} | {:1.3f} '.format(delta_mu,average_c[0],average_c[1],np.sqrt(variance_c[1])))
                    self.log('{:1.3f} | {:1.3f} | {:1.3f} | {:1.3f} '.format(delta_mu,average_c[0],average_c[1],np.sqrt(variance_c[1])))
                MC_object.dump_configuration('{:1.3f}'.format(grid_mu[1]))                

        elif self.parameters.parameters['Mode'] == 'VC-SGC-MC' :
            """"HERE IS THE VARIANCE CONSTRAINED SEMI GRAND CANONICAL MC"""
            
            array_mu = np.array(self.parameters.parameters['MuArray']) 
            array_concentration = np.array(self.parameters.parameters['ConcentrationArray'])
            if len(array_mu) != len(array_species) and self.rank == 0 : 
                raise TimeoutError('Array of chemical potential and species in the system are inconsistant')
            
            if len(array_concentration) != len(array_species) and self.rank == 0 : 
                raise TimeoutError('Array of concentrations and species in the system are inconsistant')
            
            if self.parameters.parameters["Restart"] : 
                dic_mu = self.manage_restart(self.Worker, dic_mu=dic_mu)
                _, array_species = self.get_number_of_atoms_each_species()
            
            if self.rank == 0 : 
                array_txt = [ f' <c{i}> |' for i in range(len(array_concentration)) ] + [' max( <c^2> - <c>^2 )']
                print("".join(array_txt)) 
                self.log("".join(array_txt))

            #npt script
            npt_script = self.create_npt_script(self.parameters.scripts['NPTScript'])
            self.Worker.run_commands(npt_script)
            self.Worker.run_commands(f"run {self.parameters.parameters['ThermalisationSteps']}")

            VCMC_object = VC_SGCMC(array_mu, 
                                   array_concentration, 
                                   array_species, 
                                   self.parameters.parameters['Kappa'],
                                   self.Worker, 
                                   self.parameters.parameters["WritingDirectory"],
                                   equiv_mu=self.parameters.parameters["EquivMu"])
    
            # main program 
            patched_writing_step = int(self.parameters.parameters['WritingStep']/self.parameters.parameters['FrequencyMC'])
            nb_main_step = int(self.parameters.parameters['NumberNPTSteps']/self.parameters.parameters['FrequencyMC'])
            for nb_it in range(nb_main_step) : 
                VCMC_object.perform_lammps_script(f"run {self.parameters.parameters['FrequencyMC']}")
                _ = VCMC_object.perform_VC_SGCMC(self.parameters.parameters['FractionSwap'],self.parameters.parameters['Temperature'])

                if nb_it%patched_writing_step == 0 : 
                    VCMC_object.dump_configuration(f'{nb_it}')
                    if self.rank == 0 : 
                        average_c, variance_c = VCMC_object.compute_average()
                        array_txt = [' {:1.3f} |'.format(c) for c in average_c ] + [' {:1.10f}'.format(np.amax(np.sqrt(variance_c)))]
                        print("".join(array_txt))
                        self.log("".join(array_txt))
        
        else : 
            raise NotImplementedError('This mode is not implemented !')

        return

"""
Here is just the launching part of the script ...
"""
parser = argparse.ArgumentParser('MC')
parser.add_argument('-p','--parameters',default="SGMC.xml")
args = parser.parse_args()
xml_file = args.parameters if os.path.exists(args.parameters) else 'SGMC.xml'

MC_obj = MCManager(MPI.COMM_WORLD, xml_file)
MC_obj.run()