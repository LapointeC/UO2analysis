import numpy as np
import os
from mpi4py import MPI
from typing import List
from ..parsers.BABFParser import BABFParser
from .LAMMPSWorker import LAMMPSWorker
from .ReferenceWorker import ReferenceWorker
from ..results.ResultsBABF import ResultsBABF

class BABFWorker(LAMMPSWorker,ReferenceWorker):
    """
        The hyperplane-constrained sampling run.
        results: ResultsHolder instance
            essentially a dictionary, defining both 
            input and output data- temeperature, reaction coordinate
            and other are defined, and others are filled.
            *Crucially, if a parameter is specified in results, 
            it overrides any value in the parameter file*
            Must have at least `ReactionCoordinate` and `Temperature` defined
        
        The PAFI workflow can be summarised as:
            1) Execute `PreRun` script
            2) Apply `fix_pafi` constraint at defined `ReactionCoordinate`
            3) Execute `PreTherm` script
            4) Thermalization for `ThermSteps` steps at `Temperature`
            5) Execute `constrained_average()` function
                In standard PAFI this runs for `SampleSteps` steps and 
                time averages the output of `fix_pafi`, as shown below.
                See https://docs.lammps.org/fix_pafi.html for details.
            6) Extract the average displacment from path if `PostDump==1`
            7) Minimize in-plane to check system returns to path.
                The check is the max per-atom displacement : `MaxJump`
                If `MaxJump` is larger than `MaxJumpMaxJumpThresh` then 
                the sample is retained but marked as `Valid=False`
            8) Execute `PostRun` script
        
        The `contrained_average()` function is therefore suitable for
        any form of hyperplane-constrained averaging.
        """
    
    def __init__(self, comm: MPI.Intracomm, 
                 parameters: BABFParser, tag: int,
                 rank: int, roots: List[int]) -> None:
        super().__init__(comm, parameters, tag, rank, roots)
    
    def update_free_energy_quantities(self, results : ResultsBABF, temperature : float, block : bool = False) -> ResultsBABF : 
        """Perform a global update of ResultsBABF object with new estimation of free energy quantities
        
        Parameters:
        -----------
        results : ResultsBABF 
            Object containing all free energy data and methods
        temperature : float 
            Simulation temperature
        block : bool 
            Key word for constrained babf

        Results
        -------
        ResultsBABF
            Updated free energy object

        
        """
        energy_lammps = self.get_energy()
        energy_reference = self.evaluate_reference_energy()

        """Little temperature update !"""
        results.update_temperature( self.evaluate_potential_temperature(), block=block )

        mixed_potential = results.evaluate_U_lambda(energy_reference,energy_lammps)
        if results.free_energy is None : 
            free_energy = np.zeros(len(mixed_potential))
        else : 
            free_energy = results.free_energy

        """Here we update p_A(\lambda|q)"""
        conditional_p_lambda = results.evaluate_conditional_p_lambda(temperature,mixed_potential,free_energy)
        results.update_data_babf(mixed_potential*conditional_p_lambda,'sum_w_AxU_dl',block=block)
        results.update_data_babf(np.power(mixed_potential,2)*conditional_p_lambda,'sum_w_AxU2_dl',block=block)
        results.update_data_babf(conditional_p_lambda,'sum_w_A',block=block)
        results.data_babf['pc_lam_q'] = conditional_p_lambda   

        
        if block : 
            energy_lammps += self.evaluate_standard_constrained_energy()
            mixed_potential_c = results.evaluate_U_lambda(energy_reference,energy_lammps)
            
            if results.free_energy is None : 
                free_energy_c = np.zeros(len(mixed_potential))
            else : 
                free_energy_c = results.free_energy_block
            """Here we update \pi_A^c(\lambda|q) for constrained free energy"""
            conditional_pi_lambda = results.evaluate_conditional_p_lambda(temperature,mixed_potential_c,free_energy_c)
            results.update_data_babf(mixed_potential_c*conditional_pi_lambda,'sum_w_AxU_dl',block=True)
            results.update_data_babf(np.power(mixed_potential_c,2)*conditional_pi_lambda,'sum_w_AxU2_dl',block=True)
            results.update_data_babf(conditional_pi_lambda,'sum_w_A',block=True)
            results.data_babf_block['pc_lam_q'] = conditional_pi_lambda           
    
        return results

    """DYNAMICS DEFINITIONS"""
    def Implicit_Verlet_dynamic(self,results : ResultsBABF, temperature : float, delta_t : float, block : bool = False) -> ResultsBABF:
        """Perform midpoint Euler-Verlet for overdumped dynamic
        ref : Free energy computations a mathematical perspective, p. 94
        Parameters
        ----------
        results : ResultsBABF 
            Object containing all free energy data and methods
        temperature : float 
            Simulation temperature
        delta_t : float 
            Time step for stochastic dynamic 
        block : bool 
            Key word for constrained babf

        Results
        -------
        ResultsBABF
            Updated free energy object
        """

        gamma = 1/(delta_t*100)

        """v_n -> v_n+1/4"""
        self.v = self.v*np.array(np.exp(-gamma*0.5*delta_t)) + self.sampling_Wigner_process(temperature,0.5*delta_t,self.v.shape)

        """v_n+1/4 -> v_n+1/2"""
        self.v = self.v + self.force*0.5*delta_t

        """q_n -> q_n+1"""
        self.x = self.x + self.v*delta_t

        """update positions"""
        self.centering_system()
        self.x = self.pbc(self.x)
        self.x_reference = self.pbc(self.x_reference)
        self.scatter('x',self.x)

        """update forces"""
        forces_lammps = self.evaluate_lammps_forces()
        if block : 
            forces_lammps += self.evaluate_constrained_forces()

        forces_reference = self.evaluate_reference_forces()
        effective_forces = results.evaluate_effective_forces_dynamic(forces_reference,forces_lammps)
        self.force = effective_forces

        """update result object"""
        results = self.update_free_energy_quantities(results,
                                                     temperature,
                                                     block=block)

        """p_n+1/2 -> p_n+3/4"""
        self.v = self.v + effective_forces*0.5*delta_t

        """p_n+3/4 -> p_n+1"""
        self.v = self.v*np.exp(-gamma*0.5*delta_t) + self.sampling_Wigner_process(temperature,0.5*delta_t,self.v.shape)

        return results

    def BAOAB_scheme(self,results : ResultsBABF, temperature : float , delta_t : float, block : bool = False) -> ResultsBABF:
        """Perform BAOAB predictor-corrector schem
        ref : Free energy computations a mathematical perspective, p. 94
        Parameters
        ----------
        results : ResultsBABF 
            Object containing all free energy data and methods
        temperature : float 
            Simulation temperature
        delta_t : float 
            time step for stochastic dynamic 
        block : bool 
            Key word for constrained babf

        Results
        -------
        ResultsBABF
            Updated free energy object
        """
        gamma = 1/(delta_t*100)

        """v^_n-1/2 -> v_n"""
        self.v = self.v + self.mass_array@self.force*0.5*delta_t

        """v_n -> v_n+1/2 (update of forces)"""
        forces_lammps = self.evaluate_lammps_forces()
        if block : 
            forces_lammps += self.evaluate_constrained_forces()

        forces_reference = self.evaluate_reference_forces()
        effective_forces = results.evaluate_effective_forces_dynamic(forces_reference,forces_lammps)
        self.force = effective_forces       
        self.v = self.v + self.mass_array@effective_forces*0.5*delta_t

        """update result object"""
        results = self.update_free_energy_quantities(results,
                                                     temperature,
                                                     block=block)

        """q_n -> q_n+1/2"""
        self.x = self.x + self.v*0.5*delta_t

        """p_n+1/2 -> p^_n+1/2"""
        self.v = np.exp(-gamma*delta_t)*self.v + np.sqrt(1.0-np.exp(-2*gamma*delta_t))*self.sampling_Wigner_process(temperature,0.5*delta_t,self.v.shape)

        """q_n+1/2 -> q_n+1"""
        self.x = self.x + self.v*0.5*delta_t

        """update in lammps !"""
        self.centering_system()
        self.x = self.pbc(self.x)
        self.x_reference = self.pbc(self.x_reference)
        self.scatter('x',self.x)

        return results


    def Overdamped_Langevin_dynamic(self, results : ResultsBABF, temperature : float, delta_t : float, block : bool = False) -> ResultsBABF:
        """Perform Langevin scheme for overdumped dynamic
        ref : Free energy computations a mathematical perspective, p. 99
        Parameters
        ----------
        results : ResultsBABF 
            Object containing all free energy data and methods
        temperature : float 
            Simulation temperature
        delta_t : float 
            time step for stochastic dynamic 
        block : bool 
            Key word for constrained babf

        Results
        -------
        ResultsBABF
            Updated free energy object
        """

        """Langevin dynamic update overdumped"""
        forces_lammps = self.evaluate_lammps_forces()
        if block :
            forces_lammps += self.evaluate_constrained_forces()

        forces_reference = self.evaluate_reference_forces()
        effective_forces = results.evaluate_effective_forces_dynamic(forces_reference,forces_lammps)
        self.force = effective_forces
        self.x = self.x + effective_forces*delta_t + self.sampling_Wigner_process(temperature,delta_t,self.x.shape)

        """update result object"""
        results = self.update_free_energy_quantities(results,
                                                     temperature,
                                                     block=block)

        """update in lammps !"""
        self.centering_system()
        self.x = self.pbc(self.x)
        self.x_reference = self.pbc(self.x_reference)
        self.scatter('x',self.x)

        return results


    """UPDATE SCHEM"""
    def update_step_BABF(self,results:ResultsBABF, block : bool = False) -> ResultsBABF:
        """Unitary step for stochastic dynamics with update of results objects
        Paramters 
        ---------
        block : bool 
            Key word for constrained BABF method

        Returns
        -------
        ResultsBABF
            updated object : 
            block = False : p_A(\lambda|q_i)
                            \sum_{i}^{s} [ U(q_i,\lambda) - U_{ref}(q_i,\lambda) ] p_A(\lambda | q_i) 
                            \sum_{i}^{s} p_A(\lambda | q_i)  are updated

            block = True : p^c_A(\lambda|q_i), pi^c_A(\lambda|q_i)
                           \sum_{i}^{s} [ U(q_i,\lambda) - U_{ref}(q_i,\lambda) ] p^c_A(\lambda | q_i) 
                           \sum_{i}^{s} p^c_A(\lambda | q_i) 
                           \sum_{i}^{s} [ U_c(q_i,\lambda) - U_{ref}(q_i,\lambda) ] \pi^c_A(\lambda | q_i) 
                           \sum_{i}^{s} \pi^c_A(\lambda | q_i)  are updated
        """
        parameters = lambda k: self.parameters(k)
        dynamic = parameters('Dynamic')
        delta_t = parameters('DeltaT')
        temperature = parameters('Temperature')

        dictionnary_dynamic = {"ImpliciteVerlet": lambda res, b : self.Implicit_Verlet_dynamic(res,temperature,delta_t, block=b),
                               "BAOAB": lambda res, b : self.BAOAB_scheme(res, temperature, delta_t, block=b),
                               "OverdampedLangevin": lambda res, b : self.Overdamped_Langevin_dynamic(res, temperature, delta_t, block=b)}

        #if dynamic == 'ImpliciteVerlet' :
        #    return self.Implicit_Verlet_dynamic(results, temperature , delta_t, block=block)
        #if dynamic == 'BAOAB' : 
        #    return self.BAOAB_scheme(results, temperature, delta_t, block=block)
        #if dynamic == 'OverdampedLangevin' : 
        #    return self.Overdamped_Langevin_dynamic(results,temperature,delta_t, block=block)
        return dictionnary_dynamic[dynamic](results,block)

    """THERMALISATION STEPS"""
    def thermalisation_steps(self,results:ResultsBABF, block : bool = False) -> ResultsBABF :
        """Perform thermalisation step without updating free energy estimator"""
        parameters = lambda k: self.parameters(k)
        thermalisation_step = parameters('ThermalisationStep')    
        if self.rank == 0 : 
            print('... Starting thermalisation procedure ... (%1.1e steps)'%(thermalisation_step))

        for _ in range(thermalisation_step) : 
            results = self.update_step_BABF(results, block=block)
        
        if self.rank == 0 : 
            print('... Finishing thermalisation procedure ...')

        return results

  
            



        





        


