import numpy as np
import os
from mpi4py import MPI
from typing import List
from ..parsers.BABFParser import BABFParser
from .LAMMPSWorker import LAMMPSWorker
from .ReferenceWorker import ReferenceWorker
from ..results.ResultsBABF import ResultsBABF

from DynamicsSchemes import Overdamped_Langevin_dynamic, BAOAB_scheme, Implicit_Verlet_dynamic

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
    
    def update_free_energy_quantities(self, results : ResultsBABF, temperature : float) -> ResultsBABF : 
        """Perform a global update of ResultsBABF object with new estimation of free energy quantities
        
        Parameters:
        -----------
        
        results : ResultsBABF 
            Object containing all free energy data and methods
        
        temperature : float 
            Simulation temperature


        Results
        -------
        
        ResultsBABF
            Updated free energy object

        
        """
        energy_lammps = self.get_energy()
        energy_reference = self.evaluate_reference_energy()

        """Little temperature update !"""
        results.update_temperature( self.evaluate_potential_temperature(), block=self.block)

        mixed_potential = results.evaluate_U_lambda(energy_reference,energy_lammps)
        if results.free_energy is None : 
            free_energy = np.zeros(len(mixed_potential))
        else : 
            free_energy = results.free_energy

        """Here we update p_A(\lambda|q)"""
        conditional_p_lambda = results.evaluate_conditional_p_lambda(temperature,mixed_potential,free_energy)
        results.update_data_babf(mixed_potential*conditional_p_lambda,'sum_w_AxU_dl',block=self.block)
        results.update_data_babf(conditional_p_lambda,'sum_w_A',block=self.block)
        
        """here is the variance estimator for BABF schem
        [ \mathcal{O}p - < \mathcal{O}p > ]^2 = \mathcal{O}^2p^2 - 2\mathcal{O}p < \mathcal{O}p > + < \mathcal{O}p >^2"""
        mean_force = results.evaluate_derivative_free_energy(block=self.block)
        Op = mixed_potential*conditional_p_lambda
        variance_estimator = np.power(Op, 2.0) - 2*mean_force*Op + np.power(mean_force, 2.0)
        results.update_data_babf(variance_estimator,'sum_w_A2xU2_dl',block=self.block)
        
        results.data_babf['pc_lam_q'] = conditional_p_lambda   

        if self.block : 
            energy_lammps += self.evaluate_standard_constrained_energy()
            mixed_potential_c = results.evaluate_U_lambda(energy_reference,energy_lammps)
            
            if results.free_energy is None : 
                free_energy_c = np.zeros(len(mixed_potential))
            else : 
                free_energy_c = results.free_energy_block
            
            """Here we update \pi_A^c(\lambda|q) for constrained free energy"""
            conditional_pi_lambda = results.evaluate_conditional_p_lambda(temperature,mixed_potential_c,free_energy_c)
            results.update_data_babf(mixed_potential_c*conditional_pi_lambda,'sum_w_AxU_dl',block=True)

            """here is the variance estimator for BABF schem
            [ \mathcal{O}p - < \mathcal{O}p > ]^2 = \mathcal{O}^2p^2 - 2\mathcal{O}p < \mathcal{O}p > + < \mathcal{O}p >^2"""
            mean_force = results.evaluate_derivative_free_energy(block=True)
            Op = mixed_potential*conditional_p_lambda
            variance_estimator = np.power(Op, 2.0) - 2*mean_force*Op + np.power(mean_force, 2.0)
            results.update_data_babf(variance_estimator,'sum_w_A2xU2_dl',block=True)
                 
            results.update_data_babf(conditional_pi_lambda,'sum_w_A',block=True)
            results.data_babf_block['pc_lam_q'] = conditional_pi_lambda
            results.data_babf_block['pc_lam_q'] = conditional_pi_lambda           
    
        if self.Jarzynski : 
            work_jar = self.compute_deltaW(self.force,(self.x - self.previous_x))
            self.update_Jarzynski(results, work_jar)

        return results

    def update_Jarzynski(self, results : ResultsBABF, 
                         jarzynski_work : float) -> None : 
        """Update Jarzynski correction for constrained dynamics

        Parameters
        ----------

        results : ResultsBABF 
            Object containing all free energy data

        
        jarzynski_work : float
            Work due to constrained forces
        """           

        """Jarzynski update !"""
        jar_work_lambda = results.Jarzynski_work_lambda(jarzynski_work)
        results.update_data_babf(jar_work_lambda*results.data_babf_block['pc_lam_q'],'sum_jarzynski_work')

        return

    def EvaluateEffectiveForces(self, results : ResultsBABF) -> np.ndarray : 
        """Compute effective force to propagate stochastic dynamics 
        
        \mathbb{F}(q) = \int_{0}^{1} \Nabla_{q} U_{\lambda}(q) p(q|\lambda) d \lambda

        Parameters
        ----------

        results : ResultsBABF 
            Object containig all free energy data

        Returns 
        -------

        np.ndarray
            Effective force to apply for stochastic dynamics

        """
        forces_lammps = self.evaluate_lammps_forces()
        if self.block :
            forces_lammps += self.evaluate_constrained_forces()

        forces_reference = self.evaluate_reference_forces()
        effective_forces = results.evaluate_effective_forces_dynamic(forces_reference,forces_lammps)
        self.force = effective_forces  
        return self.force

    """UPDATE SCHEM"""
    def update_step(self,results: ResultsBABF) -> ResultsBABF:
        """Unitary step for stochastic dynamics with update of results objects
        Paramters 
        ---------
        
        results: ResultsBABF 
            ResultsBABF object to update

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

        dictionnary_dynamic = {"ImpliciteVerlet": lambda res : Implicit_Verlet_dynamic(self,res,temperature,delta_t),
                               "BAOAB": lambda res : BAOAB_scheme(self,res, temperature, delta_t),
                               "OverdampedLangevin": lambda res : Overdamped_Langevin_dynamic(self,res, temperature, delta_t)}

        up_worker, up_results = dictionnary_dynamic[dynamic](results)
        return up_results

