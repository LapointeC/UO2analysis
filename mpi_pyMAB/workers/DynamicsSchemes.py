import numpy as np

from mpi4py import MPI
from typing import List, Tuple
from ..results.ResultsBABF import ResultsBABF
from .BABFWorker import BABFWorker

"""DYNAMICS DEFINITIONS"""
def Implicit_Verlet_dynamic(worker : BABFWorker,
                            results : ResultsBABF, 
                            temperature : float, 
                            delta_t : float) -> Tuple[BABFWorker,ResultsBABF]:
    
    """Perform midpoint Euler-Verlet for overdumped dynamic
    ref : Free energy computations a mathematical perspective, p. 94

    Parameters
    ----------
    
    worker : BABFWorker
        Object containig BABF worker for LAMMPS calculations

    results : ResultsBABF 
        Object containing all free energy data and methods
    
    temperature : float 
        Simulation temperature
    
    delta_t : float 
        Time step for stochastic dynamic 

    Results
    -------
    
    BABFWorker
        Updated worker object

    ResultsBABF
        Updated free energy object
    """
    
    gamma = 1/(delta_t*100)
    
    """v_n -> v_n+1/4"""
    worker.v = worker.v*np.array(np.exp(-gamma*0.5*delta_t)) + worker.sampling_Wigner_process(temperature,0.5*delta_t,worker.v.shape)
    
    """v_n+1/4 -> v_n+1/2"""
    worker.v = worker.v + worker.force*0.5*delta_t
    
    """q_n -> q_n+1"""
    worker.previous_x = worker.x
    worker.x = worker.x + worker.v*delta_t

    """update positions"""
    worker.centering_system()
    worker.x = worker.pbc(worker.x)
    worker.x_reference = worker.pbc(worker.x_reference)
    worker.scatter('x',worker.x)
    
    """update forces"""
    effective_forces = worker.EvaluateEffectiveForces(results)
    
    """update result object"""
    results = worker.update_free_energy_quantities(results,
                                                   temperature)
    
    """p_n+1/2 -> p_n+3/4"""
    worker.v = worker.v + effective_forces*0.5*delta_t
    
    """p_n+3/4 -> p_n+1"""
    worker.v = worker.v*np.exp(-gamma*0.5*delta_t) + worker.sampling_Wigner_process(temperature,0.5*delta_t,worker.v.shape)
    
    return worker, results

def BAOAB_scheme(worker : BABFWorker,
                results : ResultsBABF, 
                temperature : float , 
                delta_t : float) -> Tuple[BABFWorker,ResultsBABF] :
    """Perform BAOAB predictor-corrector schem
    ref : Free energy computations a mathematical perspective, p. 94
    
    Parameters
    ----------
    
    worker : BABFWorker
        Object containig BABF worker for LAMMPS calculations

    results : ResultsBABF 
        Object containing all free energy data and methods
    
    temperature : float 
        Simulation temperature
    
    delta_t : float 
        time step for stochastic dynamic 

    Results
    -------
    
    BABFWorker
        Updated worker object
     
    ResultsBABF
        Updated free energy object
    """
    gamma = 1/(delta_t*100)

    """v^_n-1/2 -> v_n"""
    worker.v = worker.v + worker.mass_array@worker.force*0.5*delta_t

    """v_n -> v_n+1/2 (update of forces)"""
    effective_forces = worker.EvaluateEffectiveForces(results)     
    worker.v = worker.v + worker.mass_array@effective_forces*0.5*delta_t

    """update result object"""
    results = worker.update_free_energy_quantities(results,
                                                 temperature)

    """q_n -> q_n+1/2"""
    worker.previous_x = worker.x
    worker.x = worker.x + worker.v*0.5*delta_t

    """p_n+1/2 -> p^_n+1/2"""
    worker.v = np.exp(-gamma*delta_t)*worker.v + np.sqrt(1.0-np.exp(-2*gamma*delta_t))*worker.sampling_Wigner_process(temperature,0.5*delta_t,worker.v.shape)

    """q_n+1/2 -> q_n+1"""
    worker.previous_x = worker.x
    worker.x = worker.x + worker.v*0.5*delta_t

    """update in lammps !"""
    worker.centering_system()
    worker.x = worker.pbc(worker.x)
    worker.x_reference = worker.pbc(worker.x_reference)
    worker.scatter('x',worker.x)

    return worker, results


def Overdamped_Langevin_dynamic(worker :BABFWorker,
                                results : ResultsBABF, 
                                temperature : float, 
                                delta_t : float) -> Tuple[BABFWorker,ResultsBABF] :
    """Perform Langevin scheme for overdumped dynamic
    ref : Free energy computations a mathematical perspective, p. 99
    
    Parameters
    ----------
    
    worker : BABFWorker
        Object containig BABF worker for LAMMPS calculations
    
    results : ResultsBABF 
        Object containing all free energy data and methods
    
    temperature : float 
        Simulation temperature
    
    delta_t : float 
        time step for stochastic dynamic 
    
    Results
    -------
    
    BABFWorker
        Updated worker object
      
    ResultsBABF
        Updated free energy object
    """

    """Langevin dynamic update overdumped"""
    effective_forces = worker.EvaluateEffectiveForces(results)
    
    worker.previous_x = worker.x
    worker.x = worker.x + effective_forces*delta_t + worker.sampling_Wigner_process(temperature,delta_t,worker.x.shape)

    """update result object"""
    results = worker.update_free_energy_quantities(results,
                                                   temperature)

    """update in lammps !"""
    worker.centering_system()
    worker.x = worker.pbc(worker.x)
    worker.x_reference = worker.pbc(worker.x_reference)
    worker.scatter('x',worker.x)

    return worker, results


"""THERMALISATION STEPS"""
def thermalisation_steps(worker : BABFWorker,
                         results:ResultsBABF) -> Tuple[BABFWorker,ResultsBABF] :
    """Perform thermalisation step without updating free energy estimator
    
    Parameters
    ----------

    worker : BABFWorker
        Object containig BABF worker for LAMMPS calculations

    results : ResultsBABF 
        Object containing all free energy data and methods   

        
    Returns
    -------
    
    BABFWorker
        Updated worker object

    ResultsBABF
        Updated free energy object    
    """
    parameters = lambda k: worker.parameters(k)
    thermalisation_step = parameters('ThermalisationStep')    
    if worker.rank == 0 : 
        print('... Starting thermalisation procedure ... (%1.1e steps)'%(thermalisation_step))

    for _ in range(thermalisation_step) : 
        results = worker.update_step(results)
        
    if worker.rank == 0 : 
        print('... Finishing thermalisation procedure ...')

    return worker, results