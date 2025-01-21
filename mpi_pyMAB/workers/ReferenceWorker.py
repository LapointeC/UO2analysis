from __future__ import annotations
import numpy as np
import os
from typing import Any, List
from ..parsers.BaseParser import BaseParser
from mpi4py import MPI
from .BaseWorker import BaseWorker

class ReferenceWorker(BaseWorker):
    """LAMMPS worker for PAFI, inheriting BaseWorker

        Initialization routine:
        - runs "Input" script, loading first configuration
        - extracts cell data
        - defines various functions wrapping computes etc.
        - defines initialize_hyperplane function to set plane

        Parameters
        ----------
        
        comm : MPI.Intracomm
            MPI communicator
        
        parameters : PAFIParser
            Predefined or custom PAFIParser object
        
        worker_instance : int
            unique worker rank
        """
    def __init__(self, comm: MPI.Intracomm, 
                 parameters: BaseParser, worker_instance: int,
                 rank: int, roots: List[int]) -> None:
        super().__init__(comm, parameters, worker_instance, rank, roots)
        
        self.name = "ReferenceWorker"
        self.parameters = parameters
        
        parameters = lambda k: self.parameters(k)
        self.reference_model = parameters('ReferenceModel')
        self.constrained_potential = parameters('ConstrainedPotential')
        self.species = parameters('Species')
        self.build_mass_array()
        self.reference_energy : float = 0.0

    def set_reference_energy(self, reference_energy : float) -> None : 
        """Set the reference energy for reference model"""
        self.reference_energy = reference_energy

    def build_mass_array(self) -> None :
        """Check the number of species to parametrise reference model"""
        if len(self.species) > 1 :
            print('Not yet implemented !')
        else : 
            self.species : List[str] = [ self.species[0] for _ in range(self.x_reference.shape[0])]
            self.mass_array : np.ndarray = np.diag(np.array([ self.mass_dictionary[self.species[0]] for _ in range(self.x_reference.shape[0])]))

    """REFERENCE MODEL, only einstein model for the moment..."""
    def evaluate_einstein_energy(self) -> float : 
        """Evaluate reference Einstein energy : E_{eins}(q) = 1/2 \sum_{i=1}^N m_i*\omega^2 \Vert q_i - q_{i,ref} \Vert^2

        Returns 
        ------- 
        
        float
            reference Einstein energy : E_{eins}(q) = 1/2 \sum_{i=1}^N m_i*\omega^2 \Vert q_i - q_{i,ref} \Vert^2
        """ 
        omega_einstein = self.reference_model['omega']
        energy_einstein = self.reference_energy
        energy_einstein +=  np.sum([ 0.5*self.mass_array[k]*(omega_einstein**2)*np.linalg.norm((self.x - self.x_reference)[k,:])**2 
                                    for k in range(self.x_reference.shape[0]) ])
        return energy_einstein
            
    def evaluate_einstein_forces(self) -> np.ndarray : 
        """Evaluate reference Einstein forces : f_{eins}(q) 1/2 \sum_{i=1}^N m_i*\omega^2 \Vert q_i - q_{i,ref} \Vert^2

        Returns 
        ------- 
        
        float
            reference Einstein energy : E_{eins}(q) = 1/2 \sum_{i=1}^N m_i*\omega^2 \Vert q_i - q_{i,ref} \Vert^2
        """ 
        omega_einstein = self.reference_model['omega']
        return -(omega_einstein**2)*self.mass_array@(self.x - self.x_reference)

    def evaluate_reference_energy(self) -> float :
        """Evaluate energy for reference model : E_{ref}(q)
        Add something about options...

        Returns
        -------
        
        float 
            Energy for reference system E_{ref}(q)
        """
        if self.reference_model['model'] == 'Einstein' : 
            return self.evaluate_einstein_energy()

        if self.reference_model['model'] == 'Harmonic' :     
            print('Not yet implemented !')
            exit(0)

        if self.reference_model['model'] == 'Morse' : 
            print('Not yet implemented !')
            exit(0)

    def evaluate_reference_forces(self) -> np.ndarray :
        """Evaluate reference forces : f_{ref}(q)
        Add something about options...

        Returns
        -------
        
        np.ndarray
            Forces for reference system f_{ref}(q)
        """
        if self.reference_model['model'] == 'Einstein' : 
            return self.evaluate_einstein_forces()

        if self.reference_model['model'] == 'Harmonic' :     
            print('Not yet implemented !')
            exit(0)

        if self.reference_model['model'] == 'Morse' : 
            print('Not yet implemented !')
            exit(0)

    """Temperature estimation from reference energy"""
    def evaluate_potential_temperature_einstein(self) -> float : 
        """Evaluate potential temperature : T_{eins} =  E_{eins}(q)/(3NkB)

        Returns 
        ------- 
        
        float
            potential temperature : T_{eins} =  E_{eins}(q)/(3NkB)
        """        
        omega_einstein = self.reference_model['omega']
        energy_einstein =  np.sum([ 0.5*self.mass_array[k]*(omega_einstein**2)*np.linalg.norm((self.x - self.x_reference)[k,:])**2 
                for k in range(self.x_reference.shape[0]) ])
        return energy_einstein/(3*self.x_reference.shape[0]*self.kB)

    def evaluate_potential_temperature(self) -> float :
        """Evaluate potential temperature based on equipartition theorem
        Add something about options...

        Returns
        -------
        
        float 
            potential temperature
        """
        if self.reference_model['model'] == 'Einstein' : 
            return self.evaluate_potential_temperature_einstein()

        if self.reference_model['model'] == 'Harmonic' :     
            print('Not yet implemented !')
            exit(0)

        if self.reference_model['model'] == 'Morse' : 
            print('Not yet implemented !')
            exit(0)

    """CONSTRAINED FORCES EXPRESSION"""
    def evaluate_constrained_forces(self) -> np.ndarray : 
        """Evaluate constrained external forces : f_{c}(q)
        Add something about options...

        Returns
        -------
        
        np.ndarray
            Constrained external forces on the system f_{c}(q)
        """
        if self.constrained_potential['model'] == 'Standard' : 
            return self.evaluate_standard_constrained_forces()

        else :     
            print('Not yet implemented !')
            exit(0) 

    def phi(self,x : float) -> float : 
        """Evaluate phi function used for standard constrained forces
        
        Parameters
        ----------
        
        x : float 

        Returns
        -------
        
        float
            phi(x)

        """
        return np.max([0.0, np.reciprocal(1.0 + np.cosh(x)) + np.reciprocal(1.0 + np.cosh(1.0)) ])

    def evaluate_standard_constrained_forces(self) -> np.ndarray : 
        """Evaluate standard constrained standard forces 
        
        Returns 
        -------
        
        np.ndarray 
            f_c(q)
        """
        delta_x = self.x - self.x_reference
        C, delta, Rc = self.constrained_potential['C'], self.constrained_potential['delta'], self.constrained_potential['Rc']
        scaling_phi = self.phi((np.linalg.norm(delta_x) - Rc - delta)/delta)
        return - (delta_x*C)*scaling_phi/(2.0*delta*np.linalg.norm(delta_x))


    """CONSTRAINED ENERGY EXPRESSION"""
    def primitive_phi(self,x : float) -> float : 
        """Evaluate primitive of phi function

        Returns
        -------
        
        float 
            \int_0^x phi(x') dx'
        """
        if self.phi(x) > 0.0 :
            return -2.0/(np.exp(x)+1.0) + x*np.reciprocal(1.0 + np.cosh(1.0))
        else : 
            return 0.0

    def evaluate_standard_constrained_energy(self) -> float : 
        """Evaluate standard constrained standard forces 
        
        Returns 
        -------
        
        np.ndarray 
            f_c(q)
        """
        delta_x = self.x - self.x_reference
        C, delta, Rc = self.constrained_potential['C'], self.constrained_potential['delta'], self.constrained_potential['Rc']      
        scaling_primitive_phi = self.primitive_phi((np.linalg.norm(delta_x) - Rc - delta)/delta)
        return C*scaling_primitive_phi/2.0


    def evaluate_constrained_forces(self) -> float : 
        """Evaluate constrained external energy : U_{c}(q)
        Add something about options...

        Returns
        -------
        
        float
            Constrained external energy on the system U_{c}(q)
        """
        if self.constrained_potential['model'] == 'Standard' : 
            return self.evaluate_standard_constrained_energy()

        else :     
            print('Not yet implemented !')
            exit(0)


    """EXTERNAL WORK COMPUTATION FOR JARZYNSKI EQUALITY"""
    def compute_deltaW(self,forces : np.ndarray, displacements : np.ndarray) -> float : 
        """Compute the variation of work : \delta W
        Parameters
        ----------

        forces : np.ndarray 
            Forces applyied on the system : f_{c}(q)
        displacement : np.ndarray 
            displacements resulting from forces application : \delta q

        Returns
        -------
        float
            \detla W = f_{c}(q) \cdot \detla q
        """       
       
        return np.trace(forces.T@displacements)