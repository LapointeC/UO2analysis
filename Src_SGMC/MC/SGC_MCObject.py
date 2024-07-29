import numpy as np
from .LAMMPSWorker import LAMMPSWorker
from numpy.random import uniform, randint
from typing import List, Dict, TypedDict, Tuple 

from mpi4py import MPI

class Average(TypedDict) : 
    """Little class to easily compute average !"""
    sum_concentration : np.ndarray
    sum_square_concentration : np.ndarray
    compt : float
    
    average_concentration : np.ndarray
    variance_concentration : np.ndarray
    

class SGCMC : 
    def __init__(self, mu_array : np.ndarray, n_species_array : np.ndarray, lammps_worker : LAMMPSWorker, writing_dir : str, equiv_mu : List[float] = [1.0,2.0]) -> None :

        self.worker = lammps_worker
        self.kB = 8.617333262e-5
        self.mu_array = mu_array
        self.delta_N_array = np.zeros(len(self.mu_array))
        self.Natom = self.worker.get_natoms()
        self.n_species_array = n_species_array

        self.equiv_mu = [int(id_mu) for id_mu in equiv_mu]
        self.writing_dir = writing_dir
        # debug !
        #np.random.seed(123456)

        self.old_energy = None
        self.average_results : Average = {'compt':0.0,
                                          'sum_concentration':np.zeros(len(self.mu_array)),
                                          'sum_square_concentration':np.zeros(len(self.mu_array)),
                                          'variance_concentration':np.zeros(len(self.mu_array))}

    def change_species(self, id_atom : int, species : str | int) -> None : 
        """Change the species in Lammps system
        
        Parameters:
        -----------

        id_atom : int 
            Id of atom to swap 

        species : str | int 
            New species of the atom id_atom 
        """
        type_array = self.worker.L.gather('type',0,1)
        #type_array[id_atom] = species
        type_array[id_atom] = self.equiv_mu[species-1]
        self.worker.L.scatter('type',0,1,type_array)
        return 

    def compute_deltaE(self, id_atom : int, new_species : str | int) -> Tuple[float,int|str] : 
        """Compute the energy difference between previous system and proposal for Metroplis
        
        Parameters:
        -----------

        id_atom : int 
            Id of the atom to swap

        new_species : str | int
            New species after the swap


        Returns: 
        --------

        float 
            Energy difference : E_new - E_old 

        int|str 
            Old species of the atom number id_atom
        """
        #first we extract the old energy !

        if self.old_energy is None : 
            old_energy = self.worker.get_energy()
        else : 
            old_energy = self.old_energy

        #gather type ! 
        type_array = self.worker.L.gather('type',0,1)
        old_species_t = type_array[id_atom]

        old_species = self.equiv_mu.index(old_species_t) + 1

        if new_species == old_species : 
        #if new_species == old_species :
            return 0.0, old_species

        else :
            #scatter new type
            type_array[id_atom] = self.equiv_mu[new_species-1]
            #type_array[id_atom] = new_species
            self.worker.L.scatter('type',0,1,type_array)

            #new energy
            self.worker.run_commands("run 0 post no")
            new_energy = self.worker.get_energy()

            #update old energy
            self.old_energy = new_energy

            return new_energy - old_energy, old_species
    
    def trial_step_SGCMC(self, id_atom : int, new_species : str | int, temperature : float) -> float : 
        """Perform the trial step of SGCMC
        
        Parameters:
        -----------

        id_atom : int 
            Id of the atom to perform extended Metropolis criterion

        new_species : str 
            New species for this atom 

        temperature : float 
            Temperature for the extended Metropolis criterion

        Returns:
        --------

        float 
            Acceptance result : 1 if accepted 0 otherwise
        """
        deltaE, old_species = self.compute_deltaE(id_atom, new_species)

        #change delta N for acceptance
        tmp_delta_N_array = self.delta_N_array
        tmp_delta_N_array[old_species-1] += -1.0
        tmp_delta_N_array[new_species-1] += 1.0
        delta_mu = np.sum(self.mu_array*tmp_delta_N_array)

        acceptance_criteria =  (- deltaE + delta_mu)/(self.kB*temperature)

        self.delta_N_array = np.zeros(len(self.mu_array))

        if self.worker.rank == 0 :
            noise = np.log(uniform(0.0,1.0))
        else : 
            noise = None 

        noise = self.worker.comm.bcast(noise, root=0)
        #if self.worker.rank == 0 : 
        #    print(acceptance_criteria,np.exp(acceptance_criteria), np.exp(noise), deltaE, delta_mu,'MC step')

        if acceptance_criteria >= noise :
            #swap is accpeted ! 
            self.n_species_array[old_species-1] += -1.0
            self.n_species_array[new_species-1] += 1.0
            return 1.0
        
        else :
            #go back to the previous config !
            self.change_species(id_atom, old_species)
            self.old_energy += -deltaE  
            return 0.0

    # DEBUG
    #def get_number_of_atoms_each_species(self) -> Tuple[np.ndarray] : 
    #    type_array = self.worker.gather('type')
    #    array_id, array_nb = np.unique(type_array.flatten(), return_counts=True)
    #    return array_id, array_nb

    def perform_SGCMC(self, fraction_swap : float, temperature : float) -> float : 
        """Perform SGCMC on all system 
        
        Parameters:
        -----------

        fraction_swap : float 
            percentage of the system to perform swap

        temperature : float 
            Temperature for extended Metropolis criterion
        
        Returns:
        --------

        float 
            Average acceptance ratio 
        """

        sum_acceptance = 0.0
        for _ in range(int(fraction_swap*self.Natom)) : 
            if self.worker.rank == 0 :
                id_to_test = randint(0,self.Natom)
                species_to_test = randint(1,len(self.mu_array)+1)
            else : 
                id_to_test, species_to_test = None, None 

            id_to_test = self.worker.comm.bcast(id_to_test, root=0)
            species_to_test = self.worker.comm.bcast(species_to_test, root=0)

            sum_acceptance += self.trial_step_SGCMC(id_to_test, species_to_test, temperature)

            #compute average 
            self.average_results['compt'] += 1.0
            self.average_results['sum_concentration'] += np.array(self.n_species_array)/np.sum(self.n_species_array)
            self.average_results['sum_square_concentration'] += np.array(self.n_species_array)**2/(np.sum(self.n_species_array)**2)

        return sum_acceptance/int(fraction_swap*self.Natom) 

    def perform_lammps_script(self, script : List[str]|str) -> None :
        """Run Lammps commands
        
        Parameters:
        -----------

        script : List[str]|str
            Lammps commands to execute 
        """
        self.worker.run_commands(script)
        return 
    
    def compute_average(self) -> Tuple[np.ndarray,np.ndarray] : 
        """Compute average of concentration for all species in the system
        
        Returns:
        --------

        np.ndarray 
            Average of concentration for each element 

        np.ndarray 
            Variance of concentration for each element 
        """
        self.average_results['average_concentration'] = self.average_results['sum_concentration']/self.average_results['compt']
        self.average_results['variance_concentration'] = self.average_results['sum_square_concentration']/self.average_results['compt'] - self.average_results['average_concentration']**2
        return self.average_results['average_concentration'], self.average_results['variance_concentration']
    
    def dump_configuration(self, name_configuration : str) -> None :
        """Dump the configuration 
        
        Parameters:
        -----------

        name_configuration : str 
            Name of the configuration to dump
        """
        self.worker.run_commands(f'write_data {self.writing_dir}/swap_mu{name_configuration}.lmp')
        return 

    def write_restart(self) -> None :
        #TO DO, WRITE A RESTARTER...
        pass


class VC_SGCMC(SGCMC) : 
    def __init__(self, mu_array : np.ndarray, concentration_array : np.ndarray, n_species_array : np.ndarray, kappa : float, lammps_worker : LAMMPSWorker, writing_dir : str) -> None :

        self.kappa = kappa
        self.concentration_array = concentration_array
        super().__init__(mu_array, n_species_array, lammps_worker, writing_dir)

    
    def trial_step_VC_SGCMC(self, id_atom : int, new_species : str | int, temperature : float) -> float : 
        """Perform the trial step of SGCMC
        
        Parameters:
        -----------

        id_atom : int 
            Id of the atom to perform extended Metropolis criterion

        new_species : str 
            New species for this atom 

        temperature : float 
            Temperature for the extended Metropolis criterion

        Returns:
        --------

        float 
            Acceptance result : 1 if accepted 0 otherwise
        """
        deltaE, old_species = self.compute_deltaE(id_atom, new_species)
        if old_species == new_species and abs(deltaE) > 1e-5 : 
            self.change_species(id_atom, old_species)
            self.old_energy += - deltaE
            return 0.0

        #change delta N for acceptance
        tmp_delta_N_array = self.delta_N_array.copy()
        tmp_delta_N_array[old_species-1] += -1.0
        tmp_delta_N_array[new_species-1] += 1.0
        delta_mu = np.sum(self.mu_array*tmp_delta_N_array)

        #Concentration variance constraint
        Natoms = np.sum(self.n_species_array)
        delta_concentration = (tmp_delta_N_array - self.delta_N_array)/Natoms
        average_concentration = (2*np.array(self.n_species_array) + tmp_delta_N_array)/(2*Natoms)
        concentration_constrained = 2*self.kappa*Natoms**2*np.sum(delta_concentration*(average_concentration-self.concentration_array))

        #CHECK SIGNS !
        acceptance_criteria = - (deltaE - delta_mu - concentration_constrained)/(self.kB*temperature)
        self.delta_N_array = np.zeros(len(self.mu_array))

        #swap is accpeted ! 
        if self.worker.rank == 0 :
            noise = np.log(uniform(0.0,1.0))
        else : 
            noise = None 

        noise = self.worker.comm.bcast(noise, root=0)

        if acceptance_criteria >= noise :
            self.n_species_array[old_species-1] += -1.0
            self.n_species_array[new_species-1] += 1.0
            return 1.0
        
        else :
            #go back to the previous config !
            self.change_species(id_atom, old_species)
            self.old_energy += - deltaE
            return 0.0

    def perform_VC_SGCMC(self, fraction_swap : float, temperature : float) -> float : 
        """Perform SGCMC on all system 
        
        Parameters:
        -----------

        fraction_swap : float 
            percentage of the system to perform swap

        temperature : float 
            Temperature for extended Metropolis criterion
        
        Returns:
        --------

        float 
            Average acceptance ratio 
        """
        sum_acceptance = 0.0
        for _ in range(int(fraction_swap*self.Natom)) : 
            if self.worker.rank == 0 :
                id_to_test = randint(0,self.Natom)
                species_to_test = randint(1,len(self.mu_array)+1)
            else : 
                id_to_test, species_to_test = None, None 

            id_to_test = self.worker.comm.bcast(id_to_test, root=0)
            species_to_test = self.worker.comm.bcast(species_to_test, root=0)

            sum_acceptance += self.trial_step_VC_SGCMC(id_to_test, species_to_test, temperature)

            #compute average 
            self.average_results['compt'] += 1.0
            self.average_results['sum_concentration'] += np.array(self.n_species_array)/np.sum(self.n_species_array)
            self.average_results['sum_square_concentration'] += np.array(self.n_species_array)**2/(np.sum(self.n_species_array)**2)
        

        return sum_acceptance/int(fraction_swap*self.Natom) 
    
    def dump_configuration(self, name_configuration : str) -> None :
        """Dump the configuration 
        
        Parameters:
        -----------

        name_configuration : str 
            Name of the configuration to dump
        """
        self.worker.run_commands(f'write_data {self.writing_dir}/VC_SGCMC{name_configuration}.lmp')
        return 

    def write_restart(self) -> None :
        #TO DO, WRITE A RESTARTER...
        pass