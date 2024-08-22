from __future__ import annotations

import os, sys
import numpy as np
import ase 
from ase import Atom, Atoms
from ase.calculators.vasp import Vasp
from typing import List, Dict, Tuple

from ToyPotential import CoulombianCalc

class ARTWorker : 
    def __init__(self, system : Atoms, list_idx : List[str], calc_vasp : Vasp, rcut : float, delta_xi : float, max_disp : List[float], alpha_rand : float, delta_t_relax : float, dumping : float, lambda_c : float, mu : float, tol_force : float, debug : bool = False) -> None : 
        self.ini_system = system
        self.rcut = rcut
        self.delta_xi = delta_xi
        self.max_disp = max_disp
        self.alpha_rand = alpha_rand
        self.delta_t_relax = delta_t_relax
        self.dumping = dumping
        self.calc_vasp = calc_vasp
        self.list_idx = list_idx
        self.lambda_c = lambda_c
        self.mu = mu 
        self.tol_force = tol_force
        self.iteration = 1.0
        self.debug = debug

    def get_energy_toy(self, system : Atoms) -> float : 
        """Compute toy energy of the system
        
        Paramters
        ---------

        system : Atoms 
            ASE object for the system 

        Returns 
        -------

        float 
            Toy energy of the system
        """
        
        ToyCalc = CoulombianCalc(system) 
        return ToyCalc.compute_coulombian_energy()

    def get_forces_toy(self, system : Atoms) -> np.ndarray : 
        """Compute vasp forces of the system
        
        Paramters
        ---------

        system : Atoms 
            ASE object for the system 

        Returns 
        -------

        np.ndarray 
            Toy forces of the system
        """
        
        ToyCalc = CoulombianCalc(system) 
        return ToyCalc.compute_coulombian_forces()  

    def get_energy_vasp(self, system : Atoms) -> float : 
        """Compute vasp energy of the system
        
        Paramters
        ---------

        system : Atoms 
            ASE object for the system 

        Returns 
        -------

        float 
            VASP energy of the system
        """
        
        system.calc = self.calc_vasp 
        return system.get_total_energy()

    def get_forces_vasp(self, system : Atoms) -> np.ndarray : 
        """Compute vasp forces of the system
        
        Paramters
        ---------

        system : Atoms 
            ASE object for the system 

        Returns 
        -------

        np.ndarray 
            VASP forces of the system
        """
        
        system.calc = self.calc_vasp 
        return system.get_forces()   

    def center_calculator(self, system : Atoms) -> np.ndarray : 
        """build the center of mass of subset of idx list_idx
        
        Parameters
        ----------
        system : Atoms 
            Ase object containing all the system 

        Returns 
        -------
        np.ndarray 
            center of gravity of atoms of interest
        """
        sum_mass = np.sum([ system[idx].mass for idx in self.list_idx ])
        tmp_array = np.zeros(3)
        for idx in self.list_idx : 
            tmp_array += system[idx].mass*system[idx].position
        return tmp_array/sum_mass

    def extract_minimum_lambda_eigen_vector(self, eigen_val : np.ndarray, eigen_vect : np.ndarray) -> Tuple[float, np.ndarray] : 
        """"Extract the eigen vector corresponding to the minimum eigenvalue in the classical 
        numpy array form (N,3)

        Parameters
        ----------
        eigen_val : np.ndarray 
            array of eigein values
        eigen_vect : np.ndarray 
            array of corresponding eigen vector

        Returns
        -------
        np.ndarray 
            eigen vector corresponding to the minimum eigenvalue (in shape (N,3)) and normalised !
        """
        min_idx = np.argmin(eigen_val)
        eigen_vector =  eigen_vect[:,min_idx].reshape(int(len(eigen_val)/3),3)
        return eigen_val[min_idx], eigen_vector/np.linalg.norm(eigen_vector)


    def Hessian_evaluation(self, left_force_xi : np.ndarray,right_force_xi : np.ndarray, masse : float) -> np.ndarray :
        """Build the \sum_{j=1}^3 \Delta E_{ij} e_i \otimes e_j components for Hessian matrix
        
        Parameters 
        ----------
        left_force_xi : np.ndarray 
            left force array

        right_force_xi : np.ndarray 
            right force array 
        
        masse : float 
            mass (sqrt{m_i m_j})

        Returns 
        -------
        np.ndarray 
            \sum_{j=1}^3 \Delta E_{ij} e_i \otimes e_j components for Hessian matrix
        """
        return np.asarray([(right_force_xi[k] - left_force_xi[k])/(2*self.delta_xi*masse) for k in range(len(left_force_xi))])

    def Hessian_diag(self ,system : Atoms, center_hessian : np.ndarray) -> Tuple[float,np.ndarray,np.ndarray] :
        """Perform dynamical matrix sampling and then diagonalisation
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system 
        center_hessian : np.ndarray 
            gravity center array of atoms of interest

        Returns 
        -------
        Delta_hessian_norm : float 
            Relative symmetric Frobenuis norm of dynamical matrix ( \frac{\Vert H - H^T  \Vert_F}{\Vert H \Vert_F} )
        eigen_values : np.ndarray 
            array of eigen values of dynamical matrix
        eigen_vectors : np.ndarray 
            array of corresponding eigen vector of dynamical matrix
        """
        disp_x = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]

        """Hessian building"""
        Hessian_matrix = np.zeros((3*system.get_number_of_atoms(),3*system.get_number_of_atoms()), dtype=float)
        for id_i in range(system.get_global_number_of_atoms()) :
            for id_j in range(system.get_global_number_of_atoms()) :
                """condition on centered system for Hessian calculation !"""
                if np.linalg.norm(system[id_i].position - center_hessian) < 2.0*self.rcut and np.linalg.norm(system[id_j].position - center_hessian) < 2.0*self.rcut : 
                    mass_ij = np.sqrt(system[id_i].mass*system[id_j].mass)
                    for x_alpha, disp_alpha in enumerate(disp_x):
                        two_points_forces_list_xi_k = []
                        for signe in [-1,1]:                       
                            """Build perturbation and launch vasp calculation !"""
                            tmp_system = system.copy()
                            tmp_system.positions[id_i,:] += disp_alpha*self.delta_xi*signe
                            if self.debug : 
                                force_delta_ialpha_to_jbeta = self.get_forces_toy(tmp_system)[id_j,:]
                            else :
                                force_delta_ialpha_to_jbeta = self.get_forces_vasp(tmp_system)[id_j,:]
                            two_points_forces_list_xi_k.append(force_delta_ialpha_to_jbeta)

                        for x_beta, disp_beta in enumerate(disp_x) :
                            Hessian_ialpha_to_jbeta = np.dot(self.Hessian_evaluation(two_points_forces_list_xi_k[1],two_points_forces_list_xi_k[0],mass_ij),disp_alpha)
                            Hessian_matrix[3*int(id_i)+x_alpha,3*int(id_j)+x_beta] = Hessian_ialpha_to_jbeta
                else : 
                    continue 

        Delta_hessian_norm = np.linalg.norm(Hessian_matrix-Hessian_matrix.T)/np.linalg.norm(Hessian_matrix)
        Hessian_matrix = 0.5*(Hessian_matrix + Hessian_matrix.T)

        eigen_values, eigen_vectors = np.linalg.eigh(Hessian_matrix,UPLO='L')

        # conversion (1e-1 rad.PHz)^2 -> (rad.THz)^2
        eigen_values *= 1.0e4
        return Delta_hessian_norm, eigen_values, eigen_vectors

    def compute_moment_inertia_tensor(self, system : Atoms, center : np.ndarray) -> Tuple[np.ndarray,np.ndarray] : 
        """Compute the diagonal terms and full moment inertia tensor
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system        
        center : np.ndarray 
            mass center of atoms of interest

        Returns
        -------
        np.ndarray
            diagonal components of inertia moment tensor

        np.ndarray
            Inertia moment tensor
        """
        sub_set_atom = [system[idx] for idx in self.list_idx]
        sum_m_norm_array = np.sum([  at.mass*np.linalg.norm(at.position - center)**2 for at in sub_set_atom ])
        sum_pos_array = np.array( [ at.position - center for at in sub_set_atom ] )
        sum_m_pos_array = np.array( [ at.mass*(at.position - center) for at in sub_set_atom ] )
        
        inertia_tensor_moment : np.ndarray = sum_m_norm_array*np.eye(3) - sum_m_pos_array.T@sum_pos_array
        return np.array([inertia_tensor_moment[i,i] for i in range(3)]), inertia_tensor_moment


    def elliptic_deformation(self, system : Atoms, center : np.ndarray, rcut_def : float, max_displacement : List[float] = [0.1,0.1,0.1], noise : bool = False) -> Atoms :
        """Apply elliptic deformation on the system depending on inertia moment tensor
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system        
        center : np.ndarray 
            mass center of atoms of interest       
        rcut_def : float
            rcut for deformation of atoms
        max_displacement : float 
            maximum displacement allowed per direction


        Returns
        -------
        Atoms 
            Updated ASE object after deformations
        """
        
        def f(x : float, C : float, r : float) -> float :
            """Apply the pseudo spherical deformation ...
            
            Parameters
            ----------

            x : float 
                Distance to the deformation center 
            C : float  
                Scaling of deforamation 
            r : float 
                rcut for deformation 

            Returns
            -------

            float 
                pseudo spherical deforamtion
            
            """
            if x > r : 
                return 0.0
            elif abs(x) < 1e-2 :
                return 0.0
            else :
                return C/x 

        def gaussian_noise(sigma : float) -> float : 
            """Apply 0D gaussian noise 
            
            Parameters 
            ----------
            sigma
                Standard deviation of the gaussian 

            Returns 
            -------
            float 
                Given realisation of the random gaussian variable
            """
            return np.random.normal(0.0,sigma)

        def constraint(displacement : float, max_displacement : float) -> float :
            """Constrained displacements wrt with maximum displacement 
            
            Parameters
            ----------

            displacement : float 
                Value of displacement for a given direction

            max_displacement : float 
                Value of maximum displacement for a given direction
            

            Returns
            -------

            float 
                Constrained displacement 
            """
            if displacement < 0.0 : 
                return max([-max_displacement, displacement])
            else : 
                return min([max_displacement, displacement])

        tensor_moment, _ = self.compute_moment_inertia_tensor(system, center)
        for idx in self.list_idx :
            """Compute the displacement along the alpha direction..."""
            for alpha in range(len(system[idx].position)) :
                centered_alpha = system[idx].position[alpha]-center[alpha]
                sum_tensor = np.sum(tensor_moment)
                if noise :
                    displacement_alpha = constraint(system[idx].position[alpha]*f(centered_alpha,(sum_tensor-tensor_moment[alpha])/sum_tensor, rcut_def),max_displacement[alpha]) \
                                     + gaussian_noise(self.alpha_rand*max_displacement[alpha])
                else : 
                    displacement_alpha = constraint(system[idx].position[alpha]*f(centered_alpha,(sum_tensor-tensor_moment[alpha])/sum_tensor, rcut_def),max_displacement[alpha])
                system[idx].position[alpha] += displacement_alpha

        return system

    def hyperplane_dumped_Verlet_relaxation(self, system : Atoms, calc_vasp : Vasp, eigen_vector : np.ndarray, nb_iteration : int = 5) -> Atoms : 
        """Perform Verlet relaxation in orthogonal hyperplane of a given eigen vector
        
        Parameters
        ----------
        system : Atoms 
            ASE object containing all the system 
        calc_vasp : Vasp
            VASP ASe calculator to evaluate forces      
        nb_iteration : int 
            number of Verlet iteration 

        Returns
        -------
        Atoms 
            Updated ASE object after dumped Verlet integration
        """
        
        velocities = np.zeros(system.positions.shape)
        last_forces = np.zeros(system.positions.shape)
        inv_mass_matrix = np.diag(np.asarray([ 1.0/(at.mass) for at in system ]))

        for _ in range(nb_iteration) :
            if self.debug :  
                forces = self.get_forces_toy(system)
            else :
                system.calc = calc_vasp
                forces = self.get_forces_vasp(system)
            projected_forces = forces - np.trace(forces.T@eigen_vector)*eigen_vector

            """Application of dumped Verlet schem"""
            system.positions += velocities*self.delta_t_relax + 0.5*inv_mass_matrix@projected_forces*(self.delta_t_relax**2)
            velocities += (self.delta_t_relax*inv_mass_matrix@(projected_forces + last_forces) - self.dumping*velocities)/3.0
            last_forces += projected_forces

        return system

    def run_deformation_step(self, system : Atoms, noise : bool = False, tolerance : float = 0.1) -> Tuple[bool, Atoms] :
        """Perform on ART hopping step before method reached lambda_c criterion
        In this part of the algorithm the method applies local deformation one the system until lambda_c criterion
        is reached 
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system
        noise : bool 
            Boolean to apply noise on local deformation displacements
        tolerance : float 
            tolerance on relative dynamical norm
        
        Returns 
        -------
        bool 
            convergence boolean 

        Atoms 
            updated ASE object after ARTn step    
        """     
        
        def uniform_noise(width : List[float]) -> np.ndarray : 
            """Apply 0D uniform noise 
            
            Parameters 
            ----------
            width
                width of the uniform noise

            Returns 
            -------
            float 
                Given realisation of the random uniform variable
            """
            return np.array([ np.random.uniform(-width_k,width_k) for width_k in width])
        
        center = self.center_calculator(system)        
        system = self.elliptic_deformation(system, center, self.rcut, self.max_disp, noise = noise)

        for id, at in enumerate(system) : 
            if id not in self.list_idx : 
                at.position += uniform_noise([self.alpha_rand*el for el in self.max_disp])

        """Compute the eigen vector of elliptical deformation"""
        eigen_vector_def = np.zeros(system.positions.shape)
        for idx in self.list_idx : 
            eigen_vector_def[idx,:] = np.ones(3)
        eigen_vector_def *= 1.0/np.linalg.norm(eigen_vector_def)

        system = self.hyperplane_dumped_Verlet_relaxation(system, self.calc_vasp, eigen_vector_def)
        
        Delta_hessian_norm, eigen_val_hessian, _ = self.Hessian_diag(system, center)
        if Delta_hessian_norm > tolerance : 
            print('Problem with Hessian symmetrisation')
            exit(0)
        else : 
            if np.amin(eigen_val_hessian) < self.lambda_c : 
                return True, system
            else :
                return False, system

    def run_ARTn_step(self ,system : Atoms, tolerance : float = 0.01) -> Tuple[bool, Atoms, float | None, np.ndarray | None] :
        """Perform on ART hopping step after method reached lambda_c criterion
        In this part of the algorithm, the method is seeking the lowest eigen value direction to find 
        the saddle point
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system
        tolerance : float 
            tolerance on relative dynamical matrix norm
        
        Returns 
        -------
        bool 
            convergence boolean 

        Atoms 
            updated ASE object after ARTn step  

        float | None
            Minimum eigenvalue if convergence is reached None otherwise
        
        np.ndarray | None 
            Associated eigenvector if convergence is reached None otherwise
          
        """
        system.calc = self.calc_vasp
        
        """update skim for ARTn method"""
        mass_center_idx = self.center_calculator(system)
        Delta_hessian_norm, eigen_val, eigen_vector = self.Hessian_diag(system, mass_center_idx)
        if Delta_hessian_norm > tolerance : 
            print('Problem with Hessian symmetrisation')
            exit(0)

        else : 
            if self.debug :
                forces = self.get_forces_toy(system)
            else :
                forces = self.get_forces_vasp(system)
            if np.amax(np.linalg.norm(forces, axis = 1)) < self.tol_force : 
                min_eig_val, min_eig_vect = self.extract_minimum_lambda_eigen_vector(eigen_val,eigen_vector)
                return True, system, min_eig_val, min_eig_vect
            
            else : 
                """compute the eigen vector corresponding to the lowest eigen value"""
                _, min_eig_vect = self.extract_minimum_lambda_eigen_vector(eigen_val,eigen_vector)
                
                """Perform dumped Verlet relaxation in orthogonal subspace of the previous eigen vector"""
                system = self.hyperplane_dumped_Verlet_relaxation(system, self.calc_vasp, min_eig_vect)
                
                """Updating the system !"""
                scalar = np.trace(forces@min_eig_vect)
                if scalar > 0.0 : 
                    system.positions += - self.mu*scalar*min_eig_vect/np.sqrt(self.iteration)
                else : 
                    system.positions += self.mu*scalar*min_eig_vect/np.sqrt(self.iteration)
                self.iteration += 1.0
                return False, system, None, None

    def run_push_saddle_step(self, min_system : Atoms, saddle_system : Atoms, eig_val : float, eig_vect : np.ndarray) -> Atoms : 
        """Perform pushing out step at saddle point to reach an other minimum, system is pushed 
        in the direction corresponding to the lowest eigenvalue. Displacement is inversly proportionnal to
        the 
        
        Parameters 
        ----------
        min_system : Atoms 
            ASE object containing the previous minimum 

        sadlle_system : Atoms 
            ASE object containing the current saddle point system 

        eig_val : float 
            Minimum eigenvalue from the Hessian
        
        eig_vect : np.ndarray 
            Associated eigen vector

        Returns
        -------

        Atoms 
            Updated Atoms object after pushing out procedure

        """
        
        def displacements_constraints(displacement_vector : np.ndarray) -> np.ndarray :
            """Update displacement wrt non isotropic maximum displacement
            
            Parameters
            ----------
            displacement_vector : np.ndarray
                displacement vector to modify

            
            Returns
            -------

            np.ndarray
                Updated displacement vector wrt the non isotropic displacement constraints
            
            """ 
            displacement_vector_copy = displacement_vector.copy()
            for i in range(displacement_vector.shape[0]) : 
                for j in range(displacement_vector.shape[0]) : 
                    if abs(displacement_vector[i,j]) > self.max_disp[j] : 
                        displacement_vector_copy[i,j] = self.max_disp[j]*np.sign(displacement_vector[i,j])
            
            return displacement_vector_copy

        scalar = np.trace(eig_vect.T@(saddle_system.positions - min_system.positions))/(np.linalg.norm(saddle_system.positions - min_system.positions)*np.linalg.norm(eig_vect))
        displacement_vector = self.mu*self.max_disp*scalar*eig_vect/eig_val #convertion factor should be added here 
        if scalar > 0 : 
            saddle_system.positions += displacements_constraints(displacement_vector)
            #np.where(np.abs(displacement_vector) < self.max_disp, displacement_vector, np.sign(displacement_vector)*self.max_disp )
        else : 
            saddle_system.positions += -displacements_constraints(displacement_vector)
            #np.where(np.abs(displacement_vector) < self.max_disp, -displacement_vector, -np.sign(displacement_vector)*self.max_disp )
        return saddle_system

    def run_relaxation_step(self ,system : Atoms) -> Tuple[bool, Atoms] :
        """Perform relaxation procedure until new minimum is reached (criteion on self.tol_force is reached).
        Then return the minium system and its corresponding energy
        
        Parameters 
        ----------
        system : Atoms 
            ASE object containing all the system
        
        Returns 
        -------
        bool 
            convergence boolean 

        Atoms 
            updated ASE object after relaxation step  

        float | None
            Energy at the minimum if convergence is reached None otherwise
                  
        """
        system.calc = self.calc_vasp
        
        """Check the force convergence criterion"""
        if self.debug : 
            forces = self.get_forces_toy(system)
        else :
            forces = self.get_forces_vasp(system)
        if np.amax(np.linalg.norm(forces, axis = 1)) < self.tol_force :
            return True, system     
        
        else :      
            """Perform dumped Verlet relaxation on whole system (eigen vector is zero vector...)"""
            system = self.hyperplane_dumped_Verlet_relaxation(system, self.calc_vasp, np.zeros(forces.shape))
            return False, system