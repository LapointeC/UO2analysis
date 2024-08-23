import numpy as np
from math import ceil
from .lammps_worker import LammpsWorker

from ase import Atoms 
import shutil, os
import warnings


class HarmonicVibration : 
    """Generate / diagonalise dymical matrix of a given system to extract harmonic vibration modes
    two point estimation is used to build the dynamical matrix"""
    def __init__(self, 
                 system : Atoms,
                 potential_file : os.PathLike[str],
                 displacement_amplitude : float = 1e-3,
                 relative_symmetric_norm : float = 1e-2,
                 working_directory : os.PathLike[str] = './harmonic_vib') -> None : 
        """Init method for ```HarmonicVibration```
        
        Parameters:
        -----------

        system : Atoms 
            ```Atoms``` system to perform vibrationnal calculation

        potential_file : os.PathLike[str]
            Path to the lammps potential to use

        displacement_amplitude : float
            Amplitude of perturbation displacement to build the dynamical matrix (in AA)

        relative_symmetric_norm : float 
            Tolerance on relative symmetric norm \epsilon = \Vert D^T - D \Vert / \Vert D \Vert, where D is the dynamical matrix and \Vert \cdot \Vert is the Frobenuis norm
        
        working_directory : os.PathLike[str]
                Path to lammps working directory
        """

        self.kB = 8.6173303e-5
        self.hbar = 6.582119e-16 

        self.system = system 
        self.delta_xi = displacement_amplitude
        self.relative_symmetric_norm = relative_symmetric_norm

        self.potential_file = potential_file
        self.working_directory = working_directory

        self.lammps_worker = LammpsWorker(self.working_directory, self.system)
        self.omega = None 
        self.U = None

    def InitSimulation(self, name_file : str = 'in.lmp') -> None : 
        """Initialise lammps instance for vibration calculations
        
        Parameters:
        -----------

        name_file : str
            Name of lammps geometry file
        """
        if not os.path.exists(self.working_directory) :
            os.mkdir(self.working_directory)
        else : 
            shutil.rmtree(self.working_directory)
            os.mkdir(self.working_directory) 

        self.lammps_worker.InitLammpsInstance()
        self.lammps_worker.DumpAtomsSystem(name_file=name_file)
        os.system('ln -s {:s} {:s}/pot.fs'.format(self.potential_file,self.working_directory))
        os.chdir(self.working_directory)
        self.lammps_worker.ReadInputLines()
        return

    def GetOmega2(self) -> np.ndarray : 
        """Extract omega^2 array
        
        Returns:
        --------

        np.ndarray 

            omega^2 array (in rad^2.THz^2)
        """
        return self.omega2
    
    def GetUmatrix(self) -> np.ndarray : 
        """Extract U array between normal mode and cartesian coordinate
        
        Returns:
        --------

        np.ndarray 

            U array (xi_{ia,nu})
        """     
        return self.U

    def GetEinsteinOmega(self, temperature : float = 300.0) -> float : 
        """Compute Einstein pulsation of the system for a given temperature 
        
        Parameters:
        -----------

        temperature : float 
            Temperature in K

        Returns:
        --------

        float 
            Einstein pulsation at temperature
        """
        real_omega = [np.sqrt(omega2) for omega2 in self.omega2 if omega2 > 1e-2]
        convert_radTHz_to_radHz = 1.0e12
        numerator = np.sum([ np.log((omega*convert_radTHz_to_radHz*self.hbar)/(self.kB*temperature)) for omega in real_omega ])/len(real_omega)
        return self.kB*temperature*(np.exp(numerator)/self.hbar)/convert_radTHz_to_radHz

    def CheckDynamicalSymmetricNorm(self, relative_norm : float) -> None : 
        """Check the relative symmetric norm of the dynamical matrix 
        
        Parameters:
        -----------

        relative_norm : float 
            Relative symmetric norm of the dynamical matrix
        """
        if relative_norm > self.relative_symmetric_norm : 
            warnings.warn('Dynamical matrix is not symmetric, relative norm : {:1.4f}'.format(relative_norm))
        return 

    def CheckFrequencies(self, eigenvalues : np.ndarray) -> None : 
        """Check vibration modes to ensure that the configuration is corresponding to a minimum (only 3 frequencies should be equal to 0 in periodic systems)

        Parameters:
        -----------

        eigenvalues : np.ndarray 
            Vibration frequencies vector 

        """
        imaginary_modes = [el for el in eigenvalues if el < 1e-2]
        if len(imaginary_modes) > 3 : 
            warnings.warn('Instabilities in dynamical matrix : {:2d} modes are imaginary'.format(len(imaginary_modes)))
        
        idx_to_keep = [id for id in range(len(eigenvalues)) if eigenvalues[id] > 1e-2]
        self.U = self.U[:,idx_to_keep]
        self.omega2 = self.omega2[idx_to_keep]
        return 

    def DynamicalMatrixEvaluation(self ,left_force_xi : np.ndarray, right_force_xi : np.ndarray, masse : float) -> np.ndarray:
        """Evaluate the dynamical vector for atom i
        
        Parameters:
        -----------

        left_force_xi : np.ndarray
            Force vector for left perturbation

        right_force_xi : np.ndarray
            Force vector for right perturbation

        mass : float 
            Mass of the atoms
        
        Returns:
        --------

        np.ndarray
            Dynamical vector for atom i
        """
        return (np.array(right_force_xi) - np.array(left_force_xi))/(2*self.delta_xi*masse) 

    def VibrationDiagCalculation(self) : 
        """Perform the whole building / diagonalisation of the dynamical matrix for the system"""
        print('... Starting of LAMMPS perturbations...')
        
        dxi = [np.array([1.0,0.0,0.0]),np.array([0.0,1.0,0.0]),np.array([0.0,0.0,1.0])]
        Dynamical_matrix = np.zeros((3*len(self.system),3*len(self.system)), dtype=float)
        
        for i,at_i in enumerate(self.system) :
            for j,at_j in enumerate(self.system) :
                mass_ij = np.sqrt(at_i.mass*at_j.mass)
                for alpha, x_alpha in enumerate(dxi):
                    two_points_forces_list_xi_k = []
                    for signe in [-1,1]:
                        force_delta_ialpha_to_jbeta = self.lammps_worker.Force_i_on_j(i,j, signe*self.delta_xi*x_alpha)
                        two_points_forces_list_xi_k.append(force_delta_ialpha_to_jbeta)
                    for beta, x_beta in enumerate(dxi) :
                        Dynamical_matrix_ialpha_to_jbeta = np.dot(self.DynamicalMatrixEvaluation(two_points_forces_list_xi_k[1],two_points_forces_list_xi_k[0],mass_ij),x_beta)
                        Dynamical_matrix[3*i+alpha,3*j+beta] = Dynamical_matrix_ialpha_to_jbeta

        print('... Full Dynamical matrix is built ...')
        Delta_dynamical_norm = np.linalg.norm(Dynamical_matrix-Dynamical_matrix.T)/np.linalg.norm(Dynamical_matrix)
        self.CheckDynamicalSymmetricNorm(Delta_dynamical_norm)

        Dynamical_matrix = 0.5*(Dynamical_matrix + Dynamical_matrix.T)
        print('... Dynamical matrix will be diagonalised ...')
        eigen_values, eigen_vectors = np.linalg.eigh(Dynamical_matrix, UPLO='L')
        print('... Dynamical matrix is fully diagonalised ...')

        # conversion (1e-1 rad.PHz)^2 -> (rad.THz)^2
        eigen_values *= 1.0e4

        self.dynamical_matrix = Dynamical_matrix
        self.omega2 = eigen_values
        self.U = eigen_vectors
        self.CheckFrequencies(eigen_values)
        return