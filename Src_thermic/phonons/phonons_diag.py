import numpy as np
from math import ceil
from .lammps_worker import LammpsWorker

from ase import Atoms 
import shutil, os
import warnings


class HarmonicVibration : 
    def __init__(self, 
                 system : Atoms,
                 potential_file : str,
                 displacement_amplitude : float = 1e-3,
                 relative_symmetric_norm : float = 1e-2,
                 working_directory : str = '/harmonic_vib') -> None : 

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
        return self.omega2
    
    def GetUmatrix(self) -> np.ndarray : 
        return self.U

    def GetEinsteinOmega(self, temperature : float = 300.0) -> float : 
        real_omega = [np.sqrt(omega2) for omega2 in self.omega2 if omega2 > 1e-2]
        convert_radTHz_to_radHz = 1.0e12
        numerator = np.sum([ np.log((omega*convert_radTHz_to_radHz*self.hbar)/(self.kB*temperature)) for omega in real_omega ])/len(real_omega)
        return self.kB*temperature*(np.exp(numerator)/self.hbar)/convert_radTHz_to_radHz

    def CheckDynamicalSymmetricNorm(self, relative_norm : float) -> None : 
        if relative_norm > self.relative_symmetric_norm : 
            warnings.warn('Dynamical matrix is not symmetric, relative norm : {:1.4f}'.format(relative_norm))

    def CheckFrequencies(self, eigenvalues : np.ndarray) -> None : 
        imaginary_modes = [el for el in eigenvalues if el < 1e-2]
        if len(imaginary_modes) > 3 : 
            warnings.warn('Instabilities in dynamical matrix : {:2d} modes are imaginary'.format(len(imaginary_modes)))
        
        idx_to_keep = [id for id in range(len(eigenvalues)) if eigenvalues[id] > 1e-2]
        self.U = self.U[:,idx_to_keep]
        self.omega2 = self.omega2[idx_to_keep]

    def DynamicalMatrixEvaluation(self ,left_force_xi : np.ndarray, right_force_xi : np.ndarray, masse : float) -> np.ndarray:
        return (np.array(right_force_xi) - np.array(left_force_xi))/(2*self.delta_xi*masse) 

    def VibrationDiagCalculation(self) : 
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