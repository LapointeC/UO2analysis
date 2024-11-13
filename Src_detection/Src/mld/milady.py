"""This module defines an ASE interface to Milady.

This module is based on work of Jussi Enkovaara and John
Kitchin for ASE interface to VASP.

Alternatively, user can set the environmental flag $MILDAY_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import shutil
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any, Union, TypedDict
from xml.etree import ElementTree

import ase
from ase import Atoms

from ase.build import sort
from typing import List
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from .create_inputs import GenerateMiladyInput

from .milady_writer import fill_milady_descriptor, write_milady_poscar

DEFAULTS = {"r_cut": 5.0}

class MiladyError(RuntimeError) :
    """Base class of error types related to Milady objects."""

class MiladySetupError(MiladyError) :
    """Calculation cannot be performed with the given parameters.

    Reasons to raise this errors are:
      * The calculator is not properly configured
        (missing executable, environment variables, ...)
      * The given atoms object is not supported
      * Calculator parameters are unsupported

    Typically raised before a calculation."""

class DBtype(TypedDict) : 
    """Derived type for internal dictionnaries of DBManager"""
    atoms : Atoms
    energy : float
    forces : np.ndarray
    stress : np.ndarray

class Optimiser : 
    """General Optimiser class for Milady code. More information about optimiser options are avaible
    at :  https://ai-atoms.github.io/milady-docs/contents/ml
    
    Optimiser is mainly a dictionnary which contains all optimisation options for Milady

    """
    def __init__(self, param : dict) : 
        self.param = param

    @classmethod
    def Milady(cls,
               debug : bool = False,
               mld_fit_type : int = 0,
               snap_regularization_type : int = 4,
               svd_rcond : float = -1.0,
               snap_class_constraints : List[str] = ['00'],
               lambda_krr : float = -1.0,
               min_lambda_krr : float =  1.0e-9,
               max_lambda_krr : float =  1.0e10,  
               type_of_loss : int = 1,             
               db_file : str = 'db_model.in',
               db_path : str = 'DB/',
               #build_subdata : bool = False,
               #seed : int = 17869451,
               weighted : bool = False,
               weighted_auto : bool = False,
               weighted_3ch : bool = False,
               fix_no_of_elements : int = 1,
               chemical_elements : List[str] = ['Fe'],
               chemical_elements_invisible : List[str] = [''],
               weight_per_element : List[float] = [1.0],
               weight_per_element_3ch : List[float] = [1.0],
               ref_energy_per_element : List[float] = [0.0],
               sign_stress : float  = 1.0,
               sign_stress_big_box : float = 1.0,
               write_desc : bool = True,
               write_desc_dump : bool = False,
               read_desc_dump : bool = False,
               desc_forces : bool = True,
               val_desc_max : float = 1.0e4,
               optimize_weights : bool = False,
               optimize_weights_L1 : bool = False,
               optimize_weights_L2 : bool = False,
               optimize_weights_Le : bool = False,
               #optimize_ga_population : bool = False,
               class_no_optimize_weights : List[str] = [''],
               max_iter_optimize_weights : int = 40,
               factor_energy_error : float = 1.0,
               factor_force_error : float = 1.0,
               factor_stress_error : float = 1.0,
               lbfgs_m_hess : int = 40,
               lbfgs_max_steps : int = 2e4,
               #lbfgs_print : List[int] = [100,0],
               lbfgs_eps : float = 0.01,
               lbfgs_gtol : float = 0.4,
               lbfgs_xtol : float = 1.0e-15,
               iread_ml : int = 0,
               isave_ml : int = 0,
               #toy_model : bool = False,
               #kcross : bool = False,
               #search_hyp : bool = False, 
               #sparsification : bool = False,
               #sparsification_by_acd : bool = False,
               #sparsification_by_entropy : bool = False,
               #sparsification_by_cur : bool = False,
               marginal_likelihood : bool = False,
               #target_type : int = 1,
               #force_comp : int = 1,
               n_kcross : int = 0,
               dim_data : int = 100,
               nd_fingerprint : int = 3,
               #n_frac : float = 0.0,
               #max_data : int = 0,
               #rescale : bool = False,
               #s_max_r : int = 10,
               #s_max_i : int = 0,
               #s_min_r : int = 0,
               #s_min_i : int = 1,
               #kelem : int = 0,
               #ns_data : int = 100,
               #i_begin : int = 0,
               #pref : str = '00'
               ) : 

        param = {}
        saved_args = locals()
        for key in saved_args.keys() :  
            if key != 'cls' and key != 'param' : 
                param[key] = saved_args[key]

        return cls(param)

class Regressor : 
    """General Regressor class for Milady code. More information about optimiser options are avaible
    at :  https://ai-atoms.github.io/milady-docs/contents/ml
    
    Regressor is mainly a dictionnary which contains all regression options for Milady. 
    Main method of Regressor are : 
        - ComputeDescritpors: Only compute atomic descriptors for a subset of configurations
        - Linear: Build linear ML potential based on atomic descriptors  
        - Quadratic: Build quadratric ML potential based on atomic descriptors
        - PolynomialsChaos:: Build ML potential thanks to Hermite polynomial chaos based on atomic descriptors 
        - Nlinear: Build N-linear ML potential based on atomic descriptors
        - Kernel: Build Kernel ML potential based on atomic descriptors
        
    """
    def __init__(self, param : dict) : 
        self.param = param 

    @classmethod 
    def ComputeDescriptors(cls, write_design_matrix : bool = False) :
        """Compute only descriptors for a given subset of configurations
        
        Parameters 
        ----------

            write_design_matrix: bool 
                Boolean to write the design matrix of the system
        """ 
        param = {}
        param['ml_type'] = -1
        param['write_design_matrix'] = write_design_matrix

        return cls(param)

    @classmethod
    def Linear(cls, train_only : bool = False, 
               write_design_matrix : bool = False) : 
        """Fit a linear model in descriptors for a given subset of configurations
        
        Parameters
        ----------

            train_only: bool
                Boolean to perform only training and no test
            
            write_design_matrix: bool 
                Boolean to write the design matrix of the system           
        """
        
        param = {}
        param['ml_type'] = 0
        param['mld_order'] = 1
        param['train_only'] = train_only
        param['write_design_matrix'] = write_design_matrix

        return cls(param)

    @classmethod
    def Quadratic(cls, mld_type_quadratic : int = 1, 
               train_only : bool = False, 
               write_design_matrix : bool = False) : 
        """Fit a quadratic model in descriptors for a given subset of configurations
        
        Parameters
        ----------

            mld_type_quadratic: int 
                Specify the type quadratic potential:
                    - 1 is for QNML (see Goryeava et al. 2022)
                    - 2 is for QML

            train_only: bool
                Boolean to perform only training and no test
            
            write_design_matrix: bool 
                Boolean to write the design matrix of the system           
        """

        param = {}
        param['ml_type'] = 0
        param['mld_order'] = 2
        param['mld_type_quadratic'] = mld_type_quadratic
        param['train_only'] = train_only
        param['write_design_matrix'] = write_design_matrix

        return cls(param)
    
    @classmethod
    def PolynomialsChaos(cls, polyc_n_poly : int = 2,
                         polyc_n_hermite : int = 2,
                         train_only : bool = False, 
                         write_design_matrix : bool = False) :
        """Fit a polynomial chaos model in descriptors for a given subset of configurations
        
        Parameters
        ----------

            polyc_n_poly: int 
                Degree of polynomial chaos extension
            
            polyc_n_hermite: int
                Maximal degree of Hermite polynomials (Milday handle up to 4)
            
            train_only: bool
                Boolean to perform only training and no test
            
            write_design_matrix: bool 
                Boolean to write the design matrix of the system           
        """       

        param = {}
        param['ml_type'] = 0
        param['mld_order'] = 3
        param['poltc_n_poly'] = polyc_n_poly
        param['polyc_n_hermite'] = polyc_n_hermite
        param['train_only'] = train_only
        param['write_design_matrix'] = write_design_matrix

        return cls(param)

    @classmethod
    def Nlinear(cls, order_nlinear : int = 2,
               train_only : bool = False, 
               write_design_matrix : bool = False) : 
        """Fit a N-linear model in descriptors for a given subset of configurations
        
        Parameters
        ----------

            order_nlinear: int
                Maximum order of N-linear model 
            
            train_only: bool
                Boolean to perform only training and no test
            
            write_design_matrix: bool 
                Boolean to write the design matrix of the system           
        """      

        param = {}
        param['snap_order'] = 22
        param['order_nlinear'] = order_nlinear
        param['train_only'] = train_only
        param['write_design_matrix'] = write_design_matrix

        return cls(param)
    
    @classmethod
    def Kernel(cls, kernel_type : str = 'polynomial', 
               lenght_kernel : float = 0.05,
               sigma_kernel : float = 0.0,
               kernel_power : float = 2.0,
               write_kernel_matrix : bool = False,
               kernel_dump : int = 1,
               classes_for_mcd : List[str] = ['01'],
               np_kernel_ref : int = 200,
               np_kernel_full : int = 800,
               power_mcd : float = 0.5,
               cur_kval : int = -1,
               cur_rval : int = -1,
               cur_eps : float = 1.0) : 
        """Fit a kernel model in descriptors for a given subset of configurations more informations 
        https://ai-atoms.github.io/milady-docs/contents/ml/kernels.html
        
        Parameters
        ----------

            kernel_type: str
                Type of kernel used for the regression. Following ones are implemented 
                square-exponential, polynomial, Mahalanobis-Batchattarya, random-square-exponential and random-polynomial
            
            lenght_kernel: float
                Caracteristic lenght of kernel in descriptor space
                
            sigma_kernel: float
                Caracteristic offset/amplitude of kernel 
            
            kernel_power: float
                Degree of kernel for polynomials kernels

            write_kernel_matrix: bool 
                Boolean to write the kernel design matrix

            kernel_dump: int
                Specify the sparse point algorithm for data selection.
                    - 1 normalised selection based on MCD/Mahalanobis distances
                    - 2 normalised selection based on MCD/Mahalanobis distances
                    - 3 selection based on CUR decomposition

            classes_for_mcd: List[str]
                List of reference classes for MCD selection

            np_kernel_ref: int
                Number of sparse points selected in reference classes

            np_kernel_full: int
                Total number of sparse points selected in whole database

            power_mcd: float 
                Power used for MCD / Mahalanobis metric (0.5 corresponding to Mahalanobis)

            cur_kval: int 
                Order of the SVD selection needed for CUR decomposition

            cur_rval: int
                Number of rows selected with CUR decomposition procedure

            cur_eps: float
                Maximum error (in L2 matrix norm) dues to CUR sampling procedure

        """             

        dic_equiv = {'square-exponential':1,
                     'polynomial':4,
                     'Mahalanobis-Batchattarya':6,
                     'random-square-exponential':7,
                     'random-polynomial':44}
        
        if kernel_type not in dic_equiv.keys() : 
            raise MiladySetupError("Kernel name is not implemented")
        
        param = {}
        saved_args = locals()
        for key in saved_args.keys() : 
            if key != 'cls' and key != 'param': 
                if key == 'kernel_type' : 
                    param[key] = dic_equiv[saved_args[key]]
                else : 
                    param[key] = saved_args[key]

        return cls(param)


class Descriptor : 
    """General Descriptor class for Milady code. More information about optimiser options are avaible
    at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
    
    Descriptor is mainly a dictionnary which contains all descriptor options for Milady. 
    Main method of Descriptor are : 
        - G2 : Behler radial descriptor
        - G3 : Behler radial descritpor evolution  
        - Behler : Behler descriptor
        - AFS : Angular Fourier Series descriptor
        - SOAP : SOAP descritpor from Bartok-Csanyi 
        - PSO3 : Power spectrum SO3
        - PSO3_3bodies : Power spectrum SO3 including 3 bodies interactions
        - BSO3 : bi-spectrum SO3
        - BSO4 : bi-spectrum SO4
        - Hybrid_G2_AFS : Hybrid descriptor between G2 and AFS
        - Hybrid_G2_BSO4 : Hybrid descriptor between G2 and BSO4
        - MPT : Moment Polynomials Tensor from Shapeev
        - PiP : Permutationally invariant Polynomials from Oord
        - ACE : Atomic Cluster Expansion from Drautz
        
    """
    def __init__(self,param : dict, dimension : int) : 
        self.param = param
        self.dimension = dimension

    @classmethod        
    def G2(cls, r_cut : float = DEFAULTS["r_cut"], 
           n_g2_eta : int = 1, 
           n_g2_rs : int = 1,
           eta_max_g2 : float = 0.8) :
        """Compute G2 radial descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_g2_eta: int 
                Main parameter to build eta grid for G2 descriptor

            n_g2_rs: int
                Main parameter to build radial grid for G2 descriptor 

            eta_max_g2: float 
                Maximum value for eta parameter

        """

        param = {}
        param['descriptor_type'] = 1
        param['r_cut'] = r_cut
        param['n_g2_eta'] = n_g2_eta
        param['n_g2_rs'] = n_g2_rs
        param['eta_max_g2'] = eta_max_g2

        return cls(param, n_g2_eta*n_g2_rs)

    @classmethod
    def G3(cls, r_cut : float = DEFAULTS["r_cut"], 
           n_g3_eta : int = 1, 
           n_g3_zeta : int = 1,
           n_g3_lambda : int = 1) :
        """Compute G3 radial descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_g3_eta: int 
                Main parameter to build eta grid for G3 descriptor

            n_g3_zeta: int
                Main parameter to build zeta grid for G3 descriptor 

            n_g3_lambda: int 
                Main parameter to build lambda grid for G3 descriptor
                
        """

        param = {}
        param["descriptor_type"] = 2
        param["r_cut"] = r_cut
        param["n_g3_eta"] = n_g3_eta
        param["n_g3_zeta"] = n_g3_zeta
        param["n_g3_lambda"] = n_g3_lambda
        
        #put dimension
        return cls(param, None)     

    # Put the Behler here !
    @classmethod
    def Behler(cls, r_cut : float = DEFAULTS["r_cut"],
               strict_behler : bool = False) :
        """Compute Behler descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            strict_behler: bool 
                Boolean to used exact setting for Behler paper
                
        """
        param = {}
        param['descriptor_type'] = 3
        param['r_cut'] = r_cut
        param['strict_behler'] = strict_behler

        return cls(param, None)

    @classmethod
    def AFS(cls, r_cut : float = DEFAULTS["r_cut"],
            afs_type : int = 1,
            n_rbf : int = 1,
            n_cheb : int = 1) :
        """Compute AFS descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            afs_type: int
                Type of AFS dsecriptor : 
                    - 1, radial functions are the same for a given triplet 
                    - 2, radial functions are different for a given triplet

            n_rbf: int
                Number of radial channels 

            n_cheb: int 
                Number of Tchebychev polynomials
                
        """

        param = {}
        param["descriptor_type"] = 4
        param["r_cut"] = r_cut
        param["afs_type"] = afs_type
        param["n_rbf"] = n_rbf
        param["n_cheb"] = n_cheb
        
        return cls(param, n_rbf*(n_cheb+1) if (afs_type == 1) else n_rbf**2*(n_cheb+1))         

    @classmethod 
    def SOAP(cls, r_cut : float = DEFAULTS["r_cut"],
             l_max : int = 1,
             n_soap : int = 1,
             nspecies_soap : int = 1,
             r_cut_width_soap : float = 5.0,
             lsoap_diag : bool = False,
             lsoap_lnorm : bool = False,
             lsoap_norm: bool = False) : 
        """Compute SOAP descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            l_max: int 
                l_max for spherical harmonic decomposition

            n_soap: int
                Number of Gaussian functions

            nspecies_soap: int 
                Number of different species taken into account for SOAP calcuation

            r_cut_width_soap: float 
               Intermediate regime for the cutoff function 

            lsoap_diag: bool 
                Boolean to use only diagonal SOAP in radial functions

            lsoap_lnorm: bool 
                Boolean to normalised angular channels

            lsoap_norm: bool 
                Boolean to normalised SOAP descriptor
                
        """


        param = {}
        param['descriptor_type'] = 5
        param['r_cut'] = r_cut
        param['l_max'] = l_max
        param['n_soap'] = n_soap
        param['r_cut_width_soap'] = r_cut_width_soap
        param['nspecies_soap'] = nspecies_soap
        param['lsoap_diag'] = lsoap_diag
        param['lsoap_lnorm'] = lsoap_lnorm
        param['lsoap_norm'] = lsoap_norm

        return cls(param, (l_max+1)*n_soap*nspecies_soap*(n_soap*nspecies_soap+1)/2 if 
                   lsoap_diag else (l_max+1)*n_soap*nspecies_soap*(nspecies_soap+1)/2)

    @classmethod
    def PSO3(cls, r_cut : float = DEFAULTS["r_cut"],
             n_rbf : int = 1, 
             l_max : int = 1,
             radial_pow_so3 : int = 1) : 
        """Compute power spectrum SO3 descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_rbf: int
                Number of radial channels

            l_max: int
                l_max for spherical harmonic decomposition               
                
            radial_pow_so3: int
                Type of radial fucntion used to compute PSO3 descritpor
                    - 1, original basis which is the same as AFS
                    - 2, radial basis based on Tchebyshev polynomials
        """

        param = {}
        param['descriptor_type'] = 6 
        param['r_cut'] = r_cut
        param['n_rbf'] = n_rbf
        param['l_max'] = l_max
        param['radial_pow_so3'] = radial_pow_so3

        return cls(param, n_rbf*(l_max+1) if radial_pow_so3 == 1 else (n_rbf+1)*(l_max+1))

    @classmethod 
    def PSO3_3bodies(cls, r_cut : float = DEFAULTS["r_cut"],
                     n_rbf : int = 1,
                     l_max : int = 1,
                     radial_pow_so3 : int = 1) :
        """Compute power spectrum SO3 - 3 bodies descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_rbf: int
                Number of radial channels

            l_max: int
                l_max for spherical harmonic decomposition               
                
            radial_pow_so3: int
                Type of radial fucntion used to compute PSO3 descritpor
                    - 1, original basis which is the same as AFS
                    - 2, radial basis based on Tchebyshev polynomials
        """

        param = {}
        param['descriptor_type'] = 603
        param['r_cut'] = r_cut
        param['n_rbf'] = n_rbf
        param['l_max'] = l_max
        param['radial_pow_s03'] = radial_pow_so3

        return cls(param, (n_rbf**2)*l_max if radial_pow_so3 == 1 else l_max*(n_rbf+1)**2) 

    @classmethod
    def BSO3(cls, r_cut : float = DEFAULTS["r_cut"], 
             n_rbf : int = 1, 
             l_max : int = 1,
             lbso3_diag : bool = False) :
        """Compute bi-spectrum SO3 descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_rbf: int
                Number of radial channels

            l_max: int
                l_max for spherical harmonic decomposition               

            lbso3_diag: bool 
                Boolean to use only diagonal component for spherical decomposition (i.e l_1 = l_2)

        """

        param = {} 
        param['descriptor_type'] = 7
        param['r_cut'] = r_cut
        param['n_rbf'] = n_rbf
        param['l_max'] = l_max
        param['lbso3_diag'] = lbso3_diag

        # No possible estimations ...
        return cls(param, None)

    @classmethod
    def BSO4(cls, r_cut : float = DEFAULTS["r_cut"],
             j_max : float = 1.5,
             lbso4_diag : bool = False, 
             inv_r0_input : float = 0.993633802276324) :
        """Compute bi-spectrum SO4 descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            j_max: float
                j_max for hyper-spherical harmonic decomposition               

            lbso4_diag: bool 
                Boolean to use only diagonal component for spherical decomposition (i.e j_1 = j_2)

            inv_r0_input: float
                Value of the maximum projection at north pole in pi unit. Value has to be slighlty lower than 1
                You should trust the default choice ...
                
        """

        param = {}
        param['descriptor_type'] = 9
        param['r_cut'] = r_cut
        param['lbso4_diag'] = lbso4_diag
        param['j_max'] = j_max
        param['inv_r0_input'] = inv_r0_input

        #need to be tabulated ...
        return cls(param,None)

    @classmethod
    def Hybrid_G2_AFS(cls, r_cut : float = DEFAULTS["r_cut"],
                      n_g2_eta : int = 1, 
                      n_g2_rs : int = 1,
                      eta_max_g2 : float = 0.8,
                      afs_type : int = 1,
                      n_rfb : int = 1,
                      n_cheb : int = 1) : 

        """Compute G2 + AFS descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_g2_eta: int 
                Main parameter to build eta grid for G2 descriptor

            n_g2_rs: int
                Main parameter to build radial grid for G2 descriptor 

            eta_max_g2: float 
                Maximum value for eta parameter
    
            afs_type: int
                Type of AFS dsecriptor : 
                    - 1, radial functions are the same for a given triplet 
                    - 2, radial functions are different for a given triplet

            n_rbf: int
                Number of radial channels 

            n_cheb: int 
                Number of Tchebychev polynomials

        """

        param = {}
        param['descriptor_type'] = 14
        param['r_cut'] = r_cut
        param['n_g2_eta'] = n_g2_eta
        param['n_g2_rs'] = n_g2_rs
        param['eta_max_g2'] = eta_max_g2
        param['afs_type'] = afs_type
        param['n_rbf'] = n_rfb
        param['n_cheb'] = n_cheb

        return cls(param, n_g2_eta*n_g2_rs + n_rfb*(n_cheb+1) if afs_type == 1 else 
                   n_g2_eta*n_g2_rs + (n_cheb+1)*n_rfb**2)
    
    @classmethod 
    def Hybrid_G2_BSO4(cls, r_cut : float = DEFAULTS["r_cut"],
                      n_g2_eta : int = 1, 
                      n_g2_rs : int = 1,
                      eta_max_g2 : float = 0.8,
                      lbso4 : bool = False, 
                      j_max : float = 1.5,
                      inv_r0_input : float = 0.993633802276324) : 

        """Compute G2 + BSO4 descritpors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            n_g2_eta: int 
                Main parameter to build eta grid for G2 descriptor

            n_g2_rs: int
                Main parameter to build radial grid for G2 descriptor 

            eta_max_g2: float 
                Maximum value for eta parameter

            j_max: float
                j_max for hyper-spherical harmonic decomposition               

            lbso4_diag: bool 
                Boolean to use only diagonal component for spherical decomposition (i.e j_1 = j_2)

            inv_r0_input: float
                Value of the maximum projection at north pole in pi unit. Value has to be slighlty lower than 1
                You should trust the default choice ...
                
        """

        param = {}
        param['descriptor_type'] = 19
        param['r_cut'] = r_cut
        param['n_g2_eta'] = n_g2_eta
        param['n_g2_rs'] = n_g2_rs
        param['eta_max_g2'] = eta_max_g2
        param['lbso4'] = lbso4
        param['j_max'] = j_max
        param['inv_r0_input'] = inv_r0_input

        return cls(param, None)

    @classmethod
    def MTP(cls, r_cut : float = DEFAULTS["r_cut"], 
            mtp_poly_min : int = 1,
            mtp_poly_max : int = 2) :
        """Compute MPT descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

            mtp_poly_min: int 
                Minimum polynomials degree for radial function

            mtp_poly_max: int 
                Maximum polynomials degree for radial function                
                
        """

        param = {}
        param['descriptor_type'] = 100
        param['r_cut'] = r_cut
        param['mtp_poly_min'] = mtp_poly_min
        param['mtp_poly_max'] = mtp_poly_max

        mtp_rad_order = mtp_poly_max - mtp_poly_min + 1
        return cls(param, mtp_rad_order + 2*mtp_rad_order**2 + mtp_rad_order**3)

    @classmethod
    def PiP(cls, r_cut : float = DEFAULTS["r_cut"],
            lbody_order : list[bool] = [True for _ in range(5)],
            body_D_max : list[int] = [i for i in range(5)],
            bond_dist_transform : float = 3.0, 
            bond_beta : float = 2.0,
            bond_dist_ann : float = 1.0) :
        """Compute PiP descriptors for a given subset of configurations.
        More informations are avaible at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
        
        Parameters 
        ----------

            r_cut: float 
                Cut-off raduis for descriptor calculation

                lbody_order: list[bool] (with len(body_order) = 4)
                    List of boolean which specifies cluster expansion order used for calculation
                
                body_D_max: list[int] (with len(body_D_max) = 4)
                    List of PiP number term for each cluster expansion order

                Other parameters have to specified...
        """

        param = {}
        param['descriptor_type'] = 200
        param['rcut'] = r_cut
        param['lbody_order'] = lbody_order
        param['body_D_max'] = body_D_max
        param['bond_dist_transform'] = bond_dist_transform
        param['bond_beta'] = bond_beta
        param['bond_dist_ann'] = bond_dist_ann

        return cls(param, None)
    
    @classmethod 
    def ACE(cls, r_cut : float = DEFAULTS["r_cut"]) : 
        """Compute ACE descriptors for a given subset of configurations. Work in progress for this descriptor...                
                
        """
        raise MiladySetupError('Not yet implemented...')

    @classmethod
    def Kernel2Body(cls, r_cut : float = DEFAULTS["r_cut"],
                    sigma_2b : float = 0.3,
                    delta_2b : float = 0.1,
                    np_radial_2b : int = 50) : 
        param = {}
        param['descriptor_type'] = None
        param['activate_k2b'] = True
        param['sigma_2b'] = sigma_2b
        param['delta_2b'] = delta_2b
        param['np_radial_2b'] = np_radial_2b

        return cls(param, np_radial_2b)


    def dimensionality(self) :
        if self.dimension is not None :  
            print('Descriptor dimensionality is %3d'%(self.dimension))
        else : 
            print('No information on descriptor dimensionality...')


def DescriptorsHybridation(descriptor1 : Descriptor, descriptor2 : Descriptor) -> Descriptor :
    """Build the hybrid descriptor from descriptor1 and descriptor2 
    
    Parameters 
    ----------

        descriptor1: Descriptor
            First descriptor object

        descriptor2: Descriptor
            Second descriptor object

    Returns: 
    --------

        Descriptor 
            Hybrid decriptor from descriptor1 and descriptor2

    """
    
    def change_desc_type(type1 : int, type2 : int) -> int : 
        """Build the hybrid type from type1 and type2"""
        mix_type = np.sort([type1,type2])
        return int('%s%s'%(mix_type[0],mix_type[1])) 

    def check_hybridation(mixed_type : int) -> None :
        """Check if built hybrid type is impelemented in Milady"""
        implemented_mixed_type = [14,19]
        if mixed_type not in implemented_mixed_type : 
            raise MiladySetupError('This hybridation is not yet implemented...')

    if descriptor1.param['descriptor_type'] is None : 
        new_type = descriptor2.param['descriptor_type']
    elif descriptor2.param['descriptor_type'] is None : 
        new_type = descriptor1.param['descriptor_type']
    else :
        new_type = change_desc_type(descriptor1.param['descriptor_type'],descriptor2.param['descriptor_type'])
        check_hybridation(new_type)

    param_descriptor2 = descriptor2.param
    for key_descriptor in param_descriptor2 : 
        if key_descriptor not in descriptor1.param : 
            descriptor1.param[key_descriptor] = param_descriptor2[key_descriptor]
    descriptor1.param['descriptor_type'] = new_type
    
    if descriptor1.dimension is not None and descriptor2.dimension is not None : 
        descriptor1.dimension += descriptor2.dimension
    else : 
        descriptor1.dimension = None

    return descriptor1

class DBManager : 
    """General DBManager class for Milady code. More information about optimiser options are avaible
    at :  https://ai-atoms.github.io/milady-docs/contents/ml/descriptors.html
    
    DBModel manages writing of mld files and training setup to adjust ML potential. 
    Main methods for DBModel are : 
        - write_db_model_in, which manages to write db_model.in file for Milady fits
        - prepare_db_ml, which writes all files necessary to use Milady and Milady poscars associated
        to the database  

    Parameters 
    ----------

        atoms: Union[Atoms,List[Atoms]]
            Database to compute based on Atoms objects. Database can be composed of one atom object or 
            a list of Atoms objects

        model_ini_dict : Dict[str,Any]
            Dictionnary where keys are poscar name given for each configurations. This dictionnary contains
            other dictionnary with the following keys : atoms, energy, forces, stress

            Example : 
                {'00_000_000001' : {'atoms':Atoms,'energy':float,'forces':np.ndarray,'stress':np.ndarray} }
    
            if model_ini_dict is None, this dictionnary will be buit automatically with only one class for Milady training

    """
    from ase import Atoms
    def __init__(self, atoms : Union[Atoms,List[Atoms]] = [], model_ini_dict : Dict[str,DBtype] = None, already_set=False) :
        if model_ini_dict is None : 
            if isinstance(atoms, list) :
                if already_set :
                    self.model_init_dic : Dict[str,DBtype] = {'00_000_%s'%(str(1000000+i)[1:]) : {'atoms':atoms[i],'energy':None,'forces':atoms[i].get_array('forces'),'stress':None} for i in range(len(atoms)) }
                else :   
                    self.model_init_dic : Dict[str,DBtype] = {'00_000_%s'%(str(1000000+i)[1:]) : {'atoms':atoms[i],'energy':None,'forces':None,'stress':None} for i in range(len(atoms)) }
            else : 
                self.model_init_dic : Dict[str,DBtype] = {'00_000_000001' : {'atoms':sort(atoms),'energy':None,'forces':None,'stress':None} }
        else : 
            self.model_init_dic = model_ini_dict

            # Atoms objects have to be sorted for milady
            for key in self.model_init_dic.keys() : 
                self.model_init_dic[key]['atoms'] = sort(self.model_init_dic[key]['atoms'])

        self.dic_class = {}

    def write_db_model_in(self, directory : str, constraint_list : List[Dict[str,Any]], percentage_train : List[float]) -> None : 
        """Write db_model.in file for Milady code. More informations are available at : 
        https://ai-atoms.github.io/milady-docs/contents/ml/database.html
        
        Parameters
        ----------

            directory: str
                Directory of calculations 

            constraint_list: List[Dict[str,Any]] 
                List which contains dictionnary for each unique class in self.model_init_dic :
                {'energy':['T','1.e0','1.e0'],'forces':['T','1.e0','1.e0'],'stress':['T','1.e0','1.e0']}

            For each key, first element of list is : T(rue) or F(alse) to train ML potential on this property
                          second element of list is : float which is the minimal weight associated to the class 
                          third element of list is : float which is the maximal weight associated to the class 
        
            percentage_train: List[float]
                Percentage of train for each class in the database
        """
        
        if constraint_list is None : 
            constraint_list = [{'energy':['T','1.e0','1.e0'],'forces':['T','1.e0','1.e0'],'stress':['T','1.e0','1.e0']} for _ in range(len(self.dic_class))]        
        
        if percentage_train is None : 
            percentage_train = 1.0

        with open(os.path.join(directory,'db_model.in'),'w') as f :
            for id_c, c in enumerate(self.dic_class.keys()) :
                nb_tot = self.dic_class[c]
                nb_train = int(self.dic_class[c]*percentage_train[id_c])
                
                dic_c = constraint_list[id_c]
                bool_str = ''
                float_str = ''
                for key in dic_c : 
                    bool_str += '%s '%(dic_c[key][0])
                    float_str += '%s %s '%(dic_c[key][1],dic_c[key][2])
                f.write('%s %s %5d %5d %s %s \n'%(c.split('_')[0],c.split('_')[1],nb_train,nb_tot,bool_str[:-1],float_str))


    def prepare_db_ml(self, directory = '/.',
                      constraint_list : List[dict] = None,
                      percentage_train : Union[List[float],float] = None) -> None : 
        """Write all Milady poscars associated to the database store in DBManager object.
        This method also call method to write associated db_model.in file. 


        Parameters
        ----------

        directory: str
            Directory of calculations

        constraint_list: List[dict]
                List which contains dictionnary for each unique class in self.model_init_dic :
                {'energy':['T','1.e0','1.e0'],'forces':['T','1.e0','1.e0'],'stress':['T','1.e0','1.e0']}
                
                For each key, first element of list is : T(rue) or F(alse) to train ML potential on this property
                              second element of list is : float which is the minimal weight associated to the class 
                              third element of list is : float which is the maximal weight associated to the class 
                
        percentage_train: Union[List[float],float]
            List of train percentage associated at each class. If input is a float, training percentage is set to 
            this value for each class
                
        """

        if not os.path.exists(os.path.join(directory,'DB')) : 
            os.mkdir(os.path.join(directory,'DB'))
        else : 
            shutil.rmtree(os.path.join(directory,'DB'))
            os.mkdir(os.path.join(directory,'DB'))

        for key in self.model_init_dic : 
            prefix = key[0:6]
            if prefix not in self.dic_class : 
                self.dic_class[prefix] = 1
            else : 
                self.dic_class[prefix] += 1

            if len(key.split('.')) > 1 : 
                name = key.split('.')[0]
            else :
                name = key

            write_milady_poscar('%s/DB/%s.poscar'%(directory,name),
                                self.model_init_dic[key]['atoms'],
                                self.model_init_dic[key]['energy'],
                                self.model_init_dic[key]['forces'],
                                self.model_init_dic[key]['stress'])
            

        if isinstance(percentage_train,float) : 
            percentage_train = [percentage_train for _ in range(len(self.dic_class))]

        self.write_db_model_in(directory, constraint_list, percentage_train) 
        return None

class Milady(Calculator):  # type: ignore
    """ASE interface for the Vienna Ab initio Simulation Package (VASP),
    with the Calculator interface.

        Parameters

            atoms:  object
                Attach an atoms objewrite_descrct to the calculator.

            label: str
                Prefix for the output file, and sets the working directory.
                Default is 'vasp'.

            directory: str
                Set the working directory. Is prepended to ``label``.

            restart: str or bool
                Sets a label for the directory to load files from.
                if :code:`restart=True`, the working directory from
                ``directory`` is used.

            txt: bool, None, str or writable object
                - If txt is None, output stream will be supressed

                - If txt is '-' the output will be sent through stdout

                - If txt is a string a file will be opened,\
                    and the output will be sent to that file.

                - Finally, txt can also be a an output stream,\
                    which has a 'write' attribute.

                Default is 'vasp.out'

                - Examples:

                    >>> Vasp(label='mylabel', txt='vasp.out') # Redirect stdout
                    >>> Vasp(txt='myfile.txt') # Redirect stdout
                    >>> Vasp(txt='-') # Print vasp output to stdout
                    >>> Vasp(txt=None)  # Suppress txt output

            command: str
                Custom instructions on how to execute VASP. Has priority over
                environment variables.
    """
    name = 'milady'
    ase_objtype = 'milady_calculator'  # For JSON storage

    implemented_properties = [
        'potential', 'milady-descriptor', 'milady-descriptor-forces', 'milady-descriptor-stress'
    ]
    from ase import Atom, Atoms

    def __init__(self, 
                 optimizer : Optimiser,
                 regressor : Regressor,
                 descriptor : Descriptor,
                 dbmodel : DBManager = None,
                 atoms : Union[Atoms,List[Atoms]] = None,
                 restart : bool = None,
                 label = 'milady',
                 directory : str = '.',
                 milady_command : str = None,
                 mpi_command : str = None,
                 ncpu : int = 1) :
        

        self.check_milday_calc_inputs(optimizer,regressor,descriptor)

        self.optimizer = optimizer
        self.regressor = regressor
        self.descriptor = descriptor
        self.dbmodel = dbmodel

        if atoms is None and self.dbmodel is not None : 
            self.atoms = [ self.dbmodel.model_init_dic[key]['atoms'] for key in self.dbmodel.model_init_dic.keys() ]
        
        self.restart = restart
        self.label = label
        self.directory = directory
        self.milady_command = milady_command
        self.mpi_command = mpi_command
        self.ncpu = ncpu

        self.txt = None
        self.input_generator = GenerateMiladyInput(optimizer.param,regressor.param,descriptor.param)

    @contextmanager
    def _txt_outstream(self):
        """Custom function for opening a text output stream. Uses self.txt
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes
        the new stream again when exiting.

        Examples:
        # Pass a string
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'

        # Use an existing stream
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout
        """

        txt = self.txt
        open_and_close = False  # Do we open the file?

        if txt is None:
            # Suppress stdout
            out = subprocess.DEVNULL
        else:
            if isinstance(txt, str):
                if txt == '-':
                    # subprocess.call redirects this to stdout
                    out = None
                else:
                    # Open the file in the work directory
                    txt = self._indir(txt)
                    # We wait with opening the file, until we are inside the
                    # try/finally
                    open_and_close = True
            elif hasattr(txt, 'write'):
                out = txt
            else:
                raise RuntimeError('txt should either be a string'
                                   'or an I/O stream, got {}'.format(txt))

        try:
            if open_and_close:
                out = open(txt, 'w')
            yield out
        finally:
            if open_and_close:
                out.close()

    def check_milday_calc_inputs(self,optimiser,regressor,descriptor) : 
        """Some sanity checks for key objects needed for Milady calculator"""
        self.check_optimiser(optimiser)
        self.check_regressor(regressor)
        self.check_descriptor(descriptor)

    def check_optimiser(self,optimiser) : 
        """Check type of optimiser"""
        if not isinstance(optimiser,Optimiser) :
            raise MiladySetupError(('Expected an Optimiser object, '
                 'instead got object of type {}'.format(type(optimiser))))
        
    
    def check_regressor(self,regressor) :
        """Check type of regressor""" 
        if not isinstance(regressor,Regressor) :
            raise MiladySetupError(('Expected an Regressor object, '
                 'instead got object of type {}'.format(type(regressor))))

    def check_descriptor(self,descritpor) :
        """Check type of descriptor""" 
        if not isinstance(descritpor,Descriptor) :
            raise MiladySetupError(('Expected an Descriptor object, '
                 'instead got object of type {}'.format(type(descritpor))))     
    
    def make_command_milady(self, command=None):
        """Return command if one is passed, otherwise try to find
        'ASE_MILADY_COMMAND', 'MILADY_COMMAND', 'MILADY_SCRIPT'
        If none are set, a CalculatorSetupError is raised"""
        
        milady_commands = ('ASE_MILADY_COMMAND', 'MILADY_COMMAND', 'MILADY_SCRIPT')
        
        if command is not None:
            return command
        
        else:
            # Search for the environment commands
            for env in milady_commands:
                if env in os.environ:
                    return os.environ[env]
            
            msg = ('Please set either command in calculator'
                       ' or one of the following environment '
                       'variables (prioritized as follows): {}').format(
                           ', '.join(milady_commands))
            raise calculator.CalculatorSetupError(msg)
    
    def make_command_mpi(self, command=None):
        """Return command if one is passed, otherwise try to find
        mpirun command
        If none are set, a CalculatorSetupError is raised"""
        mpi_commands = ('MPI_COMMAND',)

        if command is not None :
            return command
        else:
            # Search for the environment commands
            for env in mpi_commands:
                if env in os.environ:
                    return os.environ[env]
                
            msg = ('Please set either command in calculator'
                       ' or one of the following environment '
                       'variables (prioritized as follows): {}').format(
                           ', '.join(mpi_commands))
            raise calculator.CalculatorSetupError(msg)

    def check_atoms(self, atoms : Union[Atoms,List[Atoms]]) -> None:
        if isinstance(atoms,list) : 
            for config in atoms : 
                for check in (self.check_atoms_type, self.check_pbc, self.check_atoms_type) :
                    check(config)
                    config = sort(config)

        else : 
            for check in (self.check_atoms_type, self.check_pbc, self.check_atoms_type) :
                check(atoms)
                atoms = sort(atoms)

    def check_cell(self, atoms : Atoms) -> None:
        """Check if there is a zero unit cell.
        Raises CalculatorSetupError if the cell is wrong.
        """
        if atoms.cell.rank < 3:
            raise calculator.CalculatorSetupError(
                "The lattice vectors are zero! "
                "This is the default value - please specify a "
                "unit cell.")


    def check_pbc(self, atoms:  Atoms) -> None:
        """Check if any boundaries are not PBC, as VASP
        cannot handle non-PBC.
        Raises CalculatorSetupError.
        """
        if not atoms.pbc.all():
            raise calculator.CalculatorSetupError(
                "Vasp cannot handle non-periodic boundaries. "
                "Please enable all PBC, e.g. atoms.pbc=True")


    def check_atoms_type(self, atoms : Atoms) -> None:
        """Check that the passed atoms object is in fact an Atoms object.
        Raises CalculatorSetupError.
        """
        if not isinstance(atoms, ase.Atoms):
            raise calculator.CalculatorSetupError(
                ('Expected an Atoms object, '
                 'instead got object of type {}'.format(type(atoms))))

    def check_db_manager(self,atoms) : 
        """Check ig DBManager has been correctly setup. If DBManager is None, a default one is set up.
        If DBManager is not correctly setup raises MiladySetupError 
        """
        if self.dbmodel is None : 
            self.dbmodel = DBManager(atoms,model_ini_dict=None)
            Warning('DBManager is not defined for your calculation, default DBmanager will be setup...')
        else : 
            if not isinstance(self.dbmodel,DBManager) : 
                raise MiladySetupError('Expected an DBManager object, '
                 'instead got object of type {}'.format(type(self.dbmodel)))

    def clear(self) :
        """Clear the milady directory for calculation""" 
        if os.path.exists(self.directory) : 
            shutil.rmtree(self.directory)

    def calculate(self,
                  atoms : Union[Atoms,List[Atoms]] = None,
                  properties=['milady_descriptors']):
        """Do a Milady calculation in the specified directory.

        This will generate the necessary Milady input files, and then
        execute Milady. After execution, properties are extracted
        from the Milady output files.
        """
        # Some sanity checks before milady calculation ...
        
        if atoms is None : 
            atoms = self.atoms
        
        self.clear()
        self.check_atoms(atoms)
        self.check_db_manager(atoms)
        self.input_generator.check_descriptor()

        # setting command for milady
        milady_command = self.make_command_milady(self.milady_command)
        mpi_command = self.make_command_mpi(self.mpi_command)

        if not os.path.exists(self.directory) : 
            os.mkdir(self.directory)

        # writing all milday files
        self.input_generator.copy_standard_ml_files(directory=self.directory,label=self.label)
        self.input_generator.write_ml_file(directory=self.directory,name_file=self.label)
        self.dbmodel.prepare_db_ml(self.directory,constraint_list=None,percentage_train=1.0)

        with self._txt_outstream() as out:
            launch_command = '%s %s %s'%(mpi_command,str(self.ncpu),milady_command)
            print(launch_command)
            errorcode = self._run(command=launch_command,
                                  out=out,
                                  directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d}'.format(
                    self.name, self.directory, errorcode))

        # Read results from calculation
        self.read_results(properties, self.dbmodel)


    def _run(self, command : str = None, out : str = None, directory : str = None):
        """Method to explicitly execute Milady"""
        if command is None : 
            raise calculator.CalculatorSetupError('Launching command for Milady is not defined')

        if directory is None:
            directory = self.directory
        errorcode = subprocess.call(command,
                                    shell=True,
                                    stdout=out,
                                    cwd=directory)
        
        return errorcode
    
    def read_results(self, properties : List[str], dbmodel : DBManager) :
        implemented_local_properties = ['milady-descriptors','milady-descriptors-forces']
        dic_equiv_properties = {'milady-descriptors':'eml',
                                'milady-descriptors-forces':'fml'}
        if dbmodel is not None : 
            for property in properties : 
                if property not in implemented_local_properties : 
                    print('Selected property {:} is not implemented'.format(property))
                    continue
                else : 
                    for key in dbmodel.model_init_dic.keys() : 
                        if property in implemented_local_properties : 
                            dbmodel.model_init_dic[key]['atoms'] = fill_milady_descriptor(dbmodel.model_init_dic[key]['atoms'],
                                                                       '{:}/descDB/{:}'.format(self.directory,key),
                                                                       name_property=property,
                                                                       ext=dic_equiv_properties[property])
                        else :
                            raise calculator.CalculatorSetupError("This property is not yet implemented")
                    
                