from __future__ import annotations
from typing import Any,List, Dict, TypedDict, Optional
import numpy as np
from scipy import integrate

class derivated_dic_babf(TypedDict) :
    sum_w_AxU_dl : np.ndarray
    sum_w_AxU2_dl : np.ndarray
    sum_w_A : np.ndarray
    sum_pc_lam_q : np.ndarray
    pc_lam_q : np.ndarray
    compt : float
    temperature : float
    sum_delta_W : float | None

class ResultsBABF:
    """BABF Results object, 
    a dictionary which contains input and output values

    Methods
    -------
    __call__
    has_key
    """
    def __init__(self, lambda_grid : dict, omega : float = 0.1, block : bool = False) -> None:
        
        def convert_lambda_dict(lambda_dict) -> np.ndarray : 
            """Convert the lambda dictionnary into np array grid"""
            buffer_array_left = np.linspace(lambda_dict['Min']-lambda_dict['Rbuffer'],lambda_dict['Min'], num=lambda_dict['NumberBuffer'], endpoint=False)
            buffer_array_right = np.linspace(lambda_dict['Max'], lambda_dict['Max']+lambda_dict['Rbuffer'], num=lambda_dict['NumberBuffer']+1)
            main_array_lambda = np.linspace(lambda_dict['Min'], lambda_dict['Max'], num=lambda_dict['Number'], endpoint=False)

            if len(buffer_array_left) > 0 :
                main_array_lambda = np.concatenate((buffer_array_left,main_array_lambda),axis=0)

            return  np.concatenate((main_array_lambda,buffer_array_right), axis=0)
        
        self.kB =  8.617333262e-5
        self.lambda_grid = convert_lambda_dict(lambda_grid)
        self.data_babf : derivated_dic_babf = {'sum_w_AxU_dl':np.zeros(len(self.lambda_grid)),
                                               'sum_w_AxU2_dl':np.zeros(len(self.lambda_grid)),
                                               'sum_w_A': omega*np.ones(len(self.lambda_grid)),
                                               'sum_pc_lam_q':np.zeros(len(self.lambda_grid)),
                                               'pc_lam_q':np.zeros(len(self.lambda_grid)),
                                               'compt':0.0,
                                               'temperature':None,
                                               'sum_delta_W':0.0}
        self.free_energy = None
        self.var_free_energy = None
        
        if block : 
            self.data_babf_block : derivated_dic_babf = {'sum_w_AxU_dl':np.zeros(len(self.lambda_grid)),
                                                         'sum_w_AxU2_dl':np.zeros(len(self.lambda_grid)),
                                                         'sum_w_A': omega*np.ones(len(self.lambda_grid)),
                                                         'sum_pc_lam_q':np.zeros(len(self.lambda_grid)),
                                                         'pc_lam_q':np.zeros(len(self.lambda_grid)),
                                                         'compt':0.0,
                                                         'temperature':None,
                                                         'sum_delta_W':None}
            self.free_energy_block = None
        else : 
            self.data_babf_block = {}
            self.free_energy_block = None

    def close_index_values(self, reference_value : float = 1.0) -> int :
        """Returns the closest index value for lamda grid
        
        Returns: 
        --------

        int 
            Closest index
        """
        return np.argmin(np.abs(self.lambda_grid-reference_value))

    def update_temperature(self, temperature : float, block : bool = False) -> None :
        """Update temperature of the simulation for each step
        
        temperatrure : float 
            Temperature of the system

        block : bool 
            Boolean for block BABF method
        """
        
        if block : 
            self.data_babf_block['temperature'] = temperature
        
        self.data_babf['temperature'] = temperature
        return 

    """FREE ENERGY DERIVATIVE UPDATE"""
    def update_data_babf(self, data : np.ndarray, data_key : str = 'sum_w_AxU_dl', block : bool = False) -> None : 
        """Update free energy object
        Parameters 
        ----------
        data : np.ndarray 
            data to add for ergodic average
        data_key : str 
            name of data to update : sum_w_AxU_dl, sum_w_A, pc_lam_q
        block : bool 
            key word for constrained babf method"""
        if block : 
            self.data_babf_block[data_key] += data
        else : 
            self.data_babf[data_key] += data

    def evaluate_derivative_free_energy(self, block : bool = False) -> np.ndarray :
        """Evaluate \partial_{\lambda} A(\lambda) for ABF
        
        Returns
        -------
        np.ndarray 
            \partial_{\lambda} A(\lambda) = \sum_{i}^{s} [ U(q_i,\lambda) - U_{ref}(q_i,\lambda) ] p_A(\lambda | q_i) / \sum_{i}^{s} p_A(\lambda | q_i)
        """
        if block : 
            return self.data_babf_block['sum_w_AxU_dl']/self.data_babf_block['sum_w_A']
        else :
            return self.data_babf['sum_w_AxU_dl']/self.data_babf['sum_w_A']

    def evaluate_variance_mean_force(self) -> np.ndarray : 
        """Evaluate \partial_{\lambda} A(\lambda) for ABF
        
        Returns
        -------
        np.ndarray 
            \partial_{\lambda} A(\lambda) = \sum_{i}^{s} [ U(q_i,\lambda) - U_{ref}(q_i,\lambda) ] p_A(\lambda | q_i) / \sum_{i}^{s} p_A(\lambda | q_i)
        """

        return self.data_babf['sum_w_AxU2_dl']/self.data_babf['sum_w_A'] - np.power( self.data_babf['sum_w_AxU_dl']/self.data_babf['sum_w_A'], 2.0)

    def integration_scheme(self,x : np.ndarray, y : np.ndarray) -> np.ndarray :
        """Simpson scheme for integration  
        Parameters
        ----------
        x : np.ndarray
        y : np.ndarray 
            y = f(x)

        Returns 
        -------
        float 
            int_{support(x)} f(x)dx 

        """
        return integrate.simps(y,x)

    """MIXING METHODS"""
    def evaluate_U_lambda(self, reference_potential : float, lammps_potential : float) -> np.ndarray:
        """Build the mixing potential for TI
        Parameters
        ----------
        reference_potential : float
            U_{ref}(q)
        lammps_potential : float 
            U(q)

        Returns 
        -------
        np.ndarray
            Mixing potential array \lambda U(q) + (1 - \lambda) U_{ref}(q)       

        """
        return self.lambda_grid*lammps_potential + (1 - self.lambda_grid)*reference_potential

    def evaluate_forces_lambda(self,f_reference : np.ndarray, f_lammps : np.ndarray) -> np.ndarray :
        """build the mixing force tensor for stochastic dynamic
        Parameters
        ----------
        f_reference : np.ndarray
            reference forces f_{ref}(q) 
        f_lammps : np.ndarray 
            forces from lammps f(q)

        Returns 
        -------
        np.ndarray : tensor shape is (Nat,3,len(lambda_grid))
            Mixing force array  F^{mix}_{\lambda}(q) = \lambda f(q) + (1 - \lambda) f_{ref}(q)        
        """
        return np.tensordot(f_lammps,self.lambda_grid,axes=0) + np.tensordot(f_reference,(1-self.lambda_grid),axes=0)


    """CANONICAL MEASURE METHODS"""
    def canonical_lambda_measure(self,temperature : float, potential : np.ndarray, free_energy : np.ndarray) -> np.ndarray :
        """Compute extended canonical measure for a given position 
        Parameters
        ----------
        temperature : float 
            Temperature of sampling
        potential : np.ndarray 
            mixing potential energy depending on lambda (len(lambda_grid))
        free energy : np.ndarray
            free energy estimator depending on lambda (len(lambda_grid))

        Returns 
        -------
        np.ndarray
            evaluation of the extended canonical measure  exp(- \beta [ U_lam(q) - A_lam ] )
        """
        return np.exp(-(potential-free_energy)/(self.kB*temperature))

    def evaluate_conditional_p_lambda(self, temperature : float, potential : np.ndarray, free_energy : np.ndarray) -> np.ndarray :
        """Estimate extended p_A(lambda|q)
        Parameters
        ----------
        temperature : float 
            Temperature of sampling
        potential : np.ndarray 
            mixing potential energy depending on lambda (len(lambda_grid))
        free energy : np.ndarray
            free energy estimator depending on lambda (len(lambda_grid))

        Returns 
        -------
        np.ndarray
            evaluation of the extended  p_A(lambda|q) = exp(- \beta [ U_lam(q) - A_lam ] ) / \int_{0}^{1} exp(- \beta [ U_lam'(q) - A_lam' ] ) dlam'
        """       
     
        """here factorised by min exponent is needed to avoid overflow..."""
        min_expo = np.min(potential-free_energy)
        partition_function = self.integration_scheme(self.lambda_grid,self.canonical_lambda_measure(temperature,potential,free_energy+min_expo))
        conditionnal_p_lambda = self.canonical_lambda_measure(temperature,potential,free_energy+min_expo)/partition_function
        return conditionnal_p_lambda

    def evaluate_conditional_p_lambda_constrained(self, temperature : float, potential : np.ndarray, potential_c : np.ndarray, free_energy_c : np.ndarray) -> np.ndarray :
        ## TO DO !
        """Estimate extended p_A(lambda|q)
        Parameters
        ----------
        temperature : float 
            Temperature of sampling
        potential : np.ndarray 
            mixing potential energy depending on lambda (len(lambda_grid))
        potential_c : np.ndarray 
            mixing constrained potential energy depending on lambda (len(lambda_grid))
        free energy_c : np.ndarray
            constrained free energy estimator depending on lambda (len(lambda_grid))

        Returns 
        -------
        np.ndarray
            evaluation of the extended  p^c_A(lambda|q) = exp(- \beta [ U_lam(q) - A^c_lam ] ) / \int_{0}^{1} exp(- \beta [ U^c_lam'(q) - A^c_lam' ] ) dlam'
        """       
     
        """here factorised by min exponent is needed to avoid overflow..."""
        min_expo = np.min([np.min(potential-free_energy_c),np.min(potential_c-free_energy_c)])
        partition_function_c = self.integration_scheme(self.lambda_grid,self.canonical_lambda_measure(temperature,potential_c,free_energy_c+min_expo))
        conditionnal_p_lambda_c = self.canonical_lambda_measure(temperature,potential,free_energy_c+min_expo)/partition_function_c
        return conditionnal_p_lambda_c

    """UPDATE DYNAMICS"""
    def evaluate_average_force(self,forces_lambda : np.ndarray, conditional_p_lambda : np.ndarray) -> np.ndarray :
        """Integrating the effective force over p_A(\lambda|q)
        ----------
        forces_lambda : np.ndarray 
            Mixing force array  F^{mix}_{\lambda}(q) = \lambda f(q) + (1 - \lambda) f_{ref}(q)
        conditional_p_lambda : np.ndarray 
            p_A(lambda|q) = exp(- \beta [ U_lam(q) - A_lam ] ) / \int_{0}^{1} exp(- \beta [ U_lam'(q) - A_lam' ] ) dlam'

        Returns 
        ------- 
        np.ndarray
            F(q) the effective force for stochastic dynamics 
        """
        average_forces = np.zeros((forces_lambda.shape[0],forces_lambda.shape[1]))
        for k in range(forces_lambda.shape[0]):
            for xi in range(forces_lambda.shape[1]):
                product_k_xi = forces_lambda[k,xi,:]*conditional_p_lambda
                int_simpson_k_xi = self.integration_scheme(self.lambda_grid,product_k_xi)
                average_forces[k,xi] = int_simpson_k_xi

        return average_forces


    def evaluate_effective_forces_dynamic(self, f_reference : np.ndarray, f_lammps : np.ndarray, block : bool = False) -> np.ndarray :
        """Evaluate effective forces F(q) for stochastic dynamic 
        Parameters
        ----------
        f_reference : np.ndarray 
            forces from reference model : f_{ref}(q)
        f_lammps : np.ndarray 
            forces from lammps potential : f(q)
        block : bool 
            Key word for constrained BABF method

        Returns 
        ------- 
        np.ndarray
            F(q) the effective force for stochastic dynamics 
        """       
        mixing_force_lambda = self.evaluate_forces_lambda(f_reference,f_lammps)
        if block :
            return self.evaluate_average_force(mixing_force_lambda,self.data_babf_block['pc_lam_q'])
        else : 
            return self.evaluate_average_force(mixing_force_lambda,self.data_babf['pc_lam_q'])

    """FREE ENERGY ESTIMATIONS"""
    def evaluate_free_energy(self, block : bool = False) -> np.ndarray : 
        """Evaluate free energy by integration mean force over lambda
        Parameters
        ----------
        block : bool 
            Key word for constrained BABF method
        
        Returns:
        --------

        np.ndarray 
            Extended free energy estimation

        """               
        #A_lambda = np.zeros(len(self.lambda_grid))
        partial_lambda_A = self.evaluate_derivative_free_energy(block=block)
        
        #for k in range(len(partial_lambda_A)):
        #    sub_partial_lambda_A = partial_lambda_A[:k+1]
        #    sub_lambda = self.lambda_grid[:k+1]
        #    A_lambda[k] = self.integration_scheme(sub_lambda,sub_partial_lambda_A)

        return integrate.cumulative_simpson(partial_lambda_A, self.lambda_grid)

    def evaluate_variance_free_energy(self) -> np.ndarray : 
        """Evaluate free energy by integration mean force over lambda
        Parameters
        ----------
        block : bool 
            Key word for constrained BABF method
        
        Returns:
        --------

        np.ndarray 
            Extended free energy estimation

        """               
        #var_A_lambda = np.zeros(len(self.lambda_grid))
        var_mean_force = self.evaluate_variance_mean_force()
        #for k in range(len(var_mean_force)):
        #    sub_var_A_lambda = var_mean_force[:k+1]
        #    sub_lambda = self.lambda_grid[:k+1]
        #    var_A_lambda[k] = self.integration_scheme(sub_lambda,sub_var_A_lambda)

        self.var_free_energy = integrate.cumulative_simpson(var_mean_force, self.lambda_grid)
        return self.var_free_energy

    def extract_estimator(self, block : bool = False) -> dict : 
        """Extract free energy quantities from langevin dynamics estimator 
        
        Parameters:
        -----------

        block : bool 
            Key word for block BABF method

        Returns:
        --------

        dict
            dictionnary containing free energy estimator to print
                  
        """
        index_anah = self.close_index_values(self.lambda_grid, reference_value=1.0)
        mean_force = self.evaluate_derivative_free_energy(block=False)
        if block : 
            return {'Temperature':self.data_babf_block['temperature'],'<d_lambda F(1)>':mean_force[index_anah], 'A(1)':self.free_energy[index_anah], 
                    'A_b(1)':self.free_energy_block[index_anah],'<A(1)^2> - <A(1)>^2':self.var_free_energy[index_anah]}
        
        else : 
            return {'Temperature':self.data_babf['temperature'],'<d_lambda F(1)>':mean_force[index_anah], 'A(1)':self.free_energy[index_anah], 
                    '<A(1)^2> - <A(1)>^2':self.var_free_energy[index_anah]}

    def update_free_energy(self, block : bool = False) -> None :
        """Update free energy estimator
        
        Parameters
        ----------
        block : bool 
            Key word for block BABF method

        """
        if block : 
            self.free_energy = self.evaluate_free_energy(block=False)
            self.free_energy_block = self.evaluate_free_energy(block=True)
        else : 
            self.free_energy = self.evaluate_free_energy(block=block)

        return

    def items(self)-> Any: # TODO what is the type here?
        """return items for iteration
        Returns
        -------
        dictionary.items()
        """
        return self.data_babf.items()
    
    def __call__(self,key:str)->Any:
        """_summary_

        Parameters
        ----------
        key : str
            dictionary key

        Returns
        -------
        Any
           dictionary value
        
        Raises
        ------
        ValueError
            If key not found
        """
        if self.has_key(key):
            return self.data_babf[key]
        else:
            raise ValueError(f"No key {key} in Results!")
        
    def set(self,key:str,value:Any)->None:
        """Set key,value pair in internal data

        Parameters
        ----------
        key : str
        value : Any
        """
        self.data_babf[key] = value

    def has_key(self,key:str)->bool:
        """Check if key exists

        Parameters
        ----------
        key : str
           key query

        Returns
        -------
        bool
            True if exists
        """
        return bool(key in self.data_babf.keys())
    
    def set_dict(self,dict:dict)->None:
        """Write or overwrite to results from dictionary
        writes key,value pairs via set()

        Parameters
        ----------
        dict : dict
        """
        for key,value in dict.items():
            self.set(key,value)
    def get_dict(self,keys:List[str],blanks:None|str=None)->dict:
        """Returns dictionary with given keys 

        Parameters
        ----------
        keys : List[str]
            list of keys
        blanks : None or str
            If not None, return str for missing key, default None

        Returns
        -------
        dict
            dictionary with values determined from internal data
        """
        res = {}
        for key in keys:
            if not blanks is None:
                res[key] = self.data[key] if self.has_key(key) else blanks
            else:
                res[key] = self.__call__(key)
        return res

