import numpy as np 
import ase.io
from scipy.linalg import null_space
from ..hpc.ExtractAllFromVASP import GetElementsFromVasp, GetNatomFromVasp
import os
from typing import List, Dict, TypedDict

"""Here are the derivated types used in data_surface object """
class elements(TypedDict) : 
    single : Dict[str,float]
    bulk : Dict[str,float]
    ecoh : float

class bulk(TypedDict) : 
    energy : float
    composition : List[List[float]]
    hf : float
    ecoh : float

class slab(TypedDict) : 
    energy : float
    composition : List[List[float]]
    surface : float
    bound : dict

class dic_type(TypedDict) : 
    elements : Dict[str,elements]
    bulk : Dict[str,bulk]
    slab : Dict[str,slab]
######################################################################

class DataSurface : 
    def __init__(self):
        self.dic : dic_type = {'elements':{},'bulk':{},'slab':{}}
    
    ##################################################################
    def find_corresponding_nb(self,type : str, 
                              system : str, 
                              el : str) -> int :
        """Find the corresponding number of element el in a given system for a given type
        
        Parameters
        ----------

        type : str
            Type of key in self.dic dictionnary i.e elements (single elements), bulk (perfect system), slab (surface system)

        system : str 
            Name of the system 

        el : str 
            Name of the element 

        Returns 
        -------

        int 
            number of element el in system for a given type
        """

        return [self.dic[type][system]['composition'][1][k] for k in 
        range(len(self.dic[type][system]['composition'][1])) if self.dic[type][system]['composition'][0][k]==el][0]

    ##################################################################
    """composition is [[el1,el2,...,elN],[nb1,nb2,...,nbN]]"""
    """creating entry for each elements with normalised energy"""
    ##################################################################
    
    def add_entry_el(self, el : str, 
                     type : str,
                     e_normalised : float) -> None : 
        """Add entry in element key in self.dic
        
        Parameters 
        ----------

        el : str 
            Name of the element

        type : str 
            single or bulk to compute cohesion energy

        e_ normalised : float
            Energy per atom associated to the type 
        """

        if el not in self.dic['elements'].keys() : 
            self.dic['elements'][el] = {'single':{'energy':None},'bulk':{'energy':None},'ecoh':None}
        self.dic['elements'][el][type]['energy'] = e_normalised

    ##################################################################
    def add_entry_bulk(self,bulk : str, 
                       energy : float, 
                       composition : List[List[str] | List[int]]) -> None : 
        """Add entry in bulk key in self.dic 
        
        Parameters 
        ----------

        bulk : str 
            Name of the bulk system

        energy : float 
            Energy associated to the system 

        composition : List[List[str] | List[int]]
            Composition vector under the following form [[el1,el2,...,elN],[nb1,nb2,...,nbN]]
        """

        self.dic['bulk'][bulk] = {'energy':energy,'composition':composition,'hf':None,'ecoh':None}

    ##################################################################
    def add_entry_slab(self,slab : str, 
                       energy :float, 
                       composition : List[List[str] | List[int]]) -> None :
        """Add entry in slab key in self.dic 
        
        Parameters 
        ----------

        slab : str 
            Name of the slab system

        energy : float 
            Energy associated to the system 

        composition : List[List[str] | List[int]]
            Composition vector under the following form [[el1,el2,...,elN],[nb1,nb2,...,nbN]]
        """
        self.dic['slab'][slab] = {'energy':energy,'composition':composition,'surface':None,'bounds':None}

    ##################################################################
    def compute_surface_slab(self,slab : str, 
                             path_slab : os.PathLike[str]) -> float:
        """Compute the area of the slab with the determinant of restriced lattice matrix
        
        Parameters 
        ----------

        slab : str 
            Name of the slab system 

        path_slab : str
            Path of the slab system 

        Returns 
        -------

        float 
            Surface associated to the slab system, we assumed that the slab in oriented following the 
            001 direction
        """


        slab_obj=ase.io.read(path_slab, format='vasp')         
        cell = slab_obj.cell[:]
        #we assumed that the slab is oriented following the 001 direction
        #so we keep the restriction of the cell matrix and we compute the determinant to have the area
        cell_restricted = cell[:2,:2]
        self.dic['slab'][slab]['surface'] = np.linalg.det(cell_restricted)
        return np.linalg.det(cell_restricted)

    ##################################################################
    def compute_ecoh_single(self) -> None : 
        """Compute the cohesion energy for each single element"""
        for el in self.dic['elements'].keys() : 
            ecoh_el = self.dic['elements'][el]['bulk']['energy'] - self.dic['elements'][el]['single']['energy']
            self.dic['elements'][el]['ecoh'] = ecoh_el

    ##################################################################
    def compute_ecoh_hf(self,bulk : str) -> None : 
        """Compute the formation enthalpy and the cohesion energy for a given bulk system
        
        Parameters 
        ----------
        
        bulk : str 
            Name of the bulk system

        """
        sum_bulk_energy = 0.0
        sum_single_energy = 0.0
        #print(self.dic['bulk'][bulk]['composition'])
        for k,el in enumerate(self.dic['bulk'][bulk]['composition'][0]) :
            nb_el_in_slab = self.dic['bulk'][bulk]['composition'][1][k]
            sum_bulk_energy += nb_el_in_slab*self.dic['elements'][el]['bulk']['energy']
            sum_single_energy += nb_el_in_slab*self.dic['elements'][el]['single']['energy']
        tot_el = np.sum(self.dic['bulk'][bulk]['composition'][1])
        
        """value of hf for the bulk"""
        self.dic['bulk'][bulk]['hf'] = (self.dic['bulk'][bulk]['energy'] - sum_bulk_energy)/tot_el
        """value of ecoh for the bulk"""
        self.dic['bulk'][bulk]['ecoh'] = (self.dic['bulk'][bulk]['energy'] - sum_single_energy)/tot_el
    
    ##################################################################
    def build_bounds(self,slab : str,
                     bulk : str,
                     el_removed: str) -> Dict[str,List[np.ndarray] | List[str]]:
        """Compute bounds for all chemical potential difference for a given slab system
        
        Parameters
        ----------

        slab : str 
            Name of the slab system

        bulk : str 
            Associated bulk system 

        el_removed : str 
            Element to remove in the parametric surface energy calculation

        Returns 
        -------

        Dict[str,List[np.ndarray] | List[str]]
            Dictionnary containing the domain definition for parametric surface energy

        """
        bounds = {}
        Delta_H = self.dic['bulk'][bulk]['hf']
        sum_n = np.sum(self.dic['bulk'][bulk]['composition'][1])
        
        """here are the list needed to define the definition domain of gamma"""
        domain_list = []
        domain_el = []

        for k,el in enumerate(self.dic['bulk'][bulk]['composition'][0]) :
            bounds[el] = [Delta_H*sum_n/self.dic['bulk'][bulk]['composition'][1][k],0.0]
            if el != el_removed : 
                domain_el.append(el)
                domain_list.append(self.dic['bulk'][bulk]['composition'][1][k])

        """here is the domain vector for surface energy (V_D)"""
        #nb_el_remove = [ nb for k,nb in enumerate(self.dic['bulk'][bulk]['composition'][1]) if self.dic['bulk'][bulk]['composition'][0][k] == el_removed ]
        domain_list += [-Delta_H*sum_n]
        domain_el += ['enthalpy']
        
        """Domain is defined for \Mu=(\mu_1,...,\mu_{n-1}) such as V_D@\Mu \geq 0"""
        bounds['domain'] = [np.array(domain_list),domain_el]
        self.dic['slab'][slab]['bounds'] = bounds

        return bounds

    ##################################################################
    def compute_parametric_surface_energy(self,slab : str, 
                                          bulk : str, 
                                          path_slab : os.PathLike[str], 
                                          el_remove : str, 
                                          vect_diff_mu : np.ndarray, 
                                          el_diff_mu : List[str]) -> float :
        """Compute the parametric value surface energy depending on vect_diff_mu vector
        for a given slab system 
        
        Parameters
        ----------

        slab : str 
            Name of the slab system 

        bulk : str 
            Name of the associated bulk system 

        path_slab : str 
            Path of the slab system poscar for surface calculation

        el_remove : str 
            Element to remove in the parametric energy calculation

        vect_diff_mu : np.ndarray 
            Vector of the chemical potential difference for parametric surfac energy calculation

        el_diff_mu : List[str]
            List of elements associated to the vect_diff_mu vector 

        Returns 
        -------

        float 
            Parametric energy surface for vect_diff_mu vector
        
        """
        
        """computing the surface energy for a given vector of diff_mu
        you can give the dependancy to remove for a given el"""
        E_slab = self.dic['slab'][slab]['energy']

        surface_slab = self.compute_surface_slab(slab,path_slab)
        n_bulk_el_remove = self.find_corresponding_nb('bulk',bulk,el_remove)
        N_slab_el_remove = self.find_corresponding_nb('slab',slab,el_remove)

        
        tmp_vect = []
        for i,el in enumerate(self.dic['bulk'][bulk]['composition'][0]) : 
            """corresponding to the first term :
            (\mu_{B}^{bulk}/n^{bulk}_{el_remove})*N^{slab}_{el_remove}"""
            if el == el_remove : 
                mu_B_bulk = self.dic['bulk'][bulk]['ecoh']
                sum_el = np.sum(self.dic['bulk'][bulk]['composition'][1])
                tmp_vect.append(-mu_B_bulk*sum_el*N_slab_el_remove/n_bulk_el_remove)

            else : 
                """build (N_^{slab}_i - frac{n_i}{n_bulk_el_remove}*N_slab_el_remove)"""
                N_i = self.find_corresponding_nb('slab',slab,el)
                n_i = self.dic['bulk'][bulk]['composition'][1][i]
                tmp = (N_i - n_i*N_slab_el_remove/n_bulk_el_remove)

                """\mu_{i}^{bulk}*(N_^{slab}_i - frac{n_i}{n_bulk_el_remove}*N_slab_el_remove)"""
                mu_i_bulk = self.dic['elements'][el]['ecoh']
                tmp_vect.append(-mu_i_bulk*tmp)
            
                """(\mu_i - \mu_{i}^{bulk})*(N_^{slab}_i - frac{n_i}{n_bulk_el_remove}*N_slab_el_remove)"""
                diff_mu = [val for k,val in enumerate(vect_diff_mu) if el_diff_mu[k] == el][0]
                tmp_vect.append(-diff_mu*tmp)

            """ We have to remove the single energy for each atom """
            N_i = self.find_corresponding_nb('slab',slab,el)
            e_i_single = self.dic['elements'][el]['single']['energy']
            tmp_vect.append(-e_i_single*N_i)


        """gamma computation"""
        convert_eVperA2_to_JperM2 = 16.0219 #here is the conversion from eV/A^2 to J/M^2
        gamma = (E_slab + np.sum(tmp_vect))/(2.0*surface_slab)
        return gamma*convert_eVperA2_to_JperM2

    ##################################################################
    def data_points_hyperplane(self, path_to_write : os.PathLike[str], 
                               slab : str, 
                               bulk :str, 
                               path_slab : os.PathLike[str], 
                               el_remove : str) -> np.ndarray : 
        """Generating hyperplane data for a given slab
        
        Writting all data about the parametric gamma energy and mu bounds
        \mu_i - \mu_i^{bulk} are given for each element and parametric gamma
        surface energy is computed for a given vector : 
        \mu_1 - \mu_1^{bulk}, \mu_2 - \mu_2^{bulk}, ..., \mu_(N-1) - \mu_(N-1)^{bulk}
        
        Parameters 
        ----------
        
        path_to_write : str
            Path to write the all data file about hyperplane

        slab : str
            Name of the slab system

        bulk : str 
            Name of the bulk system associated to the slab

        path_slab : str 
            Path of the slab system poscar for surface calculation       
        
        el_remove : str 
            Element to remove in the parametric energy calculation

        Returns 
        -------

        np.ndarray 
            Normal vector to the hyperplane defining the parametric energy surface

        """
        
        w = open(path_to_write,'a')
        w.write('hyperplane data for slab %s \n'%(slab))
        
        """This file is written to do plot and stability analysis. 
        These analysis can be done with the option : plot_and_stability"""
        w1 = open('%s/hp.data'%(os.path.dirname(path_to_write)),'a')

        """generate all bounding data"""
        bounding = self.build_bounds(slab,bulk,el_remove)
        list_vector_to_generate = []
        w.write('removed el is %s \n'%(el_remove))
        w.write('first im writing bounding for \mu_i - \mu_i^{bulk} \n')
        el_to_do = [el for el in bounding.keys() if el != el_remove and el != 'domain']
        for k,el in enumerate(el_to_do):
                w.write('elements : %s ==> bounds : %3.8f %3.8f \n'%(el,bounding[el][0],bounding[el][1]))
                tmp_vect = np.zeros(len(el_to_do))
                tmp_vect[k] = bounding[el][0]
                list_vector_to_generate.append(tmp_vect)
      
        """Last equation is all mu are equal to 0"""
        list_vector_to_generate.append(np.zeros(len(el_to_do)))
    
        w.write('\n')
        w.write('elements order for the next part is : \n')
        str = ''
        for el in el_to_do : 
            str += ' %s'%(el)
        w.write('%s gamma\n'%(str))

        matrix = None

        """The corresponding vectors are built and written : mu_1, mu_2, ..., mu_N, gamma"""
        for id,vect in enumerate(list_vector_to_generate) : 
            gamma_vect = self.compute_parametric_surface_energy(slab,bulk,path_slab,el_remove,vect,el_to_do)
            str_vect = ''
            for value in vect : 
                str_vect += ' %3.8f'%(value)
            str_vect += ' %3.8f'%(gamma_vect)
            w.write('%s \n'%(str_vect))
            list_vector_to_generate = np.concatenate((vect,-np.array([gamma_vect,-1.0])), axis=0)
            if id == 0 : 
                matrix = np.array([list_vector_to_generate])
            
            else : 
                matrix = np.concatenate((matrix,np.array([list_vector_to_generate])), axis=0)
        """normal vector X verifies Matrix@X = 0 ==> 
        we only need to find the kernel of the endomorphism associated to Matrix"""
        normal_vector = null_space(matrix)
        normal_vector = normal_vector/normal_vector[-2]
        if normal_vector.shape[1] > 1 : 
            print('The manifold defines by parametric gamma energy surface is not a hyperplane')
            print('Dimension of orthogonal supplementary is : %2d'%(normal_vector.shape()[1]))
            normal_vector = None
        else : 
            str_vect_ortho = ''
            for value in normal_vector : 
                str_vect_ortho += ' %3.8f'%(value)
            w.write('Orthogonal vector coordinates are :')
            w.write('%s \n'%(str_vect_ortho))          
            
            str_bounds = ''
            for k,el in enumerate(el_to_do):
                str_bounds += '%s %3.8f %3.8f : '%(el,bounding[el][0],bounding[el][1])
        
            str_domain = ''
            str_el_domain = ''
            for component in self.dic['slab'][slab]['bounds']['domain'][0] :
                str_domain += ' %3.6f'%(component)
            for el in self.dic['slab'][slab]['bounds']['domain'][1] :
                str_el_domain += ' %s'%(el)

            full_domain = '%s :%s'%(str_el_domain,str_domain)
            w1.write('%s : %s gamma offset : %s ==> %s ==> %s \n'%(slab,str,str_vect_ortho,str_bounds[:-2],full_domain))

        w.write('\n')
        w.write('Here is the normal vector for definition domain \n')
        str_domain = ''
        str_el_domain = ''
        for component in self.dic['slab'][slab]['bounds']['domain'][0] :
            str_domain += ' %3.6f'%(component)
        for el in self.dic['slab'][slab]['bounds']['domain'][1] :
            str_el_domain += ' %s'%(el)

        full_domain = '%s :%s'%(str_el_domain,str_domain)
        w.write('%s \n'%(full_domain))

        w.write('\n \n')
        w.close()
        w1.close()

        return normal_vector

def ExtractPathSlab(list_name_slab : List[str], 
                    convergence_file : os.PathLike[str]) -> List[os.PathLike[str]] :
    """"Extracting the name of the slabs thanks to the convergence file
    
    Parameters
    ----------

    list_name_slab : List[str]
        List of slab name to find

    convergence_file : str 
        Path to convergence log file

    Returns 
    -------

    List[str]
        List of corresponding slab path
    
    """ 
    list_path = []
    for name in list_name_slab :
        r = open(convergence_file,'r').readlines()
        for l in r :
            if l.split() != [] :
                if l.split()[-1] == str(1) : 
                    tmp = l.split('==>')[0][:-1]
                    tmp = tmp.split()[1]
                    if tmp.split('/')[-1] == name :
                        if os.path.exists('%s/CONTCAR'%(tmp)) : 
                            contcar_path = '%s/CONTCAR'%(tmp)
                        else : 
                            contcar_path = '%s/POSCAR'%(tmp)
                        list_path.append(contcar_path)
                        break 

    return list_path

def ReadConvergenceFile(object : DataSurface, 
                        convergence_file : os.PathLike[str]) -> DataSurface:
    """Read convergence file and fill the DataSurface object
    
    Parameters 
    ----------
    
    object : DataSurface
        DataSurface object to fill 

    convergence_file : str 
        Path to the convergence log file 

    Returns
    -------

    DataSurface 
        Filled DataSurface object 

    """
    r = open(convergence_file,'r').readlines()
    for l in r :
        if l.split() != [] :
            if l.split()[-1] == str(1) :
                energy_vasp = float(l.split()[-3])
                tmp = l.split('==>')[0][:-1]
                tmp = tmp.split()[1]

                """ single element case ! """
                if tmp.split('/')[-3] == 'elements' :
                    type = tmp.split('/')[-1] 
                    el = tmp.split('/')[-2]
                    nb_atom = GetNatomFromVasp('%s/OUTCAR'%(tmp))
                    object.add_entry_el(el,type,energy_vasp/nb_atom) 

                """ bulk case ! """
                if tmp.split('/')[-2] == 'bulk_full' :
                    path_poscar = '%s/POSCAR'%(tmp)
                    composition = GetElementsFromVasp(path_poscar)
                    name_bulk = tmp.split('/')[-1]
                    object.add_entry_bulk(name_bulk,energy_vasp,composition)                    

                """" slab case ! """
                if tmp.split('/')[-2] == 'slab' :
                    path_poscar = '%s/POSCAR'%(tmp)
                    composition = GetElementsFromVasp(path_poscar)
                    name_slab = tmp.split('/')[-1]
                    object.add_entry_slab(name_slab,energy_vasp,composition)

    return object

def ComputeandWriteParametricSurfaceEnergy(object_surface : DataSurface, 
                                           path_to_write : os.PathLike[str], 
                                           list_name_slab : List[str], 
                                           bulk : str, 
                                           list_path_slab : List[str], 
                                           el_remove : str) -> None : 
    """Fill the data_surface object to calculation parametric gamma energy, the code alse compute
    normal vector to the hyperplane in order to do some analysis and plot
    
    Parameters 
    ----------

    object_surface : DataSurface
        DataSurface object containing all energetic data

    path_to_write : str    
        Path to write hyperplane results file 

    list_name_slab : List[str]
        List of slabs whose the surface will be estimated

    bulk : str 
        Name of the associated bulk system 

    list_path_slab : str
        List of slab paths whose the surface will be estimated

    el_remove : str
        Element to remove during the surface energy estimation
    
    """
    
    """enthalpy and cohesion energy calculation !"""
    object_surface.compute_ecoh_hf(bulk)
    object_surface.compute_ecoh_single()

    """For each slab..."""
    for id_slab ,slab in enumerate(list_name_slab) : 
        """computing hyperplane data"""
        _ = object_surface.data_points_hyperplane(path_to_write,slab,bulk,list_path_slab[id_slab],el_remove)

    return 