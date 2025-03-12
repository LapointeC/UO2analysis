import numpy as np
import os
import pickle

from ase.io import read
from typing import Dict, List, TypedDict

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

class DataStorage : 
    def __init__(self) : 
        self.data : dic_type = {'elements':{},'bulk':{},'slab':{}}

    def compute_surface_slab(self, path_slab : os.PathLike[str]) -> float:
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


        slab_obj= read(path_slab, format='vasp')         
        cell = slab_obj.cell[:]
        #we assumed that the slab is oriented following the 001 direction
        #so we keep the restriction of the cell matrix and we compute the determinant to have the area
        cell_restricted = cell[:2,:2]
        return np.linalg.det(cell_restricted)

    def update_el(self, el : str,
                  type : str,
                  energy : float) -> None :
        """Update element entry for pickle file 
        
        Parameters
        ----------

        el : str
            Element to update

        type : str
            single or bulk to compute cohesion energy

        energy : float 
            Associated per atom energy
        """
        if el not in self.data['elements'].keys() : 
            self.data['elements'][el] = {'single':{'energy':None},'bulk':{'energy':None},'ecoh':None}
        self.data['elements'][el][type]['energy'] = energy     
        return 
    
    def update_bulk(self, name_bulk : str,
                    energy : float,
                    list_el : List[str],
                    nb_el : List[int]) -> None :
        """Update bulk entry for pickle file 
        
        Parameters
        ----------
        
        name_bulk : str
            Bulk to update

        energy : float
            Energy associated to the bulk

        list_el : List[str]
            List of elements in bulk [el1, ..., eln]

        nb_el : List[int]
            Associated number of element [nb1, ..., nbn]
        """       
        self.data['bulk'][name_bulk] = {'energy':energy,
                                        'composition':[list_el,nb_el],
                                        'hf':None,
                                        'ecoh':None} 
        return 
    
    def update_slab(self, name_slab : str,
                    energy_slab : float,
                    list_el : List[str],
                    nb_el : List[int],
                    surface : float) -> None :
        """Update slab entry for pickle file 
        
        Parameters
        ----------
        
        name_slab : str
            Slab to update

        energy_slab : float
            Energy associated to the slab

        list_el : List[str]
            List of elements in slab [el1, ..., eln]

        nb_el : List[int]
            Associated number of element [nb1, ..., nbn]

        surface : float
            Surface of the slab in AA²
        """    
        self.data['slab'][name_slab] = {'energy':energy_slab,
                                        'composition':[list_el,nb_el],
                                        'surface':surface,
                                        'bounds':None} 
        return 

    def fill_data(self, path_data : os.PathLike[str],
                  list_el : List[str],
                  nb_el : List[int],
                  energy : float) -> None :
        """Update the internal dictionnary with new data
        
        Parameters
        ----------

        path_data : os.PathLike[str] 
            Path to the OUTCAR associated to a given calculation

        list_el : List[str]
            List of elements in the calculated system [el1, ..., eln] 

        nb_el : List[int]
            Associated number of element [nb1, ..., nbn]

        energy : float 
            Energy associated to the system
        """
        split_path = path_data.split('/')
        data_type = split_path[-3]
        data_type_el = split_path[-4]
        # slab case 
        if data_type == 'slab' : 
            path_poscar = f'{os.path.basename(path_data)}/CONTCAR'
            surface = self.compute_surface_slab(path_poscar)
            name_slab = split_path[-2]
            
            self.update_slab(name_slab,
                             energy,
                             list_el,
                             nb_el,
                             surface)
        # bulk case 
        if data_type == 'bulk' : 
            name_bulk = split_path[-2]

            self.update_bulk(name_bulk,
                             energy,
                             list_el,
                             nb_el)
            
        if data_type_el == 'elements' :
            name_element = split_path[-3]
            type = split_path[-2]
            
            self.update_el(name_element,
                           type,
                           energy/np.sum(nb_el))
        return 

    def write_pkl(self, path_writing : os.PathLike[str]) -> None : 
        """Write the pickle fill containing all data from calculations
        
        Parameters
        ----------

        path_writing : os.PathLike[str]
            Path to the pickle file to write 
        """
        pickle.dump(self.data, open(path_writing,'wb'))
        return 