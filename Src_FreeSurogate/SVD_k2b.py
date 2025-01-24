import os, sys 
import numpy as np
import pickle
import scipy

from ase import Atoms
from typing import Dict, Tuple, List, TypedDict

import scipy.linalg

class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    atoms : Atoms

class SVDk2b_Descriptor : 
    def __init__(self, path_dir_k2b_pkl : os.PathLike[str],
                 path_bso4_pkl : os.PathLike[str],
                 path_writing : os.PathLike[str]) -> None : 
        self.path_dir_k2b_pkl = path_dir_k2b_pkl
        self.path_bso4_pkl = path_bso4_pkl
        self.path_writing = path_writing

    def load_pickle(self, path_pickle : os.PathLike[str]) -> Dict[str, Data] : 
        return pickle.load(open(path_pickle, 'rb'))

    def write_pickle(self, data : Dict[str, Atoms]) -> None :
        pickle.dump(data, open(f'{self.path_writing}/svd_desc_mab.pickle','wb'))
        return

    def GetDescritporConfig(self, data_config : Dict[str,Data],
                            key : str,
                            dimtoavoid : int = 2) -> np.ndarray : 
        if dimtoavoid == 0 : 
            return data_config[key]['atoms'].get_array('milady-descriptors')
        else :
            return data_config[key]['atoms'].get_array('milady-descriptors')[:,:-dimtoavoid-1]

    def LoadCollectionk2b_simple(self, key : str) -> np.ndarray : 
        # descriptor tensor shape is NxDxNb_k2b
        descriptor_tensor = None 

        paths_picklek2b = [f'{self.path_dir_k2b_pkl}/{f}' for f in os.listdir(self.path_dir_k2b_pkl)]
        nb_pickek2b = len(paths_picklek2b)
        for id_p, path_pickle in enumerate(paths_picklek2b) :  
            data_pickle_idp = self.load_pickle(path_pickle) 
            desc_p_c = self.GetDescritporConfig(data_pickle_idp, key)
            if id_p == 0 :
                descriptor_tensor = np.zeros((desc_p_c.shape[0],desc_p_c.shape[1],nb_pickek2b))
            
            descriptor_tensor[:,:,id_p] = desc_p_c

        return descriptor_tensor

    def BuildSVDK2b_simple(self, tensor_desc : np.ndarray) -> np.ndarray : 
        svd_desc = np.zeros((tensor_desc.shape[0],min(tensor_desc.shape[1],tensor_desc.shape[2])))
        for id_at in range(tensor_desc.shape[0]) : 
            _, svd_desc_config, _ = np.linalg.svd(tensor_desc[id_at,:,:], full_matrices=True)
            svd_desc[id_at,:] = svd_desc_config
        return svd_desc

    def LoadCollectionk2b(self) -> np.ndarray : 
        # descriptor tensor shape is MxDxNb_k2b
        descriptor_tensor = None 

        paths_picklek2b = [f'{self.path_dir_k2b_pkl}/{f}' for f in os.listdir(self.path_dir_k2b_pkl)]
        nb_pickek2b = len(paths_picklek2b)
        for id_p, path_pickle in enumerate(paths_picklek2b) :
            data_pickle_idp = self.load_pickle(path_pickle)
            nb_config = len(data_pickle_idp)
            for id_c, conf in enumerate(data_pickle_idp.keys()) : 
                desc_p_c = self.GetDescritporConfig(data_pickle_idp, conf)
                if id_c == 0 and id_p == 0: 
                    size_desc = desc_p_c.shape[1]
                    nb_atoms = desc_p_c.shape[0]
                    descriptor_tensor = np.zeros((nb_config, nb_atoms, size_desc, nb_pickek2b))

                descriptor_tensor[id_c,:,:,id_p] = desc_p_c
        
        return descriptor_tensor
    
    def LoadManyBodyDescriptor(self) -> Tuple[np.ndarray, Dict[str,Data]] : 
        data_pkl_manyb = self.load_pickle(self.path_bso4_pkl)
        desc_many = None
        for id_c, conf in enumerate(data_pkl_manyb) : 
            desc_mb_c = self.GetDescritporConfig(data_pkl_manyb, conf, dimtoavoid=0)
            if id_c == 0 : 
                desc_many = np.zeros((len(data_pkl_manyb),desc_mb_c.shape[0],desc_mb_c.shape[1]))
            
            desc_many[id_c,:,:] = desc_mb_c
        return desc_many, data_pkl_manyb.copy()

    def BuildSVDK2b(self, tensor_desc : np.ndarray) -> np.ndarray : 
        svd_desc = np.zeros((tensor_desc.shape[0],tensor_desc.shape[1],min(tensor_desc.shape[2],tensor_desc.shape[3])))
        for id_config in range(tensor_desc.shape[0]) : 
            for id_at in range(tensor_desc.shape[1]) : 

                _, svd_desc_config, _ = np.linalg.svd(tensor_desc[id_config,id_at,:,:], full_matrices=True)
                svd_desc[id_config,id_at,:] = svd_desc_config

        return svd_desc
    
    def BuildSVDDescriptor(self) -> None :
        tensor_descriptor = self.LoadCollectionk2b()
        print(f'... tensor descriptor is built : {tensor_descriptor.shape} ...')
        new_array_descriptor = self.BuildSVDK2b(tensor_descriptor)
        print(f'... SVD descriptor is built : {new_array_descriptor.shape} ...')
        array_descriptor_many, old_pkl_data = self.LoadManyBodyDescriptor()
        print(f'... MB descriptor is built : {array_descriptor_many.shape} ...')
        full_descriptor = np.concatenate((new_array_descriptor,array_descriptor_many), axis=1)
        print(f'... full descriptor is built : {full_descriptor.shape} ...')
        new_pkl_data = old_pkl_data.copy()        
        for id_conf, conf in enumerate(old_pkl_data) : 
            old_desc = old_pkl_data[conf]['atoms'].get_array('milady-descriptors')
            new_desc = np.zeros((old_desc.shape[0], full_descriptor.shape[1]))
            
            # broadcast
            new_desc[:,:] = full_descriptor[id_conf,:]/float(old_desc.shape[0])
            # 2 steps for new array !
            new_pkl_data[conf]['atoms'].set_array('milady-descriptors', None)
            new_pkl_data[conf]['atoms'].set_array('milady-descriptors', new_desc, dtype=float)

        self.write_pickle(new_pkl_data)
        return
    

    def BuildSmartSVDDesc(self) -> None : 
        pkl_data_mb = self.load_pickle(self.path_bso4_pkl)

        new_pkl = pkl_data_mb.copy()
        for k,conf in enumerate(pkl_data_mb.keys()) :
            print(f'Conf : {conf}  {k+1}/{len(pkl_data_mb)}') 
            desc_mb = self.GetDescritporConfig(pkl_data_mb, conf,dimtoavoid=0)
            print(f'... MB descriptor is built : {desc_mb.shape} ...')
            tensor_desc_k2b = self.LoadCollectionk2b_simple(conf)
            print(f'... tensor descriptor is built : {tensor_desc_k2b.shape} ...')
            desc_svd = self.BuildSVDK2b_simple(tensor_desc_k2b)
            print(f'... SVD descriptor is built : {desc_svd.shape} ...')
            full_desc = np.concatenate((desc_mb,desc_svd) , axis = 1)
            print()

            new_pkl[conf]['atoms'].set_array('milady-descriptors', None)
            new_pkl[conf]['atoms'].set_array('milady-descriptors', full_desc, dtype=float)
        
        self.write_pickle(new_pkl)
        return

##########################
### INPUTS
##########################
path_pkl_k2b = '/home/lapointe/WorkML/FreeEnergySurrogate/data/full_k2b'
path_pkl_many_body = '/home/lapointe/WorkML/FreeEnergySurrogate/data/mab_desc_j4_r6.pickle'
path_writing = '/home/lapointe/WorkML/FreeEnergySurrogate/data'
##########################

obj_svd = SVDk2b_Descriptor(path_pkl_k2b,
                            path_pkl_many_body,
                            path_writing)
obj_svd.BuildSmartSVDDesc()