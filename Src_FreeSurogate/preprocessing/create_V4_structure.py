import numpy as np
import os, sys, shutil
import random

from ase import Atoms
from ase.io import write
from typing import Dict, List
from tools import RecursiveBuilder

class Generator : 
    def __init__(self, path_gin : os.PathLike[str],
                 path_free : os.PathLike[str]) -> None : 
        self.path_gin = path_gin
        self.path_free = path_free

        if os.path.exists(self.path_free) : 
            shutil.rmtree(self.path_free)
        os.mkdir(self.path_free)

    def read_gin(self, path : os.PathLike[str],
                 dic_species : Dict[int,str]) -> Atoms :
        pos = None
        list_species = []
        cell = np.zeros((3,3))
        compt_pos = 0

        with open(path, 'r') as reader :
            lines = reader.readlines()
            for id_l, l in enumerate(lines) :
                if id_l == 1 :
                    cell[0,:] = np.array([ float(el) for el in l.split() ])
                if id_l == 2 :
                    cell[1,:] = np.array([ float(el) for el in l.split() ])
                if id_l == 3 :
                    cell[2,:] = np.array([ float(el) for el in l.split() ])
                if id_l == 4 :
                    pos = np.zeros(( int(l) , 3 ))
                if id_l >= 5 :
                    pos_l = np.array([ float(el) for el in l.split()[:-1] ])
                    species = dic_species[int(l.split()[-1])]

                    pos[compt_pos,:] = pos_l@cell
                    list_species.append(species)
                    compt_pos += 1

        return Atoms(symbols=list_species,
                     positions=pos,
                     cell=cell,
                     pbc=(True,True,True))


    def WriteConvFile(self, path : os.PathLike[str]) -> None : 
        with open(f'{path}/conv.file', 'w') as w : 
            w.write('yes')
            w.close()
        return 

    def WriteFakeEigenValues(self, path : os.PathLike[str]) -> None :
        with open(f'{path}/eigenvalues.dat', 'w') as w : 
            for _ in range(3) :
                w.write('0.0 \n')
            w.write('5.8 \n')
            w.close()
        return

    def GenerateAllSturcture(self, 
                             dic_species : Dict[int,str]) -> None : 
        list_path_gin = [f'{self.path_gin}/{f}' for f in os.listdir(self.path_gin)]
        for path_gin in list_path_gin : 
            name = os.path.basename(path_gin).split('.')[0].split('_')[-1]

            os.mkdir(f'{self.path_free}/{name}')
            atoms_gin = self.read_gin(path_gin,dic_species)
            os.system(f'cp {path_gin} {self.path_free}/{name}/loop.gin')
            write(f'{self.path_free}/{name}/in.lmp',
                  atoms_gin,
                  format='lammps-data')
            self.WriteConvFile(f'{self.path_free}/{name}')
            self.WriteFakeEigenValues(f'{self.path_free}/{name}')
            
        return 
    


    def ExcludeKey(self , path_vac : os.PathLike[str]) -> List[str] :
        list_path_vac = RecursiveBuilder(path_vac, file2find='conv.file')
        return [os.path.basename(path) for path in list_path_vac]

    def BuildSubset(self,
                    path_subsets : os.PathLike[str], 
                    nb_dir : int,
                    calc_per_dir : int,
                    key2exclude : List[str] = []) -> None :
        
        if not os.path.exists(path_subsets) : 
            os.mkdir(path_subsets)

        list_dir = [f'{self.path_free}/{d}' for d in os.listdir(self.path_free) if d not in key2exclude]
        random.shuffle(list_dir)
        for k in range(nb_dir) : 
            path_work = f'{path_subsets}/dir_vac_{k+1}'
            list_subset = list_dir[k*calc_per_dir:(k+1)*calc_per_dir]
            if len(list_subset) == 0 :
                break 

            else : 
                os.mkdir(path_work)
                for path in list_subset : 
                    os.system(f'cp -r {path} {path_work}')


###########################
### INPUTS
###########################
path_gin = '/home/lapointe/WorkML/FreeEnergySurrogate/data/V4'
path_free_vac = '/home/lapointe/WorkML/FreeEnergySurrogate/data/V4_free_dir'
path_subsets2exclude = '/home/lapointe/WorkML/FreeEnergySurrogate/data/V4_free_dir_subset'
path_subsets = '/home/lapointe/WorkML/FreeEnergySurrogate/data/V4_free_dir_good_sub'
dic_equiv = {1:'Fe'}
###########################

generator_obj = Generator(path_gin,
                          path_free_vac)
key_to_exclude = generator_obj.ExcludeKey(path_subsets2exclude)
generator_obj.GenerateAllSturcture(dic_equiv)
generator_obj.BuildSubset(path_subsets,10,40,
                          key2exclude=key_to_exclude)