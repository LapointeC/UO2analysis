import os, sys
import numpy as np

from tools import RecursiveBuilder
from typing import List, Dict, TypedDict

class DictLauncher(TypedDict) : 
    nproc : int
    walltime : int
    slurm_cmd : str

class DictSetup(TypedDict) : 
    temperature : float 
    nb_therma : int
    ratio_swap : float

class LauncherMC : 
    def __init__(self, path_lmp : os.PathLike[str],
                 path_src : os.PathLike[str],
                 path_work : os.PathLike[str],
                 xml_file : str,
                 dict_launcher : DictLauncher,
                 dict_setup : DictSetup,
                 list_temperature : List[float]) -> None : 
        
        self.path_lmp = path_lmp
        self.path_src = path_src
        self.path_work = path_work

        self.xml_file = xml_file
        self.dict_launcher = dict_launcher
        self.dict_setup = dict_setup

        self.list_temperature = list_temperature

        self._check_path()

    def _check_path(self) -> None : 
        if not os.path.exists(self.path_lmp) or not os.path.exists(self.path_src) :
            raise TimeoutError(f'Input path(s) do not exist : {self.path_lmp} | {self.path_src}')
        else :
            if not os.path.exists(f'{self.path_src}/pot.fs') : 
                raise TimeoutError(f'Potential file does not exist : {self.path_src}/pot.fs')
            else :

                if not os.path.exists(self.path_work) : 
                    os.mkdir(self.path_work)
        return 

    def replace(self,field:str,key:str,value: float | int) -> str:
        """Wrapper around string replace()
           
        Parameters
        ----------
        field : str
            string to be searched
        key : str
            will search for %key%
        value : ScriptArg 
            replacement value
        Returns
        -------
        str
            the string with replaced values
        """
        return field.replace("%"+key+"%",str(value))

    def _write_xml_file(self, path_w : os.PathLike[str]) -> None : 
        xml_file = self.xml_file
        
        #MC setup
        for key, val in self.dict_setup.items() : 
            xml_file = self.replace(xml_file,
                         key,
                         val)    

        #nb proc 
        xml_file = self.replace(xml_file,'nproc',self.dict_launcher['nproc'])

        with open(f'{path_w}/MC.xml', 'w') as w :
            w.write(xml_file)
            w.close()
        return
    
    def _set_inputs(self, path_w : os.PathLike[str],
                    path_config : os.PathLike[str]) -> None :
        os.system(f'cp -r {self.path_src}/MC {path_w}')
        os.system(f'cp -r {self.path_src}/MCManager.py {path_w}')
        os.system(f'cp -r {self.path_src}/pot.fs {path_w}')
        os.system(f'cp {path_config} {path_w}/in.lmp')
        self._write_xml_file(path_w)
        return 
    

    def format_time(self, time : int) -> List[str]:
        tmp_list = []
        list_temps = [3600,60,1]
        for temps in list_temps :
            tmp_list.append(str(100+time//temps)[1:] )
            time = time%temps

        return tmp_list

    def _write_jsub(self, path_w :os.PathLike[str],
                    job_name : str) -> None : 
        
        hms =  self.format_time(self.dict_launcher['walltime']) #[int(el) for el in self.format_time(self.dict_launcher['walltime']) ]
        with open(f'{path_w}/jsub','w') as jsub : 
            jsub.write('#!/bin/bash \n')
            jsub.write('#SBATCH --job-name=%s     # job name \n'%(job_name))
            jsub.write('#SBATCH --ntasks=%s                 # number of MP tasks \n'%(str(self.dict_launcher['nproc'])))
            jsub.write('#SBATCH --ntasks-per-node=40          # number of MPI tasks per node \n')
            jsub.write('#SBATCH --hint=nomultithread         # we get physical cores not logical \n')
            jsub.write('#SBATCH --time=%s:%s:%s              # maximum execution time (HH:MM:SS) \n'%(hms[0],hms[1],hms[2]))
            jsub.write('#SBATCH --output=%s.out # output file name \n'%(job_name))
            jsub.write('#SBATCH --error=%s.err  # error file name \n'%(job_name))
            jsub.write('\n')
            jsub.write('suff=%s \n'%(job_name))
            jsub.write('module purge \n')
            jsub.write('module load python/3.10.4 \n')
            jsub.write('module load intel-compilers/19.0.4 \n')
            jsub.write('module load intel-mkl/2019.4 \n')
            jsub.write('module load intel-mpi/2019.4 \n')
            jsub.write('module load cmake/3.21.3 \n')
            jsub.write('conda activate python_pafi \n')
            jsub.write('\n')
            jsub.write('export OMP_NUM_THREADS=1 \n')
            jsub.write('\n')
            jsub.write('SUBMISSION=${SLURM_SUBMIT_DIR} \n')
            jsub.write('outputfile=${SLURM_SUBMIT_DIR}/${suff}.out \n')
            jsub.write('\n')
            jsub.write('ulimit -c unlimited \n')
            jsub.write('srun  --mpi=pmi2 -K1 --resv-ports -n ${SLURM_NTASKS} python MCManager.py -p MC.xml > out_MC')
            jsub.close()
        return 


    def _launch_calculations(self, temperature : float) -> None : 
        path_w_T = f'{self.path_work}/{temperature}'
        if not os.path.exists(path_w_T) : 
            os.mkdir(path_w_T)
        
        path_configs = [f'{self.path_lmp}/{f}' for f in os.listdir(self.path_lmp)]
        for conf in path_configs : 
            name_conf = f"{os.path.basename(conf).split('.')[0]}_{int(temperature)}K"
            path_w_T_conf = f"{path_w_T}/{os.path.basename(conf).split('.')[0]}"

            if not os.path.exists(path_w_T_conf) :
                os.mkdir(path_w_T_conf)

            self._set_inputs(path_w_T_conf,
                             conf)
            self._write_jsub(path_w_T_conf,
                             name_conf)
            self._write_xml_file(path_w_T_conf)

            os.chdir(path_w_T_conf)
            #print(f'{self.dict_launcher["slurm_cmd"]}')
            os.system(f'{self.dict_launcher["slurm_cmd"]}')

        return 
    
    def LaunchCalculationsTemperature(self) -> None : 
        for temp in self.list_temperature : 
            self.dict_setup['temperature'] = temp
            self._launch_calculations(temp)

        return
    
    def _iterate_writing(self, path_w : os.PathLike[str]) -> None : 
        if not os.path.exists(f'{path_w}/it.data') : 
            with open(f'{path_w}/it.data', 'w') as w :
                w.write('1')

        else : 
            with open(f'{path_w}/it.data', 'r') as r : 
                it = int(r.readlines()[0])

            # update
            with open(f'{path_w}/it.data', 'a') as w :
                w.write(f'{int(it+1)}')

            # update restart file 
            restart_file = max([ f"{path_w}/write_data/{f}" for f in os.listdir(f'{path_w}/write_data') ],
                key = os.path.getatime)   
            os.system(f'cp {restart_file} {path_w}/in.lmp')
            
            # update config files
            os.system(f'mv {path_w}/write_data {path_w}/write_data_{it+1}')

        return 

    def RelaunchCalculations(self) -> None :
        list_path_calc = RecursiveBuilder(self.path_work, file2find='out_MC')
        for path_calc in list_path_calc : 
            self._iterate_writing(path_calc)
            os.system(self.dict_launcher['slurm_cmd'])

        return


######################
### XML 
######################
xml_file = """
<MC>
        <Parameters>
                <CoresPerWorker>%nproc%</CoresPerWorker>
                <Mode>VC-SGC-MC</Mode>

                <LogLammps>False</LogLammps>
                <Verbose>0</Verbose>
                <ThermalisationSteps>%nb_therma%</ThermalisationSteps>
                <Temperature>%temperature%</Temperature>
                <MuArray>0.0 0.3709309654827413 -1.2629814907453727 1.602009004502251 -2.1423211605802903</MuArray>
                <ConcentrationArray>0.2 0.2 0.2 0.2 0.2</ConcentrationArray>
                <Kappa>0.5</Kappa>

                <FractionSwap>%ratio_swap%</FractionSwap>
                <DeltaTau>0.0025</DeltaTau>
                <DampingT>0.5</DampingT>
                <DampingP>6.25</DampingP>

                <FrequencyMC>100</FrequencyMC>
                <NumberNPTSteps>50000</NumberNPTSteps>
        </Parameters>

        <Pathways> 
                <Configuration>./in.lmp</Configuration>
                <Potential>./pot.fs</Potential>
                <WritingDirectory>./write_data</WritingDirectory>
        </Pathways>

        <Scripts>
                <Input>
                        boundary        p p p
                        units           metal
                        atom_style      atomic
                        atom_modify map array sort 0 0.0

                        read_data       %Configuration%

                        pair_style      milady
                        pair_coeff      * *  pot.fs Mo Nb Ta V W
                        timestep        0.0025
                        group            Mo type 1
                        group            Nb type 2
                        group            Ta type 3
                        group            V type 4
                        group            W type 5

                </Input>
        </Scripts>

</MC>
"""
##############################

##############################
### INPUTS
##############################
path_lmp = '/home/lapointe/ToAxel/VC-ML/RandomConfigs'
path_src = '/home/lapointe/ToAxel/SGCMC/Src'
path_work = '/home/lapointe/ToAxel/VC-ML/test_launch'
list_temp = [400.0 ,1500.0]

dict_launch : DictLauncher = {'nproc':320,
                              'walltime':68400,
                              'slurm_cmd':'sbatch --account=aih@cpu jsub'}
dict_setup : DictSetup = {'nb_therma':150,
                          'ratio_swap':0.30,
                          'temperature':None}
#####################


obj_launch = LauncherMC(path_lmp,
                        path_src,
                        path_work,
                        xml_file,
                        dict_launch,
                        dict_setup,
                        list_temp)

obj_launch.LaunchCalculationsTemperature()