import os, sys 
import numpy as np
import pickle
import matplotlib as mpl

from ase import Atoms
from typing import Dict, Tuple, List, TypedDict
import seaborn as sns

import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import CubicSpline
from matplotlib.colors import to_rgba

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts}'})
ext = 'pdf'

class Data(TypedDict) : 
    array_temperature : np.ndarray
    array_ref_FE : np.ndarray 
    array_anah_FE : np.ndarray
    array_full_FE : np.ndarray
    array_sigma_FE : np.ndarray
    array_delta_FE : np.ndarray
    atoms : Atoms
    stress : np.ndarray
    volume : float
    energy : float
    Ff : np.ndarray

class FData(TypedDict) : 
    T : np.ndarray
    mean_F : np.ndarray
    std_F : np.ndarray

class Analyse : 
    def __init__(self, path_pkl : os.PathLike[str]) : 
        self.path_pkl = path_pkl
        self.all_data = self.load_data()

        if not self.check_bulk_integrity(self.all_data['bulk']) : 
            raise ValueError(f'Problem with bulk data {self.all_data["bulk"]["array_full_FE"]}...')

    def mixing_color(self,color1,
                     color2,
                     mixing :float) -> tuple :
        return tuple(mixing*np.asarray(color1)+(1.0-mixing)*np.asarray(color2))

    def load_data(self) -> Dict[str,Data] : 
        return pickle.load(open(self.path_pkl, 'rb'))        

    def check_bulk_integrity(self, data_bulk : Data) -> bool :
        if np.isnan(data_bulk['array_full_FE']).any() : 
            return False, 
        else : 
            return True

    def compute_formation_free_energy(self, data_bulk : Data,
                                      data_dfct : Data) -> np.ndarray : 
        Nbulk = float(len(data_bulk['atoms']))
        Ndfct = float(len(data_dfct['atoms']))
        return (data_dfct['array_full_FE'] + data_dfct['energy']) - Ndfct*(data_bulk['array_full_FE'] + data_bulk['energy'])/Nbulk
    
    def compute_formation_delta_anha_free_energy(self, data_bulk : Data,
                                      data_dfct : Data) -> np.ndarray : 
        try : 
            Nbulk = float(len(data_bulk['atoms']))
            Ndfct = float(len(data_dfct['atoms']))
            return (data_dfct['array_full_FE']) - Ndfct*(data_bulk['array_full_FE'])/Nbulk
        except : 
            return (data_dfct['array_full_FE']) - (1028.0)*(data_bulk['array_full_FE'])/(1024.0)

    def ComputeAllFormationFreeEnergy(self) -> None : 
        for key_dfct in self.all_data.keys() :
            array_Ff = self.compute_formation_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
            self.all_data[key_dfct]['Ff'] = array_Ff
            print(key_dfct, array_Ff)
        return 
    
    def Analyse_Delta_anah_FreeEnergy(self, list_T : List[float], name : str, vac_analysis : bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] : 
        if not vac_analysis : 
            full_data = np.zeros((len(self.all_data),len(list_T)))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                #array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                try : 
                    array_delta_Ff = self.compute_formation_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                except : 
                    array_delta_Ff = self.all_data[key_dfct]['Ff']
                full_data[id,:] = array_delta_Ff

            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            color_array = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            
            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)

            # data F
            mean_F = np.zeros(len(list_T))
            std_F = np.zeros(len(list_T))

            for id_T,T in enumerate(list_T) : 
                data_T = full_data[:-1,id_T]
                mask_nan = np.isnan(data_T)

                #mask2 = data_T < 10.0 
                mask2 = data_T > 0.0
                mask = mask2 & ~mask_nan
                #mask = ~mask_nan

                mean_delta_Ff_T = np.mean(data_T[mask])
                std_delta_Ff_T = np.std(data_T[mask])

                mean_F[id_T] = mean_delta_Ff_T
                std_F[id_T] = std_delta_Ff_T

                print(f'Temperature = {T} K,mean Ff = {mean_delta_Ff_T} eV, sigma Ff_anah {std_delta_Ff_T} eV')

                nb_bin = int(len(self.all_data)/20.0)
                #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                sns.histplot(x=data_T[mask], 
                             color=color_array[id_T], 
                             label=f'Temperature : {T} K', 
                             bins=nb_bin,
                             stat='density',
                             kde=True,
                             alpha=0.6)

            #fig = plt.gcf()  # Get the current figure
            #cbar1 = fig.colorbar(sm1, ax=plt.gca(), label='Temperature (K) for all defects', orientation='vertical',shrink=0.8)  
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for all defects', orientation='horizontal',shrink=0.1)
            cbar1.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)

        else : 
            full_data = np.zeros((len(self.all_data),len(list_T),2))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                #array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                try : 
                    array_delta_Ff = self.compute_formation_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                except : 
                    array_delta_Ff = self.all_data[key_dfct]['Ff']
                if key_dfct[0] == '0' :
                    full_data[id,:,0] = array_delta_Ff 
                    full_data[id,:,1] = np.nan
                else : 
                    full_data[id,:,1] = array_delta_Ff 
                    full_data[id,:,0] = np.nan                      


            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 

            Cmap1 = plt.get_cmap('viridis')
            Cmap2 = plt.get_cmap('magma')
            color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            color_array2 = [Cmap2(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)
            sm2 = ScalarMappable(cmap=Cmap2, norm=norm)

            # data F
            mean_F = np.zeros((len(list_T),2))
            std_F = np.zeros((len(list_T),2))

            for id_T,T in enumerate(list_T) : 
                for dfct in range(full_data.shape[2]) :
                    data_T = full_data[:,id_T,dfct]
                    mask_nan = np.isnan(data_T)

                    #mask2 = data_T < 10.0 
                    mask2 = data_T > 0.0
                    mask = mask2 & ~mask_nan
                    #mask = ~mask_nan

                    mean_delta_Ff_T = np.mean(data_T[mask])
                    std_delta_Ff_T = np.std(data_T[mask])              

                    dic_equiv = {1:r'$I_4$',0:r'$V_4$'}
                    dic_equivcolor = {1: color_array1,
                                      0: color_array2}
                    
                    mean_F[id_T,dfct] = mean_delta_Ff_T
                    std_F[id_T,dfct] = std_delta_Ff_T

                    print(f'{dic_equiv[dfct]} ==> Temperature = {T} K,mean Ff = {mean_delta_Ff_T} eV, sigma Ff_anah {std_delta_Ff_T} eV')

                    nb_bin = int(len(self.all_data)/20.0)
                    #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                    sns.histplot(x=data_T[mask], 
                                 color=dic_equivcolor[dfct][id_T], 
                                 label=r'%s ,Temperature : %2.1f K'%(dic_equiv[dfct],T), 
                                 bins=nb_bin,
                                 stat='density',
                                 kde=True,
                                 alpha=0.6)

            # Add colorbars for each defect type
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            axins2 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper left')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for $I_4$' ,orientation='horizontal',shrink=0.1)      
            cbar2 = fig.colorbar(sm2, cax=axins2, label='Temperature (K) for $V_4$' ,orientation='horizontal',shrink=0.1) #,fraction=0.1, pad=0.02

            cbar1.ax.tick_params(labelsize=14)
            cbar2.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)
            cbar2.set_label('Temperature (K) for $V_4$', fontsize=16)

        majorFormatter = mpl.ticker.FormatStrFormatter('%1.1f')
        majorLocator = mpl.ticker.MultipleLocator(0.5)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        #ax.yaxis.set_major_locator(majorLocator)
        #ax.yaxis.set_major_formatter(majorFormatter)
        #ax.yaxis.set_minor_locator(minorLocator)
        ax.tick_params(which='both', width=1,labelsize=18)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')

        ax.set_xlabel(r'$F_\textrm{f}(T)$ [%s] in eV'%(name), size=20)
        ax.set_ylabel(r'Density in A.U',size=20)
        #plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'Ff_anah_hist_{name}.{ext}',dpi=300)
    
        return np.asarray(list_T), mean_F, std_F

    def Analyse_anah_FreeEnergy(self, list_T : List[float], name : str, vac_analysis : bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] : 
        if not vac_analysis : 
            full_data = np.zeros((len(self.all_data),len(list_T)))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                #try : 
                #    array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                #except : 
                #    array_delta_Ff = self.all_data[key_dfct]['Ff']
                full_data[id,:] = array_delta_Ff

            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            color_array = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            
            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)

            # data F
            mean_F = np.zeros(len(list_T))
            std_F = np.zeros(len(list_T))

            for id_T,T in enumerate(list_T) : 
                data_T = full_data[:-1,id_T]
                mask_nan = np.isnan(data_T)

                #mask2 = data_T < 10.0 
                #mask2 = data_T > 0.0
                mask2 = np.abs(data_T) < 10.0
                mask = mask2 & ~mask_nan
                #mask = ~mask_nan

                mean_delta_Ff_T = np.mean(data_T[mask])
                std_delta_Ff_T = np.std(data_T[mask])

                mean_F[id_T] = mean_delta_Ff_T
                std_F[id_T] = std_delta_Ff_T

                print(f'Temperature = {T} K,mean Ff = {mean_delta_Ff_T} eV, sigma Ff_anah {std_delta_Ff_T} eV')

                nb_bin = int(len(self.all_data)/20.0)
                #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                sns.histplot(x=data_T[mask], 
                             color=color_array[id_T], 
                             label=f'Temperature : {T} K', 
                             bins=nb_bin,
                             stat='density',
                             kde=True,
                             alpha=0.6)

            #fig = plt.gcf()  # Get the current figure
            #cbar1 = fig.colorbar(sm1, ax=plt.gca(), label='Temperature (K) for all defects', orientation='vertical',shrink=0.8)  
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for all defects', orientation='horizontal',shrink=0.1)
            cbar1.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)

        else : 
            full_data = np.zeros((len(self.all_data),len(list_T),2))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                #try : 
                #    array_delta_Ff = self.compute_formation_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                #except : 
                #    array_delta_Ff = self.all_data[key_dfct]['Ff']
                if key_dfct[0] == '0' :
                    full_data[id,:,0] = array_delta_Ff 
                    full_data[id,:,1] = np.nan
                else : 
                    full_data[id,:,1] = array_delta_Ff 
                    full_data[id,:,0] = np.nan                      


            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 

            Cmap1 = plt.get_cmap('viridis')
            Cmap2 = plt.get_cmap('magma')
            color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            color_array2 = [Cmap2(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)
            sm2 = ScalarMappable(cmap=Cmap2, norm=norm)

            # data F
            mean_F = np.zeros((len(list_T),2))
            std_F = np.zeros((len(list_T),2))

            for id_T,T in enumerate(list_T) : 
                for dfct in range(full_data.shape[2]) :
                    data_T = full_data[:,id_T,dfct]
                    mask_nan = np.isnan(data_T)

                    #mask2 = data_T < 10.0 
                    mask2 = np.abs(data_T) < 20.0
                    mask = mask2 & ~mask_nan
                    #mask = ~mask_nan

                    mean_delta_Ff_T = np.mean(data_T[mask])
                    std_delta_Ff_T = np.std(data_T[mask])              

                    dic_equiv = {1:r'$I_4$',0:r'$V_4$'}
                    dic_equivcolor = {1: color_array1,
                                      0: color_array2}
                    
                    mean_F[id_T,dfct] = mean_delta_Ff_T
                    std_F[id_T,dfct] = std_delta_Ff_T

                    print(f'{dic_equiv[dfct]} ==> Temperature = {T} K,mean Ff = {mean_delta_Ff_T} eV, sigma Ff_anah {std_delta_Ff_T} eV')

                    nb_bin = int(len(self.all_data)/20.0)
                    #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                    sns.histplot(x=data_T[mask], 
                                 color=dic_equivcolor[dfct][id_T], 
                                 label=r'%s ,Temperature : %2.1f K'%(dic_equiv[dfct],T), 
                                 bins=nb_bin,
                                 stat='density',
                                 kde=True,
                                 alpha=0.6)

            # Add colorbars for each defect type
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            axins2 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper left')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for $I_4$' ,orientation='horizontal',shrink=0.1)      
            cbar2 = fig.colorbar(sm2, cax=axins2, label='Temperature (K) for $V_4$' ,orientation='horizontal',shrink=0.1) #,fraction=0.1, pad=0.02

            cbar1.ax.tick_params(labelsize=14)
            cbar2.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)
            cbar2.set_label('Temperature (K) for $V_4$', fontsize=16)

        majorFormatter = mpl.ticker.FormatStrFormatter('%1.1f')
        majorLocator = mpl.ticker.MultipleLocator(0.5)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)
        #ax.yaxis.set_major_locator(majorLocator)
        #ax.yaxis.set_major_formatter(majorFormatter)
        #ax.yaxis.set_minor_locator(minorLocator)
        ax.tick_params(which='both', width=1,labelsize=18)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')

        ax.set_xlabel(r'$F_\textrm{f}(T)$ [%s] in eV'%(name), size=20)
        ax.set_ylabel(r'Density in A.U',size=20)
        #plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'Delta_F_anha_{name}.{ext}',dpi=300)
    
        return np.asarray(list_T), mean_F, std_F

    def Analyse_Fblock_FreeEnergy(self, list_T : List[float], name : str, vac_analysis : bool = False) -> None : 
        if not vac_analysis : 
            full_data = np.zeros((len(self.all_data),len(list_T)))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                try : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_delta_FE'])/np.abs(self.all_data[key_dfct]['array_full_FE'])
                except : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_block'])/np.abs(self.all_data[key_dfct]['array_full_FE'])
            
            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            color_array = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)

            for id_T,T in enumerate(list_T) : 
                data_T = full_data[:-1,id_T]
                mask_nan = np.isnan(data_T)

                #mask2 = data_T < 10.0 
                mask2 = data_T > 0.0
                mask = mask2 & ~mask_nan
                #mask = ~mask_nan

                mean_Fblock_T = np.mean(data_T[mask])
                std_Fblock_T = np.std(data_T[mask])
                print(f'Temperature = {T} K,mean Ff = {mean_Fblock_T} eV, sigma Ff_anah {std_Fblock_T} eV')

                nb_bin = int(len(self.all_data)/20.0)
                #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                sns.histplot(x=data_T[mask], 
                             color=color_array[id_T], 
                             label=f'Temperature : {T} K', 
                             bins=nb_bin,
                             stat='density',
                             kde=True,
                             alpha=0.6,
                             log_scale=True)
                
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for all defects', orientation='horizontal',shrink=0.1)
            cbar1.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)

        else : 
            full_data = np.zeros((len(self.all_data),len(list_T),2))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                #array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                try : 
                    data_block = np.abs(self.all_data[key_dfct]['array_delta_FE'])/np.abs(self.all_data[key_dfct]['array_full_FE'])
                except : 
                    data_block = np.abs(self.all_data[key_dfct]['array_block'])/np.abs(self.all_data[key_dfct]['array_full_FE'])
                
                if key_dfct[0] == '0' :
                    full_data[id,:,0] = data_block
                    full_data[id,:,1] = np.nan
                else : 
                    full_data[id,:,1] = data_block 
                    full_data[id,:,0] = np.nan                      


            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            Cmap2 = plt.get_cmap('magma')
            color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            color_array2 = [Cmap2(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)
            sm2 = ScalarMappable(cmap=Cmap2, norm=norm)

            for id_T,T in enumerate(list_T) : 
                for dfct in range(full_data.shape[2]) :
                    data_T = full_data[:,id_T,dfct]
                    mask_nan = np.isnan(data_T)

                    #mask2 = data_T < 10.0 
                    mask2 = data_T > 0.0
                    mask = mask2 & ~mask_nan
                    #mask = ~mask_nan

                    mean_Fblock_T = np.mean(data_T[mask])
                    std_Fblock_T = np.std(data_T[mask])
                    dic_equiv = {1:r'$I_4$',0:r'$V_4$'}
                    dic_equivcolor = {1: color_array1,
                                      0: color_array2}
                    print(f'{dic_equiv[dfct]} ==> Temperature = {T} K,mean Fblock = {mean_Fblock_T} eV, sigma Fblock {std_Fblock_T} eV')

                    nb_bin = int(len(self.all_data)/20.0)
                    #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                    sns.histplot(x=data_T[mask], 
                                 color=dic_equivcolor[dfct][id_T], 
                                 label=r'%s ,Temperature : %2.1f K'%(dic_equiv[dfct],T), 
                                 bins=nb_bin,
                                 stat='count',
                                 kde=True,
                                 alpha=0.6,
                                 log_scale=True)

            # Add colorbars for each defect type
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            axins2 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper left')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for $I_4$', orientation='horizontal',shrink=0.1)      
            cbar2 = fig.colorbar(sm2, cax=axins2, label='Temperature (K) for $V_4$', orientation='horizontal',shrink=0.1) #,fraction=0.1, pad=0.02
            cbar1.ax.tick_params(labelsize=14)
            cbar2.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)
            cbar2.set_label('Temperature (K) for $V_4$', fontsize=16)

        #ax.set_xlim(1e-10,1e-1)
        ax.tick_params(which='both', width=1,labelsize=15)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')
        ax.set_xlabel(r'$\Delta F_\textrm{b}(T)$ [%s] in U.A'%(name), size=20)
        #plt.xscale('log')
        ax.set_ylabel(r'Density in A.U',size=20)
        #plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'delta_FE_{name}.{ext}',dpi=300)

        return


    def Analyse_overlap_distribution(self, list_T : List[float], name : str, vac_analysis : bool = False) -> None : 
        kb = 8.617e-5
        if not vac_analysis : 
            full_data = np.zeros((len(self.all_data),len(list_T)))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                try : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_delta_FE'])
                except : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_block'])

            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            color_array = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)

            for id_T,T in enumerate(list_T) : 
                data_T =  full_data[:-1,id_T] / (kb*T)
                mask_nan = np.isnan(data_T)

                #mask2 = data_T < 10.0 
                mask2 = data_T > 0.0
                mask = mask2 & ~mask_nan
                #mask = ~mask_nan

                nb_bin = int(len(self.all_data)/20.0)
                #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                sns.histplot(x=data_T[mask], 
                             color=color_array[id_T], 
                             label=f'Temperature : {T} K', 
                             bins=nb_bin,
                             stat='density',
                             kde=True,
                             alpha=0.6)
                
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for all defects', orientation='horizontal',shrink=0.1)
            cbar1.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)

        else : 
            full_data = np.zeros((len(self.all_data),len(list_T),2))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                #array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                try : 
                    data_block = np.abs(self.all_data[key_dfct]['array_delta_FE'])
                except : 
                    data_block = np.abs(self.all_data[key_dfct]['array_block'])
                
                if key_dfct[0] == '0' :
                    full_data[id,:,0] = data_block
                    full_data[id,:,1] = np.nan
                else : 
                    full_data[id,:,1] = data_block 
                    full_data[id,:,0] = np.nan                      


            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            Cmap2 = plt.get_cmap('magma')
            color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            color_array2 = [Cmap2(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)
            sm2 = ScalarMappable(cmap=Cmap2, norm=norm)

            for id_T,T in enumerate(list_T) : 
                for dfct in range(full_data.shape[2]) :
                    data_T = full_data[:,id_T,dfct] / (kb*T)
                    mask_nan = np.isnan(data_T)

                    #mask2 = data_T < 10.0 
                    mask2 = data_T > 0.0
                    mask = mask2 & ~mask_nan
                    #mask = ~mask_nan

                    mean_Fblock_T = np.mean(data_T[mask])
                    std_Fblock_T = np.std(data_T[mask])
                    dic_equiv = {1:r'$I_4$',0:r'$V_4$'}
                    dic_equivcolor = {1: color_array1,
                                      0: color_array2}
                    
                    nb_bin = int(len(self.all_data)/20.0)
                    #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                    sns.histplot(x=data_T[mask], 
                                 color=dic_equivcolor[dfct][id_T], 
                                 label=r'%s ,Temperature : %2.1f K'%(dic_equiv[dfct],T), 
                                 bins=nb_bin,
                                 stat='count',
                                 kde=True,
                                 alpha=0.6)

            # Add colorbars for each defect type
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            axins2 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper left')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for $I_4$', orientation='horizontal',shrink=0.1)      
            cbar2 = fig.colorbar(sm2, cax=axins2, label='Temperature (K) for $V_4$', orientation='horizontal',shrink=0.1) #,fraction=0.1, pad=0.02
            cbar1.ax.tick_params(labelsize=14)
            cbar2.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)
            cbar2.set_label('Temperature (K) for $V_4$', fontsize=16)

        #ax.set_xlim(1e-10,1e-1)
        ax.tick_params(which='both', width=1,labelsize=15)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')
#        ax.set_xlabel(r'$\left| \log \left( \displaystyle\int_{\mathcal{Q}^{\star}} \frac{\pi( \mathbf{q} )}{p( \mathbf{q} )} \mu_{\Pi}( \mathbf{q} ) \, d \mathbf{q} \right) \right|$ [%s] in A.U'%(name), size=18)
        ax.set_xlabel(r'$\left| \log \left( \mathbb{E}_{\mu_{\Pi}(\mathbf{q})} \left[ \frac{\pi( \mathbf{q} )}{p( \mathbf{q} )} \right] \right) \right|$ [%s] in A.U'%(name), size=18)
        #plt.xscale('log')
        ax.set_ylabel(r'Density in A.U',size=20)
        #plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'jarzynski_{name}.{ext}',dpi=300)

        return

    def Analyse_error_FreeEnergy(self, list_T : List[float], name : str, vac_analysis : bool = False, scaling_factor : float = 1e2) -> None : 
        if not vac_analysis : 
            full_data = np.zeros((len(self.all_data),len(list_T)))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                try : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_delta_FE'])
                except : 
                    full_data[id,:] = np.abs(self.all_data[key_dfct]['array_sigma_FE'])
            
            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            color_array = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)

            full_data *= scaling_factor

            for id_T,T in enumerate(list_T) : 
                data_T = full_data[:,id_T]
                mask_nan = np.isnan(data_T)

                #mask2 = data_T < 10.0 
                mask2 = data_T > 0.0
                mask = mask2 & ~mask_nan
                #mask = ~mask_nan

                mean_sigma_T = np.mean(data_T[mask])
                std_sigma_T = np.std(data_T[mask])
                print(f'Temperature = {T} K,mean error = {mean_sigma_T} eV, sigma error {std_sigma_T} eV')

                nb_bin = int(len(self.all_data)/20.0)
                #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                sns.histplot(x=data_T[mask], 
                             color=color_array[id_T], 
                             label=f'Temperature : {T} K', 
                             bins=nb_bin,
                             stat='density',
                             kde=True,
                             alpha=0.6,
                             log_scale=True)

            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for all defects', orientation='horizontal',shrink=0.1)
            cbar1.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)

        else : 
            full_data = np.zeros((len(self.all_data),len(list_T),2))
            for id, key_dfct in enumerate(self.all_data.keys()) :
                #array_delta_Ff = self.compute_formation_delta_anha_free_energy(self.all_data['bulk'], self.all_data[key_dfct])
                try : 
                    data_full = np.abs(self.all_data[key_dfct]['array_delta_FE'])
                except : 
                    data_full = np.abs(self.all_data[key_dfct]['array_sigma_FE'])
                
                if key_dfct[0] == '0' :
                    full_data[id,:,0] = data_full
                    full_data[id,:,1] = np.nan
                else : 
                    full_data[id,:,1] = data_full 
                    full_data[id,:,0] = np.nan                      

            full_data *= scaling_factor

            fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
            Cmap1 = plt.get_cmap('viridis')
            Cmap2 = plt.get_cmap('magma')
            color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(list_T))]
            color_array2 = [Cmap2(x) for x in np.linspace(0.0,1.0, num=len(list_T))]

            # Create ScalarMappables for the colorbars
            norm = Normalize(vmin=min(list_T), vmax=max(list_T))
            sm1 = ScalarMappable(cmap=Cmap1, norm=norm)
            sm2 = ScalarMappable(cmap=Cmap2, norm=norm)


            for id_T,T in enumerate(list_T) : 
                for dfct in range(full_data.shape[2]) :
                    data_T = full_data[:,id_T,dfct]
                    mask_nan = np.isnan(data_T)

                    #mask2 = data_T < 10.0 
                    mask2 = data_T > 0.0
                    mask = mask2 & ~mask_nan
                    #mask = ~mask_nan

                    mean_Fblock_T = np.mean(data_T[mask])
                    std_Fblock_T = np.std(data_T[mask])
                    dic_equiv = {1:r'$I_4$',0:r'$V_4$'}
                    dic_equivcolor = {1: color_array1,
                                      0: color_array2}
                    print(f'{dic_equiv[dfct]} ==> Temperature = {T} K,mean error = {mean_Fblock_T} eV, sigma error {std_Fblock_T} eV')

                    nb_bin = int(len(self.all_data)/20.0)
                    #plt.hist(data_T[mask], bins=100, label=f'Temperature : {T} K',density=True, color=color_array[id_T])
                    sns.histplot(x=data_T[mask], 
                                 color=dic_equivcolor[dfct][id_T], 
                                 label=r'%s ,Temperature : %2.1f K'%(dic_equiv[dfct],T), 
                                 bins=nb_bin,
                                 stat='count',
                                 kde=True,
                                 alpha=0.6,
                                 log_scale=True)

            # Add colorbars for each defect type
            axins1 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper right')

            axins2 = inset_axes(ax,
                               width="40%",
                               height='5%',
                               loc='upper left')

            cbar1 = fig.colorbar(sm1, cax=axins1, label='Temperature (K) for $I_4$', orientation='horizontal',shrink=0.1)      
            cbar2 = fig.colorbar(sm2, cax=axins2, label='Temperature (K) for $V_4$', orientation='horizontal',shrink=0.1) #,fraction=0.1, pad=0.02
            cbar1.ax.tick_params(labelsize=14)
            cbar2.ax.tick_params(labelsize=14)
            cbar1.set_label('Temperature (K) for $I_4$', fontsize=16)
            cbar2.set_label('Temperature (K) for $V_4$', fontsize=16)

        ax.tick_params(which='both', width=1,labelsize=15)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')
        ax.set_xlabel(r'$\tilde{\sigma}(T)$ [%s] in eV'%(name), size=20)
        #plt.xscale('log')
        ax.set_ylabel(r'Density in A.U', size=20)
        #plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f'error_FE_{name}.{ext}',dpi=300)

        return


    def mean_Fenergy_T(self, dic_data : Dict[str,FData],name : str) -> None : 
        fig, ax =  plt.subplots(1,1,figsize=(9,6)) 
        Cmap1 = plt.get_cmap('viridis')
        color_array1 = [Cmap1(x) for x in np.linspace(0.0,1.0, num=len(dic_data))]


        kb = 8.617e-5
        for id, key in enumerate(dic_data.keys()) : 
            array_T = dic_data[key]['T']
            mean_FT = dic_data[key]['mean_F']
            std_FT = dic_data[key]['std_F']

            ax.fill_between(array_T,mean_FT - std_FT, 
                                  mean_FT + std_FT, 
                                  alpha=0.4,
                                  color=self.mixing_color(to_rgba('white'), color_array1[id], 0.5),
                                  zorder=0)

            # linear interpolation 
            fake_T = np.linspace(np.amin(array_T), np.amax(array_T), num=300)
            
            lin = LinearRegression()
            lin.fit( array_T[:2].reshape((-1,1)), mean_FT[:2] )
            #plt.plot(fake_T, lin.predict( fake_T.reshape((-1,1)) ), color=color_array1[id], ls='--')

            # variance estimator 
            array_T_var = np.concatenate((np.array([0.0]), array_T[:2]))
            array_std_var = np.concatenate((np.array([0.0]), std_FT[:2]))
            def f(x,a,b,c) : 
                return a*x**2 + b*x + c
            
            var_estimator, pcov = curve_fit(f, array_T_var, np.power(array_std_var,2), bounds=([0.0,-np.inf,0.0], [np.inf, np.inf, np.inf]))
            print(var_estimator,np.sqrt(var_estimator[0])/kb, key)

            #ax.fill_between(array_T, lin.predict( array_T.reshape((-1,1)) ) - np.sqrt(f(array_T,*var_estimator)),
            #            lin.predict( array_T.reshape((-1,1)) ) + np.sqrt(f(array_T,*var_estimator)), 
            #            color=color_array1[id], 
            #            #label=r'%s $\langle E_\textrm{f} \rangle_{\texttt{ARTn}}$ = %1.2f eV, $\langle S_\textrm{h,f} \rangle_{\texttt{ARTn}}$ = %1.2f $k_B$'%(key,lin.intercept_,-lin.coef_[0]/kb),
            #            #s=36,
            #            alpha=0.7,
            #            zorder=10)

            ax.errorbar( array_T, mean_FT, yerr=np.sqrt(f(array_T,*var_estimator)),
                        color=color_array1[id], 
                        label=r'%s $\mathbb{E}_{\texttt{ARTn}} \left[ E_\textrm{f} \right]$ = %1.2f eV, $\mathbb{E}_{\texttt{ARTn}} \left[ S_\textrm{h,f} \right]$ = %1.2f $k_B$'%(key,lin.intercept_,-lin.coef_[0]/kb),
                        capsize=5,
                        marker='o', markersize=8,
                        mec='grey',
                        zorder=10)

            plt.plot( fake_T, lin.predict( fake_T.reshape((-1,1)) ), linestyle='--', color=color_array1[id] )

            # cubic spline 
            cs_FT = CubicSpline(array_T, mean_FT)
            plt.plot(fake_T, cs_FT(fake_T), color=color_array1[id])

            #ax.text(0.2,0.1,r'$E_f$ = %1.2f eV, $S_f$ = %1.2f $k_B$'%(lin.intercept_,lin.coef_[0]/kb), 
            #        fontsize=11, horizontalalignment='center', 
            #        verticalalignment='center', 
            #        transform = ax.transAxes )

        majorFormatter = mpl.ticker.FormatStrFormatter('%1.1f')
        majorLocator = mpl.ticker.MultipleLocator(1.0)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_formatter(majorFormatter)
        ax.yaxis.set_minor_locator(minorLocator)
        
        majorFormatter = mpl.ticker.FormatStrFormatter('%1.1f')
        majorLocator = mpl.ticker.MultipleLocator(100)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.tick_params(which='both', width=1,labelsize=15)
        ax.tick_params(which='major', length=8)
        ax.tick_params(which='minor', length=3, color='black')

        ax.set_xlabel(r'Temperature in K', size=20)
        #plt.xscale('log')
        ax.set_ylabel(r'$\mathbb{E}_{\texttt{ARTn}} \left[ F_{\textrm{f}} \right] (T)$ in eV', size=20)
        plt.legend(frameon=False, fontsize=14)
        plt.tight_layout()
        plt.savefig(f'free_energy_dfct_{name}.{ext}',dpi=300)     

#################################
### INPUTS
#################################
#path_pkl = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/mab_desc_k2bj5_r6.pickle'
path_pkl = '/home/lapointe/WorkML/FreeEnergySurrogate/full_data/it2/desc_mab_Ff.pickle'
path_pkl_eam = '/home/lapointe/WorkML/FreeEnergySurrogate/data/data_eam_new.pickle'
list_T = [300., 600.0, 900., 1200.]
list_T_eam = [200., 500., 800.]
#################################

dic_F : Dict[str,FData] = {}
dic_Fd : Dict[str,FData] = {}
#eam
print('... EAM data ...')
obj_Ff = Analyse(path_pkl_eam)

array_T, mean_F, std_F = obj_Ff.Analyse_Delta_anah_FreeEnergy(list_T_eam, 'EAM')
#array_Td, mean_Fd, std_Fd = obj_Ff.Analyse_anah_FreeEnergy(list_T_eam, 'EAM')
dic_F[r'$I_4$ [EAM]'] = {'T':array_T, 'mean_F':mean_F, 'std_F':std_F}
#dic_Fd[r'$I_4$ [EAM]'] = {'T':array_Td, 'mean_F':mean_Fd, 'std_F':std_Fd}

obj_Ff.Analyse_Fblock_FreeEnergy(list_T_eam, 'EAM')
obj_Ff.Analyse_overlap_distribution(list_T_eam,'EAM')
print()

#ml
print('... ML data ...')
obj_Ff = Analyse(path_pkl)

# formation free energy
array_T, mean_F, std_F = obj_Ff.Analyse_Delta_anah_FreeEnergy(list_T, 'ML',vac_analysis=True)
#array_Td, mean_Fd, std_Fd = obj_Ff.Analyse_anah_FreeEnergy(list_T, 'ML', vac_analysis=True)
dic_F[r'$I_4$ [ML]'] = {'T':array_T, 'mean_F':mean_F[:,1], 'std_F':std_F[:,1]}
dic_F[r'$V_4$ [ML]'] = {'T':array_T, 'mean_F':mean_F[:,0], 'std_F':std_F[:,0]}
#dic_Fd[r'$I_4$ [ML]'] = {'T':array_Td, 'mean_F':mean_Fd[:,1], 'std_F':std_Fd[:,1]}
#dic_Fd[r'$V_4$ [ML]'] = {'T':array_Td, 'mean_F':mean_Fd[:,0], 'std_F':std_Fd[:,0]}

obj_Ff.Analyse_Fblock_FreeEnergy(list_T, 'ML',vac_analysis=True)
obj_Ff.Analyse_error_FreeEnergy(list_T, 'ML', vac_analysis=True)
obj_Ff.Analyse_overlap_distribution(list_T, 'ML', vac_analysis=True)
print()


# temperature dependancy
obj_Ff.mean_Fenergy_T(dic_F,'')
#obj_Ff.mean_Fenergy_T(dic_Fd,'anah_Ff')
#plt.show()
