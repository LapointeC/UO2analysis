import numpy as np
import scipy.linalg
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

from sklearn.metrics import mean_squared_error
from scipy.stats import kde
import pickle
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts}'})

from typing import Tuple, Dict, TypedDict
from matplotlib.axes import Axes


class MyLinearRegression : 
    def __init__(self, compute_score : bool = True, 
                 lamb : float = 1e-12,
                 lamb2 : float = 5e-5) -> None : 
        self.compute_score = compute_score
        self.lamb = lamb
        self.lamb2 = lamb2
        self.coef_ = None 
        self.coef1_ = None 
        self.coef2_ = None 
        self.intercept_ = None 
        self.weights = None  

    def fit(self, X : np.ndarray, Y : np.ndarray, dim : int = None) -> None :
        self.intercept_ = np.mean(Y)

        if dim is None : 
            self.coef_, res, rank, s = np.linalg.lstsq(X, Y - self.intercept_, rcond=self.lamb)
            self.weights = self.coef_
            print(f'Rank of design matrix = {rank}')

        else : 
            self.coef1_, res, rank1, s = np.linalg.lstsq(X[:,:dim], Y - self.intercept_, rcond=self.lamb)
            self.coef2_, res, rank2, s = np.linalg.lstsq(X[:,dim:], Y - self.intercept_ - X[:,:dim]@self.coef1_, rcond=self.lamb2) 
            print(f'Rank of design matrix = {rank1}, {rank2}')
        return 
    
    def compute_rmse(self, X : np.ndarray, Y : np.ndarray) -> None : 
        return np.sqrt(np.mean(np.power(self.predict(X) - Y, 2.0),axis=0))

    def predict(self, X : np.ndarray, dim : int = None) -> np.ndarray :
        if dim is None : 
            return X@self.weights + self.intercept_
        else :
            return X[:,:dim]@self.coef1_ + X[:,dim:]@self.coef2_ + self.intercept_

class MyLinearRegressionNorm : 
    def __init__(self, compute_score : bool = True, 
                 lamb : float = 1e-5) -> None : 
        self.compute_score = compute_score
        self.lamb = lamb
        self.coef_ = None 

        self.sigma_lin = None 
        self.sigma_quad = None

        self.intercept_ = None 
        self.weights = None  

    def fit(self, X : np.ndarray, Y : np.ndarray, dim : int = None) -> None :
        self.intercept_ = np.mean(Y)

        if dim is None : 
            raise NotImplementedError('This option is not implemented')

        else : 
            Xc = X.copy()
            _ , svd_lin, _ = np.linalg.svd(X[:,:dim])
            _ , svd_quad, _ = np.linalg.svd(X[:,dim:])

            # keep sigma values
            self.sigma_lin = np.amax(svd_lin)
            self.sigma_quad = np.amax(svd_quad)

            #rescaling 
            Xc[:,:dim] *= 1.0/self.sigma_lin
            Xc[:,dim:] *= 1.0/self.sigma_quad


            self.coef_, res, rank, s = np.linalg.lstsq(Xc, Y - self.intercept_, rcond=self.lamb)
            print(f'Rank of design matrix = {rank}')
        return 
    
    def compute_rmse(self, X : np.ndarray, Y : np.ndarray) -> None : 
        return np.sqrt(np.mean(np.power(self.predict(X) - Y, 2.0),axis=0))

    def predict(self, X : np.ndarray, dim : int = None) -> np.ndarray :
        if dim is None : 
            return X@self.weights + self.intercept_
        else :
            Xc = X.copy()
            Xc[:,:dim] *= 1.0/self.sigma_lin
            Xc[:,dim:] *= 1.0/self.sigma_quad        
            return Xc@self.coef_ + self.intercept_

class DicData(TypedDict) : 
    #model : BayesianRidge
    model : MyLinearRegression
    ratio : float
    RMSE : float

class InterpolationWeight(TypedDict) : 
    weight : np.ndarray
    intercept : np.ndarray

class WeightMapper :
    def __init__(self,fusion_temperature : float) -> None :
        self.dic_data : Dict[float,DicData] = {}
        self.fusion_temperature = fusion_temperature
        self.interpolate_model : InterpolationWeight = {'weight':None,'intercept':None}

        self.size_weights = None 
        self.array_coeff = None 
        self.array_intercept = None 

    def update_mapper(self,temperature : float,
                    #model : BayesianRidge,
                    model : MyLinearRegression,
                    rmse : float) :
        if temperature not in self.dic_data.keys():
            self.dic_data[temperature] = {'model':model,'RMSE':rmse,'ratio':temperature/self.fusion_temperature}
        else :
            self.dic_data[temperature]['model'] = model
            self.dic_data[temperature]['ratio'] = temperature/self.fusion_temperature
            self.dic_data[temperature]['RMSE'] = rmse

    def extract_first_model(self) -> BayesianRidge :
        return self.dic_data[ np.amin([ key for key in self.dic_data.keys() ]) ]['model']

    def get_size_interpolation(self, size_model : int, order : int) -> None :
        self.size_weights = size_model
        self.interpolate_model['weight'] = np.zeros( (size_model, order+1) )
        self.interpolate_model['intercept'] = np.zeros( size_model )
        return 

    def InterpolateWeights(self, order_interpolation : int) -> None :
        # interpolation for coef

        first_model = self.extract_first_model()
        self.get_size_interpolation( len(first_model.coef_), order_interpolation )

        array_coef_models = np.zeros((len(first_model.coef_),len(self.dic_data)))
        array_intercept_models = np.zeros( len(self.dic_data) )
        array_ratio = np.zeros(len(self.dic_data))
        for id_temperature, temperature in enumerate(self.dic_data.keys()) : 
            array_coef_models[:,id_temperature] = self.dic_data[temperature]['model'].coef_
            array_intercept_models[id_temperature] = self.dic_data[temperature]['model'].intercept_
            array_ratio[id_temperature] = self.dic_data[temperature]['ratio']

        self.array_coeff = array_coef_models
        self.array_intercept = array_intercept_models

        # fitting coef ! 
        for id_weight, weights in enumerate(array_coef_models) : 
            array_interpolation_weights = np.polyfit(array_ratio, weights, order_interpolation)
            self.interpolate_model['weight'][id_weight,:] = array_interpolation_weights

        # fitting intercept ! 
        array_interpolation_intercepts = np.polyfit(array_ratio, array_intercept_models, order_interpolation)
        self.interpolate_model['intercept'] = array_interpolation_intercepts

       
        #plt.scatter(array_ratio,array_weight_T,color=colors[id_weight],alpha=0.7)
        #plt.plot(fake_array,np.poly1d(array_interpolation)(fake_array),color=colors[id_weight],alpha=0.5)
        #self.interpolate_model['weight'].append(array_interpolation)

        #plt.yscale('log')
        plt.ylim(-0.0002,0.0002)
        plt.tight_layout()
        plt.savefig('weight.pdf')

        # interpolation for intercept
        #list_ratio_intercept, list_intercept_T = [], []
        #for temperature in self.dic_data.keys():
        #    list_intercept_T.append(self.dic_data[temperature]['model'].intercept_)
        #    list_ratio_intercept.append(self.dic_data[temperature]['ratio'])
        #array_ratio_intercept = np.asarray(list_ratio_intercept)
        #array_intercept_T = np.asarray(list_intercept_T)
        #array_inter = np.polyfit(array_ratio_intercept,array_intercept_T,order_interpolation)
        #self.interpolate_model['intercept'] = array_inter

    def PlotWeightsInterpolation(self, color_map : str = 'gnuplot') -> None :
        plt.figure()
        Cmap1 = plt.get_cmap(color_map)
        colors = [Cmap1(el) for el in np.linspace(0.0,1.0,num=len(self.extract_first_model()))]

        dic_temperature = np.array([key for key in self.dic_data.keys()])
        
        # interpolation 
        fake_array_ratio = np.linspace(np.amin(dic_temperature)/self.fusion_temperature, np.amax(dic_temperature)/self.fusion_temperature,
                                       num=2000)
        
        for id_weight, inter_model in enumerate(self.interpolate_model['weight']) :
            plt.scatter(dic_temperature/self.fusion_temperature, self.array_coeff[id_weight,:],
                        color=colors[id_weight],alpha=0.7)

            poly_fit_weight_i = np.poly1d(inter_model) 
            weight_i_T = poly_fit_weight_i(fake_array_ratio)
            plt.plot(fake_array_ratio,weight_i_T,color=colors[id_weight],alpha=0.5)           

        return 

    def predict_weights_given_T(self,T : float) -> Tuple[np.ndarray, float]:
        array_weights = np.zeros(self.size_weights)
        # weights
        for id_w ,inter_model in enumerate(self.interpolate_model['weight']) :
             poly_fit_weight_i = np.poly1d(inter_model)
             weight_i_T = poly_fit_weight_i(T/self.fusion_temperature)
             array_weights[id_w] = weight_i_T

        # intercept
        poly_fit_intercept = np.poly1d(self.interpolate_model['intercept'])
        intercept_T = poly_fit_intercept(T/self.fusion_temperature)

        return array_weights, intercept_T

    def ScaleModelwithT(self,T : float) -> BayesianRidge :
        model_copy = BayesianRidge(compute_score=True)
        coef_T, intercept_T = self.predict_weights_given_T(T)
        model_copy.coef_ = coef_T
        model_copy.intercept_ = intercept_T
        return model_copy


class ModelFE(TypedDict) : 
    #model : BayesianRidge
    model : MyLinearRegression | BayesianRidge
    rmse : float 
    temperature : float 

class RegressorFreeEnergy : 
    def __init__(self, design_matrix : np.ndarray, 
                dic_target : Dict[float, np.ndarray],
                dic_error : Dict[float, np.ndarray],
                dic_sigma : Dict[float, np.ndarray],
                array_idx_config : np.ndarray,
                mask_mcd : np.ndarray = None,
                percentile : float = None,
                test_size : float = 0.4,
                random_state : int = 435643,
                normalised_design_matrix : bool = False) -> None :

        self.test_size = test_size
        self.random_state = random_state
        self.normalised_design_matrix = normalised_design_matrix

        self.dic_model : Dict[float, ModelFE] = {temp:{'model':None,
                                                       'rmse':None,
                                                       'temperature':None} for temp in dic_target.keys()}
        self.design_matrix = design_matrix
        self.dic_target = dic_target 
        self.dic_error = dic_error
        self.dic_sigma = dic_sigma
        self.array_idx_config = array_idx_config

        self.mask_mcd = mask_mcd
        self.percentile = percentile

    def GetMaxError(self,Ycalc : np.ndarray,
                        Ypred : np.ndarray,
                        array_idx : np.ndarray,
                        temperature : float,
                        axis : Axes,
                        coord_axis : Tuple[int],
                        tolerance_error : float) -> None :
       
        c0, c1 = coord_axis
        mask_mean = Ycalc > np.mean(Ycalc)
        
        array_error = np.abs(Ycalc - Ypred)
        mask_error = array_error > tolerance_error

        maxE_Ycalc = Ycalc[mask_error & mask_mean]
        maxE_Ypred = Ypred[mask_error & mask_mean]
        print(f'Temperature : {temperature} K')
        print(f'Max error config : {[str(int(el))[1:] for el in array_idx[mask_error & mask_mean]]}')
        print(f'Errors : {array_error[mask_error & mask_mean]}')

        axis[c0, c1].scatter(maxE_Ycalc,maxE_Ypred,color='firebrick',alpha=0.7)
        
        return 

    def PlotCorrelation(self,Ycalc : np.ndarray,
                        Ypred : np.ndarray,
                        temperature : float,
                        axis : Axes,
                        coord_axis : Tuple[int],
                        error : float,
                        sigma : float = 2.5) -> None : 
        c0, c1 = coord_axis
        """plot the regression results with respect to the data density"""

        gaussian_kernel = kde.gaussian_kde([Ycalc,Ypred])
        Cmap1 = plt.get_cmap('gnuplot')
        max_k, min_k = np.amax(gaussian_kernel([Ycalc,Ypred])), np.amin(gaussian_kernel([Ycalc,Ypred]))
        a, b = 1.0/(max_k-min_k), min_k/(min_k-max_k)
        colors = [Cmap1(a*gaussian_kernel([Ycalc[i],Ypred[i]])+b) for i in range(len(Ycalc))]

        for k in range(len(Ycalc)) :
            axis[c0, c1].scatter(Ycalc[k],Ypred[k],color=colors[k],alpha=0.7)

        fake_line = np.linspace(np.amin(Ycalc),np.max(Ycalc),num=2000)
        axis[c0, c1].plot(fake_line,fake_line,linestyle='dashed',color='black')
        axis[c0, c1].fill_between(fake_line,fake_line - sigma*error, 
                                  fake_line + sigma*error, 
                                  alpha=0.5,
                                  color='grey',
                                  zorder=0)

        axis[c0, c1].set_xlabel(r'Anharmonic $F_{\textrm{anah}}(T)$ (T=%s K) in eV'%(str(temperature)),fontsize = 18)
        axis[c0, c1].set_ylabel(r'Predicted anharmonic $\tilde{F}_{\textrm{anah}}(T)$ (T=%s K) in eV'%(str(temperature)),fontsize = 18)

        #if c1 == 0 : 
        #    axis[c0, c1].text(0.8,0.1,r'RMSE (train) : %3.2f eV'%(np.sqrt(mean_squared_error(Ycalc,Ypred))), fontsize=15, horizontalalignment='center', verticalalignment='center', transform = axis[c0,c1].transAxes )
        #if c1 == 1 : 
        #    axis[c0, c1].text(0.8,0.1,r'RMSE (test) : %3.2f eV'%(np.sqrt(mean_squared_error(Ycalc,Ypred))), fontsize=15, horizontalalignment='center', verticalalignment='center', transform = axis[c0,c1].transAxes )
        if c0 == 0 : 
            axis[c0, c1].text(0.75,0.1,r'RMSE (train) : %3.2f eV'%(np.sqrt(mean_squared_error(Ycalc,Ypred))), fontsize=20, horizontalalignment='center', verticalalignment='center', transform = axis[c0,c1].transAxes )
        if c0 == 1 : 
            axis[c0, c1].text(0.75,0.1,r'RMSE (test) : %3.2f eV'%(np.sqrt(mean_squared_error(Ycalc,Ypred))), fontsize=20, horizontalalignment='center', verticalalignment='center', transform = axis[c0,c1].transAxes )
        
        
        #majorFormatter = mpl.ticker.FormatStrFormatter('%1.1f')
        #majorLocator = mpl.ticker.MultipleLocator(0.5)
        #minorLocator = mpl.ticker.AutoMinorLocator(2)
        #axis[c0].xaxis.set_major_locator(majorLocator)
        #axis[c0].xaxis.set_major_formatter(majorFormatter)
        #axis[c0].xaxis.set_minor_locator(minorLocator)
        #axis[c0].yaxis.set_major_locator(majorLocator)
        #axis[c0].yaxis.set_major_formatter(majorFormatter)
        #axis[c0].yaxis.set_minor_locator(minorLocator)
        axis[c0,c1].tick_params(which='both', width=1,labelsize=15)
        axis[c0,c1].tick_params(which='major', length=8)
        axis[c0,c1].tick_params(which='minor', length=3, color='black')

        return

    def build_percentile_target(self, target : np.ndarray, percentile : float = 5.0) -> np.ndarray : 
        percentile_values_inf = np.percentile(target, percentile/2.0)
        percentile_values_sup = np.percentile(target, 100 - percentile/2.0)
        mask_inf = target > percentile_values_inf
        mask_sup = target < percentile_values_sup
        return mask_inf & mask_sup

    def plot_theoritical_intercept(self, array_T : np.ndarray,
                                   array_intercept : np.ndarray,
                                   Natoms : int = 1028) -> None : 
        
        def th_intercept(array_rescaleT : np.ndarray,
                         T0 : float, 
                         N_atoms : int,
                         intercept0 : float) -> np.ndarray : 
            kb = 8.617e-5
            return -kb*array_rescaleT*T0*3*(N_atoms-1)*np.log(array_rescale_T) + kb*array_rescale_T*intercept0

        plt.figure()
        array_rescale_T = np.linspace(1.0,(np.max(array_T))/np.amin(array_T), num = 150)
        
        #theoritical function
        inter0 = array_intercept[0]
        T0 = array_T[0]
        plt.plot( array_rescale_T, th_intercept(array_rescale_T, T0, Natoms, inter0),
                 label=r'Theoritical $F_h(T/T_0)$', color='lightgreen')
        
        #my intercept
        array_T *= 1.0/T0 
        plt.scatter( array_T, array_intercept, 
                    label=r'Fitted intercept $\tilde{F}_h(T/T_0)$',color='grey')
        plt.xlabel(r'Rescaled temperature $T/T_0$')
        plt.ylabel(r'Intercept function $F_h(T/T_0)$ in eV')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig('fit_inter_th.png', dpi=300)

        return 

    def BuildAllModels(self, tolerance : float = 1e-5, dim_lin : int = None) -> Dict[float, ModelFE] : 
        #fig, axis = plt.subplots(nrows=len(self.dic_target), ncols=2, figsize=(12,5*len(self.dic_target)))
        fig, axis = plt.subplots(nrows=2, ncols=len(self.dic_target), figsize=(8*len(self.dic_target),12))
        compt = 0
        list_inter, list_T = [], []
        for temp, target_temp in self.dic_target.items() : 
            #self.dic_model[temp]['model'] = BayesianRidge(compute_score=True)
            self.dic_model[temp]['model'] = MyLinearRegression(compute_score=True)
            self.dic_model[temp]['temperature'] = temp

            # check nan for each temperature ! 
            mask_nan = np.isnan(target_temp)
            mask_error = self.dic_error[temp] < tolerance
            print(len(mask_error),len( [el for el in mask_error if el] ), temp)

            med_error = np.mean(self.dic_sigma[temp][mask_error])
            print(med_error)

            mask = ~mask_nan & mask_error
            if self.mask_mcd is not None :
                mask = mask & self.mask_mcd

            Y = target_temp[mask]
            X = self.design_matrix[mask,:]
            idx_config_temp = self.array_idx_config[mask]

            if self.normalised_design_matrix : 
                mean = np.mean(X, axis = 0)
                std = np.std(X, axis = 0)
                X = (X-mean)/std 

            if self.percentile is not None : 
                mask_percentile = self.build_percentile_target(Y, self.percentile)
                Y = Y[mask_percentile]
                X = X[mask_percentile,:]
                idx_config_temp = idx_config_temp[mask_percentile]

            # train test part ! 
            X_train, X_test, Y_train, Y_test, idx_train, _ =  train_test_split(X,Y,idx_config_temp, 
                                                                 test_size=self.test_size, 
                                                                 random_state=self.random_state)
            self.dic_model[temp]['model'].fit(X_train, Y_train, dim=dim_lin)            
            Y_pred = self.dic_model[temp]['model'].predict(X, dim=dim_lin)
            Y_pred_test = self.dic_model[temp]['model'].predict(X_test, dim=dim_lin)
            Y_pred_train = self.dic_model[temp]['model'].predict(X_train, dim=dim_lin)
            self.dic_model[temp]['rmse'] = np.sqrt(mean_squared_error(Y,Y_pred))

            #self.PlotCorrelation(Y_test,
            #                     Y_pred_test,
            #                     temp,
            #                     axis,
            #                     [compt,1],
            #                     med_error)
            #self.PlotCorrelation(Y_train,
            #                 Y_pred_train,
            #                 temp,
            #                 axis,
            #                 [compt,0],
            #                 med_error)  

            #self.GetMaxError(Y_train,
            #                 Y_pred_train,
            #                 idx_train,
            #                 temp,
            #                 axis,
            #                 [compt,0],
            #                 1.5*np.sqrt(mean_squared_error(Y_train,Y_pred_train)))

            scale = 2.0
            self.PlotCorrelation(Y_test,
                                 Y_pred_test,
                                 temp,
                                 axis,
                                 [1,compt],
                                 scale*med_error)
            self.PlotCorrelation(Y_train,
                             Y_pred_train,
                             temp,
                             axis,
                             [0,compt],
                             scale*med_error)  

            self.GetMaxError(Y_train,
                             Y_pred_train,
                             idx_train,
                             temp,
                             axis,
                             [0,compt],
                             1.5*np.sqrt(mean_squared_error(Y_train,Y_pred_train)))


            compt += 1
            list_inter.append(self.dic_model[temp]['model'].intercept_)
            list_T.append(temp)

        plt.tight_layout()
        plt.savefig('free_energy_regression.pdf',dpi=300)

        # theoritical intercept...
        array_T = np.array(list_T)
        array_inter = np.array(list_inter)
        self.plot_theoritical_intercept(array_T,array_inter)

        plt.show()
        return self.dic_model