from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import null_space

from typing import List, Dict, Tuple, TypedDict

class DictPlane(TypedDict) : 
    bary : np.ndarray
    normal_vect : np.ndarray
    dic_constraint : Dict[str,List[np.ndarray] | List[str]] 

class DictIntersection(TypedDict) : 
    intersection_vector : np.ndarray
    intersection_point : np.ndarray
    inter_2D : None | np.ndarray


class IntersectionObject : 
    """Initialisation of the intersection object"""
    def __init__(self, grid_MU1 : np.ndarray, grid_MU2 : np.ndarray, list_different_plane : List[str]) -> None : 
        """Initialise IntersectionObject
        
        Parameters 
        ----------

        grid_MU1 : np.ndarray
            Discrete 2D of mu1 chemical potential 

        grid_MU2 : np.ndarray
            Discrete 2D of mu2 chemical potential        

        list_different_plane : List[str]
            List of name of hyperplane to draw

        """
        self.grid_MU_1 = grid_MU1
        self.grid_MU_2 = grid_MU2
        self.list_different_plane = list_different_plane

        self.threshold_grid = (np.amax(grid_MU1) - np.amin(grid_MU1))/grid_MU1.shape[0] + (np.amax(grid_MU2) - np.amin(grid_MU2))/grid_MU2.shape[0]

    def build_intersection_vector(self, plane1 : str, plane2 : str, method : str = 'least_square') -> np.ndarray : 
        """Compute the intersection vector between two hyperplanes
        
        Parameters
        ----------

        plane1 : str 
            Name of the first hyperplane 

        plane2 : str 
            Name of the second hyperplane

        method : str 
            Method to evaluate the intersection vector : least_square (based on L2 norm minimisation) and kernel (null space) 

        Returns
        -------

        np.ndarray 
            Intersection vector between plane1 and plane2

        """
        normal1 = self.dic_plane[plane1]['normal_vect']
        normal2 = self.dic_plane[plane2]['normal_vect']
        A = np.array([normal1[:-1],normal2[:-1]])
        if method == 'least_square' : 
            line = -A.T@np.linalg.inv(A@A.T)@np.array([[normal1[-1]],[normal2[-1]]])

        if method == 'kernel' : 
            line = null_space(A)

        return line

    def find_intersection_point(self, plane1 : str, plane2 : str, nearest_plane : List[str], threshold : float) -> np.ndarray | None : 
        """find the intersection point between two planes"""
        vect_plane_to_find = [plane1,plane2]
        last_plane = nearest_plane[0]
        coordinate_12 = None

        for plane_list in nearest_plane :  
            if last_plane[0] != plane_list[0] : 
                """build the permutation vector"""
                vect_perm1 = [last_plane[0],plane_list[0]]
                vect_perm2 = [plane_list[0],last_plane[0]]
                if vect_perm1 == vect_plane_to_find or vect_perm2 == vect_plane_to_find : 
                    id1_last, id2_last = last_plane[1][0] , last_plane[1][1]
                    id1_plane, id2_plane = plane_list[1][0], plane_list[1][1] 
                    coordinate_last = np.asarray([self.grid_MU_1[id1_last,id2_last],self.grid_MU_2[id1_last,id2_last]])
                    coordinate_plane = np.asarray([self.grid_MU_1[id1_plane,id2_plane],self.grid_MU_2[id1_plane,id2_plane]])
                    last_plane = plane_list
                    if np.linalg.norm(coordinate_last-coordinate_plane) < threshold : 
                        coordinate_12 = 0.5*(coordinate_plane+coordinate_last)
                        break

                    else : 
                        continue

            else : 
                last_plane = plane_list
                continue 
    
        return coordinate_12

    def build_plane_intersection_dictionnary(self, nearest_plane : str, dic_vect : Dict[str,np.ndarray], dic_constraint : Dict[str,List[np.ndarray] | List[str]], method : str) -> None : 
        """Build the dictionnary to plot properly intersections
        
        Parameters 
        ----------

        nearest_plane : str 
            Name of the nearest plane to compute the intersection point and line vector

        dic_vect : Dict[str,np.ndarray]
            Dictionnaray of normal vector (name of hyperplane are keys of the dictionnary)

        dic_constraint : Dict[str,List[np.ndarray] | List[str]]
            Dictionnary of chemical potential domains with the same format than in build_bounds method in DataSurfaceObject

        method : str
            Type of method used in build_intersection_vector method

        """
        self.dic_plane : Dict[str,DictPlane] =  {}
        self.dic_intersection : Dict[str, DictIntersection | bool] = {}
 
        for plane in self.list_different_plane :
            """We will first compute the barycenter of each plane"""
            barycenter_plane = np.zeros(2)
            compt_plane = 0
            for near in nearest_plane : 
                if near[0] == plane : 
                    compt_plane +=1 
                    id1, id2 = near[1]
                    barycenter_plane += np.asarray([ self.grid_MU_1[id1,id2], self.grid_MU_2[id1,id2]])
                else : 
                    continue 
            barycenter_plane *= 1.0/compt_plane

            """filling the dictionnary for each plane"""
            self.dic_plane[plane] = {'bary':barycenter_plane,'normal_vect':dic_vect[plane],'constraint':dic_constraint[plane]}

        if len(self.list_different_plane) == 1 : 
            self.dic_intersection['empty'] = True

        else : 
            """here is the filling part for the intersection object"""
            self.dic_intersection['empty'] = False 
            for i, plane_i in enumerate(self.list_different_plane) : 
                for j, plane_j in enumerate(self.list_different_plane) : 
                    if i > j : 
                        name_ij = '%s@%s'%(plane_i,plane_j)
                        line_ij = self.build_intersection_vector(plane_i,plane_j,method=method)
                        coordinate_ij = self.find_intersection_point(plane_i,plane_j,nearest_plane,2*self.threshold_grid)
                        if coordinate_ij is None : 
                            print('Troubling with intersection %s we change the threshold intersection '%(name_ij))
                            coordinate_ij = self.find_intersection_point(plane_i,plane_j,nearest_plane,15*self.threshold_grid)
                        self.dic_intersection[name_ij] = {'intersection_vector':line_ij,'intersection_point':coordinate_ij,'inter_2D':None}
        return 

    def check_generator_line(self) -> None : 
        """Check the line generated by the previous method 
        Line vector is chosen to be the most orthogonal to the barycenter vector between the two domains
        """
        for key_inter in self.dic_intersection.keys() :
            if key_inter == 'empty' : 
                if self.dic_intersection['empty'] == True : 
                    break

            if key_inter != 'empty' : 
                max_value = 1e4
                inter_to_keep = None

                intersection_vect = self.dic_intersection[key_inter]['intersection_vector'].flatten()
                inter1 = np.asarray([ intersection_vect[0], intersection_vect[1] ])
                inter2 = np.asarray([ -intersection_vect[0], intersection_vect[1] ])
                list_inter = [inter1,inter2]

                plane1, plane2 = key_inter.split('@')[0], key_inter.split('@')[1]
                bary1 , bary2 = self.dic_plane[plane1]['bary'], self.dic_plane[plane2]['bary']
                bary_12 = bary2 - bary1
                for inter in list_inter :
                    if abs(inter@bary_12) < max_value :
                        max_value = abs(inter@bary_12)
                        inter_to_keep = inter
               
                self.dic_intersection[key_inter]['inter_2D'] = inter_to_keep
        return 

    def apply_constraint(self,mu1_l : np.ndarray, mu2_l : np.ndarray, plane : str) -> Tuple[np.ndarray,np.ndarray]: 
        """Apply the constraint on removed element for chemical potential to draw limit stability domain
        
        Parameters
        ----------

        mu1_l : np.ndarray 
            Possible values of first chemical potential 

        mu2_l : np.ndarray 
            Possible values of first chemical potential 

        plane : str 
            Name of plane to apply the constraint on

        
        Returns 
        -------

        np.ndarray 
            Constrained values of first chemical potential for the given hyperplane

         np.ndarray 
            Constrained values of second chemical potential for the given hyperplane

        """
        updated_mu1, updated_mu2 = [], []
        vector_constraint = self.dic_plane[plane]['constraint'][1]
        for k in range(len(mu1_l)) : 
            if np.asarray([mu1_l[k], mu2_l[k]] + [1.0])@vector_constraint > 0.0 :
                updated_mu1.append(mu1_l[k])
                updated_mu2.append(mu2_l[k])
            else : 
                continue

        return np.asarray(updated_mu1), np.asarray(updated_mu2)

def ReadFileNormalVector(file : str) -> Tuple[Dict[str,np.ndarray], Dict[str,str], Dict[str,Dict[str,List[float]]], Dict[str,List[any]]] :
    """"Read the condensed log file for normal vector for all slab systems and extract all data to plot
    
    Parameters 
    ---------

    file : str 
        Path to the log file 

    Returns 
    -------

    dic_vect : Dict[str,np.ndarray]
        Dictionnary containing normal vector for all slab systems (keys are slab name)

    dic_mu : Dict[str,str]
        Dictionnary containing element to removed for all slab systems (keys are slab name)

    dic_bounds : Dict[str,Dict[str,List[float]]]
        Dictionnary containing all bounds data for all slab systems (keys are slab name)

    dic_constraint : Dict[str,List[List[str] | np.ndarray]]
        Dictionnary containing all constraints for all slab systems (keys are slab name)
    """
    dic_vect : Dict[str,np.ndarray] = {}
    dic_mu : Dict[str,str] = {} 
    dic_bounds : Dict[str,Dict[str,List[float]]] = {}
    dic_constraint : Dict[str,List[List[str] | np.ndarray]] = {}
    
    r = open(file,'r').readlines()
    for l in r :
        name_slab = l.split(':')[0][:-1]
        name_el = [el for el in l.split(':')[1].split()]
        vect = l.split(':')[2].split('==>')[0]
    
        tmp_bound = l.split('==>')[1][1:]
        for k in range(len(tmp_bound.split(':'))) : 
            if not name_slab in dic_bounds.keys() : 
                dic_bounds[name_slab] = {}
            dic_bounds[name_slab][tmp_bound.split(':')[k].split()[0]] = [float(tmp_bound.split(':')[k].split()[1]),float(tmp_bound.split(':')[k].split()[2])]

        tmp_constraint = l.split('==>')[2] 
        split_el = tmp_constraint.split(':')[0].split()
        domain_vector = np.array([float(val) for val in tmp_constraint.split(':')[1].split()])
        if not name_slab in dic_constraint.keys() : 
            dic_constraint[name_slab] = {}
        dic_constraint[name_slab] = [  split_el, domain_vector ]   
        

        dic_vect[name_slab] = np.array([float(el) for el in vect.split()])
        dic_mu[name_slab] = name_el

    return dic_vect, dic_mu, dic_bounds, dic_constraint


def PlotHyperplane(dic_vect : Dict[str,np.ndarray], dic_mu : Dict[str,str],dic_bounds : Dict[str,Dict[str,List[float]]], slab : str, color : str, ax : Axes3D) -> Axes3D :
    """Plot hyerplane for a given slab based on dictionnaries containing all data for all slab systems
    
    Parameters
    ----------

    dic_vect : Dict[str,np.ndarray]
        Dictionnary containing normal vector for all slab systems (keys are slab name)

    dic_mu : Dict[str,str]
        Dictionnary containing element to removed for all slab systems (keys are slab name)

    dic_bounds : Dict[str,Dict[str,List[float]]]
        Dictionnary containing all bounds data for all slab systems (keys are slab name)
    
    slab : str
        Name of the slab associated to the hyperplane to plot

    color : str
        Color of the hyperplane

    ax : Axes3D
        Matplotlib axe to update

    Returns 
    -------

    Axes3D
        Updated matplotlib axe

    """

    vect = dic_vect[slab]
    el = dic_mu[slab]

    mu1 = np.linspace(dic_bounds[slab][el[0]][0],dic_bounds[slab][el[0]][1],num=10)
    mu2 = np.linspace(dic_bounds[slab][el[1]][0],dic_bounds[slab][el[1]][1],num=10)
    MU1,MU2 = np.meshgrid(mu1,mu2)
    Z = (vect[3] + vect[0]*MU1 + vect[1]*MU2) / vect[2]
    surf = ax.plot_surface(MU1, MU2, Z,alpha=0.4, color=color , label=r'%s'%(slab), shade=False, rstride=50,cstride=50)
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
    
    return ax

def PlotAllHyperplanes3D(file_to_read : str, slab_to_plot : List[str]) -> None : 
    """Plot a subset of hyperplanes of all slab systems
    
    Parameters 
    ----------

    file_to_read : str 
        Path of log file containing all data for slab systems

    slab_to_plot : List[str]
        List of name of slab to plot

    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    dic_vect, dic_mu, dic_bounds, _ = ReadFileNormalVector(file_to_read)

    name = False
    
    Cmap1 = plt.get_cmap('rainbow')
    colors = [Cmap1(el) for el in np.linspace(0,1,num=len(dic_vect.keys()))]

    plot_list = [key for key in dic_vect.keys() if key in slab_to_plot]

    for id_color,slab in enumerate(plot_list) : 
        if not name : 
            name = slab
        color = colors[id_color]
        ax = PlotHyperplane(dic_vect,dic_mu,dic_bounds,slab,color,ax)

    plt.xlabel(r'$ \mu_{%s} - \mu_{%s}^{bulk}$ (eV)'%(dic_mu[name][0],dic_mu[name][0]),fontsize=12)
    plt.ylabel(r'$ \mu_{%s} - \mu_{%s}^{bulk}$ (eV)'%(dic_mu[name][1],dic_mu[name][1]),fontsize=12)
    ax.set_zlabel(r'$ \gamma (J/m^2$)',fontsize=12)
    ax.legend()
    #plt.savefig('3Dsurface.pdf')
    #plt.show()


"""Here is the abstractation part to find the nearest hyperplane to a collection of vectors"""
def ComputeDistanceToHyperplane(vect : np.ndarray, normal_vector : np.ndarray) -> float : 
    """Compute the energy associated to a given chemical potential vector a given hyperplane 
    
    Parameters 
    ----------

    vect : np.ndarray 
        Chemical potential vector (\mu_1 - \mu_1^{bulk}, \mu_2 - \mu_2^{bulk}, ..., \mu_(N-1) - \mu_(N-1)^{bulk})

    normal_vector : np.ndarray 
        Normal vector associated to a given hyperplane

    Returns 
    -------

    float 
        Surface energy associated to a given hyperplane for the chemical potential vector 

    """
    gamma_hyperplan = vect[:-1]@normal_vector[:-2] + normal_vector[-1]
    return abs(gamma_hyperplan)


def ComputeNearestHyperplane(list_slab : List[str], list_normal_vect : List[np.ndarray], mu_vect : np.ndarray) -> Tuple[str,float] :
    """"Find the hyperplane with the lowest surface energy for a given chemical potential vector among a list of slab
    
    Parameters
    ----------

    list_slab : List[str]
        List of name of slab 

    list_normal_vect : List[np.ndarray]
        List of associated normal vectors

    mu_vect : np.ndarray
        chemical potential vector 

    Returns 
    -------

    str 
        Name of the lowest surface energy slab

    float
        Surface energy of the slab    
    """
    
    min_distance = 10e4
    min_slab = None
    for id_slab, vect in enumerate(list_normal_vect) : 
        """gamma coordinate has to be the opposite !"""
        vect[-2] = -vect[-2]
        distance_id_slab = ComputeDistanceToHyperplane(mu_vect,vect)
        if distance_id_slab < min_distance : 
            min_distance = distance_id_slab 
            min_slab = list_slab[id_slab]

    return min_slab, min_distance 


def RecursiveMuGrid(list_vect : List[np.ndarray], list_id : List[int], dimension : int, coordinate : List[List[float]] = [], compt_dim : int=0) -> None : 
    """Build recursively the N-multidimensionnal chemical potential grid based on N-1D chemical potential vectors
    
    Parameters 
    ----------

    list_vect : List[np.ndarray]
        List of N-1D chemical potential vectors

    list_id : List[int]
        List of k id corresponding to each value in chemical potential vector

    dimension : int
        Number of dimension of the multidimensionnal grid to generate (N)

    coordinate : List[List[float]]
        List to increment during the procedure

    compt_dim : int
        Allows to initialise the procedure
    """ 
    if dimension == 0 :
        tmp_vect = np.zeros(len(list_vect))
        for k_dim,id in enumerate(list_id) : 
            tmp_vect[len(list_vect)-k_dim-1] += list_vect[len(list_vect)-k_dim-1][id]
        coordinate.append(tmp_vect) 
        return

    if compt_dim == 0 : 
        compt_dim += 1
        dimension += -1
        for id in range(len(list_vect[dimension])) :
            RecursiveMuGrid(list_vect,[id],dimension,coordinate=coordinate,compt_dim=compt_dim)

    else : 
        dimension += - 1
        for id in range(len(list_vect[dimension])) :
            RecursiveMuGrid(list_vect,list_id+[id],dimension,coordinate=coordinate,compt_dim=compt_dim)



class DictStability(TypedDict) : 
    compt : float
    stat : float
    

class MostStableSurface : 
    def __init__(self) : 
        self.dic : Dict[str,DictStability] = {}

    #########################################################
    def update_dic(self,slab : str) -> None : 
        """Updating dictionnary for slab counting
        
        Parameters 
        ----------

        slab : str
            Name of the slab system to update
        """
        
        if not slab in self.dic.keys() : 
            self.dic[slab] = {'compt':1.0,'stat':None}
        else : 
            self.dic[slab]['compt'] += 1.0

    #########################################################
    def sum_all(self) -> int : 
        """Sum everything !"""
        compt = 0 
        for slab in self.dic.keys() : 
            compt += self.dic[slab]['compt']

        return compt
    
    #########################################################
    def proba_stability(self) -> None : 
        """Compute all the stability probabilities on the N-mutlidimensionnal chemical potential grid
        P_S =  \frac{1}{K^N}  \sum_{1 \leq i_1, i_2, ..., i_N \leq K} \delta_S (mu_{i_1},mu_{i_2},..,mu_{i_N}) 

        """
        full_compt = self.sum_all()
        for slab in self.dic.keys() : 
            self.dic[slab]['stat'] = self.dic[slab]['compt']/full_compt
            print('Statistic of stability for slab : %s is %1.5f for the mu grid'%(slab,self.dic[slab]['stat']))


def RelativeStabilityOnGrid(dic_bounds : Dict[str,Dict[str,List[float]]], list_slab : List[str], dic_vect : Dict[str,np.ndarray], dic_constraint : Dict[str,List[List[str] | np.ndarray]],discr_per_mu : int = 100) -> None :
    """Compute the relative stability (in term of probabilities) between several slab system by taking into account constraints on chemical
    potential domains in N-multidimensionnal chemical potentials space

    Parameters
    ----------

    dic_bounds : Dict[str,Dict[str,List[float]]]
        Dictionnary containing all constraints on chemical potential domain for each slab

    list_slab : List[str]
        List of slab systems where the relative stability will be computed

    dic_vect : Dict[str,np.ndarray]
        Dictionnary containing all normal vectors associated to slab systems

    dic_constraint : Dict[str,List[List[str] | np.ndarray]] 
        Dictionnary containing all constraints on chemical potential domain for each slab

    discr_per_mu : int 
        Number of discretisation point on each dimension of the chemical potential grid
    """

    list_normal_vect = []
    for slab in list_slab : 
        list_normal_vect.append(dic_vect[slab])

    list_discret_vector = []
    for mu_el in dic_bounds[list_slab[0]].keys() :
        vector_bound_mu = np.linspace(dic_bounds[list_slab[0]][mu_el][0],dic_bounds[list_slab[0]][mu_el][1],num=discr_per_mu)
        list_discret_vector.append(vector_bound_mu)

    mu_grid = []
    RecursiveMuGrid(len(list_discret_vector),0,list_discret_vector,[],mu_grid)

    """the mu grid is now built :)"""
    stable_object = MostStableSurface()
    for coord_mu in mu_grid :
        mu_check_constraint = np.asarray([el for el in coord_mu] + [1.0])
        coord_mu = np.asarray([el for el in coord_mu] + [0.0])
        vector_constraint = dic_constraint[list_slab[0]][1]
        if mu_check_constraint@vector_constraint < 0.0 : 
            continue 
        else :
            nearest_plane, _ = ComputeNearestHyperplane(list_slab,list_normal_vect,coord_mu) 
            stable_object.update_dic(nearest_plane)
    
    stable_object.proba_stability() 


def RelativeStabilityOnGrid2DPlot(dic_bounds : Dict[str,Dict[str,List[float]]], list_slab : List[str], dic_vect : Dict[str,np.ndarray], dic_constraint : Dict[str,List[List[str] | np.ndarray]],discr_per_mu : int = 100) -> Tuple[np.ndarray,np.ndarray,np.ndarray,IntersectionObject] :
    """Compute the relative stability (in term of probabilities) between several slab system by taking into account constraints on chemical
    potential domains 2D chemical potentials space

    Parameters
    ----------

    dic_bounds : Dict[str,Dict[str,List[float]]]
        Dictionnary containing all constraints on chemical potential domain for each slab

    list_slab : List[str]
        List of slab systems where the relative stability will be computed

    dic_vect : Dict[str,np.ndarray]
        Dictionnary containing all normal vectors associated to slab systems

    dic_constraint : Dict[str,List[List[str] | np.ndarray]] 
        Dictionnary containing all constraints on chemical potential domain for each slab

    discr_per_mu : int 
        Number of discretisation point on each dimension of the chemical potential grid

    Returns 
    -------

    np.ndarray 
        2D array grid containing surface energy

    np.ndarray
        2D array grid corresponding to the first chemical potential vector

    np.ndarray
        2D array grid corresponding to the second chemical potential vector
      
    IntersectionObject 
        Filled IntersectionObject

    """
    list_normal_vect = []
    for slab in list_slab : 
        list_normal_vect.append(dic_vect[slab])

    list_discret_vector = []
    for mu_el in dic_bounds[list_slab[0]].keys() :
        vector_bound_mu = np.linspace(dic_bounds[list_slab[0]][mu_el][0],dic_bounds[list_slab[0]][mu_el][1],num=discr_per_mu)
        list_discret_vector.append(vector_bound_mu)
    
    grid_gamma = np.zeros((discr_per_mu,discr_per_mu))
    MU_1, MU_2 = np.meshgrid(list_discret_vector[0],list_discret_vector[1])


    """the mu grid is now built :)"""
    nearest_plane_list = []

    for id1 in range(discr_per_mu) :
        for id2 in range(discr_per_mu) :
            coord_mu = np.asarray([MU_1[id1,id2], MU_2[id1,id2]] + [0.0])

            vector_constraint = dic_constraint[list_slab[0]][1]
            if np.asarray([MU_1[id1,id2], MU_2[id1,id2]] + [1.0])@vector_constraint < 0.0 : 
                grid_gamma[id1,id2] = None

            else : 
                nearest_plane, distance = ComputeNearestHyperplane(list_slab,list_normal_vect,coord_mu) 
                grid_gamma[id1,id2] = distance
                nearest_plane_list.append([nearest_plane,[id1,id2]])

    """building od different slab list stabilty"""
    different_slab = []
    for slab in nearest_plane_list :
        if slab[0] not in different_slab : 
            different_slab.append(slab[0])

    """building intersection object"""
    inter_object = IntersectionObject(MU_1,MU_2,different_slab)
    inter_object.build_plane_intersection_dictionnary(nearest_plane_list,dic_vect,dic_constraint,method='least_square')
    inter_object.check_generator_line()
    return grid_gamma, MU_1, MU_2, inter_object


def PlotProjectionHyperplaneInto2D(file_to_read : str, slab_to_plot : List[str], grid_mu : int, level_line : bool = False) -> None : 
    """Plot relative stability domains for different slab model for 2D chemical potential case
    
    Parameters
    ----------

    file_to_read : str
        Path to the normal vector log file to read 

    slab_to_plot : List[str]
        List of slab systems to plot

    grid_mu : int 
        Number of discretisation point on each dimension to plot the 2D grid  

    level_line : bool 
        Plot the level line for surface energy (is set to False by default)

    """
    Cmap1 = plt.get_cmap('viridis')
    fig, ax = plt.subplots()

    dic_vect, dic_mu, dic_bounds, dic_constraint = ReadFileNormalVector(file_to_read)
    name = [key for key in dic_mu.keys()][0]

    gamma, mu1, mu2, inter_object = RelativeStabilityOnGrid2DPlot(dic_bounds,slab_to_plot,dic_vect,dic_constraint,grid_mu)
    
    gamma = np.ma.masked_invalid(gamma)
    ax.patch.set(color='grey', edgecolor='grey',zorder=1000)

    if level_line : 
        CS = plt.contour(mu1,mu2,gamma,5,colors='white')
        plt.clabel(CS, inline=0.2, fontsize=7)
    
    plt.contourf(mu1,mu2,gamma,1000,cmap=Cmap1)

    if not inter_object.dic_intersection['empty'] : 
        for key_inter in inter_object.dic_intersection.keys() : 
            if key_inter != 'empty' : 
                intersection_point = inter_object.dic_intersection[key_inter]['intersection_point']
                line2D = inter_object.dic_intersection[key_inter]['inter_2D']
                if intersection_point is not None :
                    mu2_line0 = intersection_point[1] - line2D[0]/line2D[1]*intersection_point[0]
                    mu1_line = mu1[0,:]
                    mu2_line = mu2_line0 + line2D[0]/line2D[1]*mu1_line
                    mu1_line_patch, mu2_line_patch = inter_object.apply_constraint(mu1_line,mu2_line,key_inter.split('@')[0])
                    plt.plot(mu1_line_patch,mu2_line_patch,linestyle='dashed',color='black',zorder=10)
                else : 
                    print('I have troube to find an intersection point (%s) for relativty stabiltiy :('%(key_inter))


    for key_plane in inter_object.dic_plane.keys() : 
        bary_plane = inter_object.dic_plane[key_plane]['bary']
        name_slab = r'$\mathcal{M}_{[%s]}^{%s}$'%(key_plane.split('_')[0],key_plane[-1])
        plt.text(bary_plane[0],bary_plane[1],name_slab,fontsize=14,horizontalalignment='center',verticalalignment='center')

    #plt.gca().patch.set(color='grey',edgecolor='grey',zorder=1000)
    plt.xlim(np.amin(mu1),np.amax(mu1))
    plt.ylim(np.amin(mu2),np.amax(mu2))
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    ax = plt.colorbar(ticks=[el for el in np.linspace(np.amin(gamma),np.amax(gamma),num=5)],format='%2.1f')
    ax.set_label(r'$\gamma$ (J/m$^2$)',rotation=270,fontsize=15,labelpad=20)


    plt.xlabel(r'$ \mu_{%s} - \mu_{%s}^{bulk}$ (eV)'%(dic_mu[name][0],dic_mu[name][0]),fontsize=12)
    plt.ylabel(r'$ \mu_{%s} - \mu_{%s}^{bulk}$ (eV)'%(dic_mu[name][1],dic_mu[name][1]),fontsize=12)
    plt.tight_layout()
    plt.savefig('2D_plot_stability.png',dpi=500)
