from operator import inv
import numpy as np
import os, sys
import matplotlib
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from ase import io
from ase.visualize import view
from scipy.spatial.transform import Rotation as R
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


#######################################################################
## imputs time :) 
#######################################################################
path='/home/lapointe/Work_ML/Heusler/surface_builder/best_model_CeRuSN'
image_path = '/home/lapointe/Work_ML/Heusler/surface_builder/image_CeRuSn'
mode = 'given_angle'
nb_rot = 10
number_fig_max = 2
replicate = [1,1,1]
list_angle=  [0.0,0.0,0.0]  #[0.0,0.0,0.0] #[0.0,20.0,110.0]
########################################################################

dic_color = {'C':'grey','H':'white','Co':'royalblue','Fe':'firebrick','Ga':'lightcoral','Ce':'palegoldenrod','Sn':'slategrey','Ru':'seagreen','Au':'gold'}
dic_size = {'C':6,'H':1,'Co':27,'Fe':26,'Ga':31,'Ce':58,'Sn':50,'Ru':44,'Au':5}

########################################################
### check difference of size between atom
########################################################
def check_difference_size(list_size):
    ratio = (max(list_size) - min(list_size))/max(list_size)
    if ratio < 0.3 : 
        return True, 1.0/ratio
    else : 
        return False, ratio
########################################################


########################################################
### plot untis cells
########################################################
def plot_unit_cell(axis,system,projections_axis,last_axis): 
    cell = system.get_cell()[:]
    list_proj = []
    for k in range(cell.shape[0]):
        proj_axis_k1 = np.dot(cell[k,:],projections_axis[0])
        proj_axis_k2 = np.dot(cell[k,:],projections_axis[1])
        proj_axis_k3 = np.dot(cell[k,:],last_axis)
        list_proj.append([proj_axis_k1,proj_axis_k2,proj_axis_k3])

    # plot time
    # projection of all axis of the cell
    for k in range(len(list_proj)) : 
        axis.plot([0.0,list_proj[k][0]],[0.0,list_proj[k][1]],color='black',linestyle='dashed')
    
    # build the translations
    for i in range(len(list_proj)) : 
        for j in range(len(list_proj)) : 
            if i == j : 
                continue 
            else : 
                axis.plot([list_proj[i][0],list_proj[i][0]+list_proj[j][0]],[list_proj[i][1],list_proj[i][1]+list_proj[j][1]],color='black',linestyle='dashed',zorder=-50)
                for k in range(len(list_proj)) : 
                    if j == k or i == k : 
                        continue 
        
                    else : 
                        axis.plot([list_proj[i][0]+list_proj[j][0],list_proj[i][0]+list_proj[j][0]+list_proj[k][0]],[list_proj[i][1]+list_proj[j][1],list_proj[i][1]+list_proj[j][1]+list_proj[k][1]],color='black',linestyle='dashed',zorder=-50)
            

    return axis
########################################################


########################################################
# plot repaire
########################################################
def plot_repaire(axis,system,projections_axis,last_axis,raduis,size_arrow):
    
    # rescaling de la taille des fleches pour le repaire
    positions = system.get_positions()
    scalaire1 = [np.dot(el,projections_axis[0]) for el in positions]
    scalaire2 = [np.dot(el,projections_axis[1]) for el in positions]
    scale = 5.0

    axis.arrow(min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis,size_arrow,0.0,width=0.1,color='white',ec='black')
    axis.text(min(scalaire1)-scale*raduis+size_arrow+1.0,min(scalaire2)-scale*raduis,r'$[%1d%1d%1d]$'%(projections_axis[0][0],projections_axis[0][1],projections_axis[0][2]),fontsize=12,horizontalalignment='left',verticalalignment='center')

    axis.arrow(min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis, 0.0, size_arrow,width=0.1,color='white',ec='black')
    axis.text(min(scalaire1)-scale*raduis,min(scalaire2)-scale*raduis+size_arrow+1.0,r'$[%1d%1d%1d]$'%(projections_axis[1][0],projections_axis[1][1],projections_axis[1][2]),fontsize=12,horizontalalignment='center')

    circle = plt.Circle((min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis), 0.3, ec='black' ,color='white',zorder=10)
    axis.add_patch(circle) 
    axis.scatter(min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis,1.0,color='black',zorder=11)
    axis.text(min(scalaire1)-scale*raduis,min(scalaire2)-scale*raduis-1.0,r'$[%1d%1d%1d]    $'%(last_axis[0],last_axis[1],last_axis[2]),fontsize=12,horizontalalignment='center',verticalalignment='center')
    return axis


########################################################
# plot name 
########################################################
def plot_name_model(axis,system,projections_axis,size,name):
    miller = name.split('_')[0]
    nb_model = name.split('.')[0][-1]

    positions = system.get_positions()
    scalaire1 = [np.dot(el,projections_axis[0]) for el in positions]
    scalaire2 = [np.dot(el,projections_axis[1]) for el in positions]   
    axis.text(0.5*(min(scalaire1)+max(scalaire1)) , 1.3*max(scalaire2)+2.0, r'$\mathcal{M}^{%s}_{[%s%s%s]}$'%(nb_model,miller[0],miller[1],miller[2]),zorder=20,fontsize=size,horizontalalignment='center',verticalalignment='center')
    
    return axis


########################################################
# plot name 
########################################################
def plot_name_layer(axis,system,projections_axis,size):

    positions = system.get_positions()
    scalaire1 = [np.dot(el,projections_axis[0]) for el in positions]
    scalaire2 = [np.dot(el,projections_axis[1]) for el in positions]
    increment = 1.4
    dic_str = {0:'I',1:'II',2:'III',3:'IV'}
    for k in range(4) : 
        axis.text(1.7*max(scalaire1), max(scalaire2)- increment*k, r'%s'%(dic_str[k]),zorder=20,fontsize=size,horizontalalignment='center',verticalalignment='center')
    
    return axis


########################################################
# plot repaire
########################################################
def plot_repaire_proj(axis,system,projections_axis,last_axis,raduis,size_arrow):
    
    # rescaling de la taille des fleches pour le repaire
    positions = system.get_positions()
    scalaire1 = [np.dot(el,projections_axis[0]) for el in positions]
    scalaire2 = [np.dot(el,projections_axis[1]) for el in positions]
    scale, scale2 = 5.0, 2.5
    
    original_repair = [np.array([1.0,0.0,0.0]),np.array([0.0,1.0,0.0]),np.array([0.0,0.0,1.0])]
    for id, vect in enumerate(original_repair) : 
        if (abs(np.dot(vect,projections_axis[0])) + abs(np.dot(vect,projections_axis[1]))) > 1e-2 :   
            axis.arrow(min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis,size_arrow*np.dot(vect,projections_axis[0]),size_arrow*np.dot(vect,projections_axis[1]),width=0.1,color='white',ec='black')
            axis.text(min(scalaire1)-scale*raduis + scale2*size_arrow*np.dot(vect,projections_axis[0]),min(scalaire2)-scale*raduis + scale2*size_arrow*np.dot(vect,projections_axis[1]),r'$\mathbf{e}_{%s}$'%(str(id+1)),fontsize=12,horizontalalignment='center',verticalalignment='center')
        else : 
            circle = plt.Circle((min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis), 0.3, ec='black' ,color='white',zorder=10)
            axis.add_patch(circle) 
            axis.scatter(min(scalaire1)-scale*raduis, min(scalaire2)-scale*raduis,1.0,color='black',zorder=11)
            axis.text(min(scalaire1)-scale*raduis - 1.0 ,min(scalaire2)-scale*raduis,r'$\mathbf{e}_{%s}$'%(str(id+1)),fontsize=12,horizontalalignment='center',verticalalignment='center')

    return axis


########################################################
### plot atoms
########################################################
def plot_atoms(axis,system,raduis,projections_axis,last_axis,label=True):
    #max_size = max([dic_size[el] for el in dic_size.keys()])
    max_size = max([dic_size[el] for el in system.get_chemical_symbols()])
    bool_size, puissance = check_difference_size([dic_size[el] for el in system.get_chemical_symbols()])
    scale1, scale2 = 5.0, 5.0 

    positions = system.get_positions()
    scalaire1 = [np.dot(el,projections_axis[0]) for el in positions]
    scalaire2 = [np.dot(el,projections_axis[1]) for el in positions]
    axis.set_xlim((  min(scalaire1) - scale1*raduis ,  max(scalaire1) + scale1*raduis ))
    axis.set_ylim((  min(scalaire2) - scale2*raduis, max(scalaire2) + scale2*raduis ))

    for id, pos in enumerate(positions) : 
        proj1 = np.dot(pos,projections_axis[0])
        proj2 = np.dot(pos,projections_axis[1])
        order = np.dot(pos,last_axis)

        raduis_min = raduis/3.0
        if not bool_size : 
            scaling = dic_size[system.get_chemical_symbols()[id]]/(max_size) 
        if bool_size :
            scaling = (dic_size[system.get_chemical_symbols()[id]]**puissance)/(max_size**puissance)

        circle_k = plt.Circle((proj1, proj2), raduis_min + scaling*(raduis-raduis_min), ec='black' ,color=dic_color[system.get_chemical_symbols()[id]],zorder=order)
        axis.add_patch(circle_k)
        if label : 
            axis.text(proj1,proj2,r'$\mathcal{%s}$'%(system.get_chemical_symbols()[id]),zorder=order+0.1,fontsize=10,horizontalalignment='center',verticalalignment='center')
    axis.set_aspect('equal', adjustable='datalim')

    return axis
########################################################

number_fig = len(os.listdir(path))
#size = (number_fig*6,6)
if number_fig%number_fig_max > 0 : 
    reste = 1
else : 
    reste = 0
size = (6*number_fig_max,(number_fig//number_fig_max + reste)*6)

if not os.path.exists(image_path): 
    os.mkdir(image_path)

if mode == 'screening' : 
    for theta in np.linspace(0.0,90.0,num=nb_rot) : 
        for phi in np.linspace(0.0,90.0,num=nb_rot) :
            for psi in np.linspace(0.0,90.0,num=nb_rot) :
                fig, axis = plt.subplots(nrows=1, ncols=number_fig, sharex=True, figsize=size)
                r = R.from_euler('zyx', [theta, phi, psi], degrees=True)
                r = r.as_matrix()

                for id, f in enumerate(os.listdir(path)) :
                    full_path = '%s/%s'%(path,f) 
                    system = io.read(full_path)

                    #ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
                    axis[id] = plot_atoms(axis[id],system,1.0,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]),label=True)
                    axis[id] = plot_unit_cell(axis[id],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]))
                    axis[id] = plot_repaire_proj(axis[id],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],r@np.array([0.0,0.0,1.0]),1.0,1.0)
                    axis[id].set_axis_off()
    

                fig.tight_layout()
                plt.savefig('%s/rendu_%s_%s_%s.png'%(image_path,str(theta),str(phi),str(psi)))

if mode == 'given_angle' :
    if number_fig <= number_fig_max :
        fig, axis = plt.subplots(nrows=1, ncols=number_fig, sharex=True, figsize=size)
        r = R.from_euler('zyx', list_angle, degrees=True)
        r = r.as_matrix()
        list_path = os.listdir(path) 
        list_path.sort()
        for id, f in enumerate(list_path) :
            id_row = id//number_fig
            full_path = '%s/%s'%(path,f) 
            system = io.read(full_path)
            system = system*replicate
            if number_fig == 1 : 
                axis = plot_atoms(axis,system,2.0,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]),label=True)
                axis = plot_unit_cell(axis,system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]))
                axis = plot_repaire_proj(axis,system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],r@np.array([0.0,0.0,1.0]),1.0,1.0)
                axis = plot_name_layer(axis,system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],20)
                #axis = plot_name_model(axis,system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],20,f)
                
                axis.set_axis_off()                

            else : 
                axis[id] = plot_atoms(axis[id],system,2.0,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]),label=True)
                axis[id] = plot_unit_cell(axis[id],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]))
                axis[id] = plot_repaire_proj(axis[id],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],r@np.array([0.0,0.0,1.0]),1.0,1.0)
                axis[id] = plot_name_model(axis[id],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],20,f)
                axis[id].set_axis_off()

    else : 
        fig, axis = plt.subplots(nrows=number_fig//number_fig_max + reste, ncols=number_fig_max, sharex=True, figsize=size)
        r = R.from_euler('zyx', list_angle, degrees=True)
        r = r.as_matrix()
        list_path = os.listdir(path) 
        list_path.sort()
        for id, f in enumerate(list_path) :
            id_row = id//number_fig_max
            id_colum = id%number_fig_max
            full_path = '%s/%s'%(path,f) 
            system = io.read(full_path)
            system = system*replicate
            axis[id_row,id_colum] = plot_atoms(axis[id_row,id_colum],system,2.0,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]),label=True)
            axis[id_row,id_colum] = plot_unit_cell(axis[id_row,id_colum],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])], r@np.array([0.0,0.0,1.0]))
            axis[id_row,id_colum] = plot_repaire_proj(axis[id_row,id_colum],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],r@np.array([0.0,0.0,1.0]),1.0,1.0)
            axis[id_row,id_colum] = plot_name_model(axis[id_row,id_colum],system,[r@np.array([1.0,0.0,0.0]),r@np.array([0.0,1.0,0.0])],20,f)
            axis[id_row,id_colum].set_axis_off()

    if reste == 1 : 
        last_id_row = number_fig//number_fig_max + reste - 1
        last_id_column = number_fig%number_fig_max - 1
        for k in range(last_id_column + 1,number_fig_max) : 
            axis[last_id_row,k].set_axis_off()

    fig.tight_layout()
    plt.savefig('%s/model_CeRuSn.png'%(image_path),dpi=500)
    #plt.savefig('%s/rendu.png'%(image_path),dpi=500)
    plt.show()

#plt.close()
