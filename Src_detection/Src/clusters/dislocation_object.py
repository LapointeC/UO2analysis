import warnings
from typing import Dict, List, Tuple, TypedDict

from ase import Atoms, Atom
from .cluster import ClusterDislo, LocalLine

import numpy as np
from scipy.spatial import ConvexHull
from ..tools import get_N_neighbour

class reference_structure(TypedDict) :
    """TypedDict class used to compute the Nye tensor"""
    structure : str
    unit_cell : np.ndarray

class DislocationObject : 
    """DislocationObject class used to find and compute properties for a given subset of Atoms systems
    Main properties which are globally computed
    
    - (i) the Nye tensor 
    - (ii) the dictionnary dislocation ```Dict[str,ClusterDislo]```.

    Each key of dislocation dictionnary contains ```ClusterDislo``` object which allows to compute local properties associated to each ```Cluster```
    The local properties are stored in ```LocalLine``` object which contains
    
    - (i) ```local_normal``` 
    - (ii) ```local_burger``` (computed with Nye tensor)
    
    ```local_normal``` and ```local_burger``` allow to reconstruct dislocation properties ...
     
    """
    def __init__(self, system : Atoms, 
                extended_system : Atoms, 
                full_system : Atoms, 
                rcut : float, 
                structure : reference_structure = {'structure':'bcc',
                                                   'unit_cell':3.1855*np.eye(3)}) -> None : 
        """Init method for ```DislocationObject```. Nye tensor calculation needs to used two buffer region for the system
        which are called ```extended_system``` and ```full_system```. Rigourisly only ```extended_system``` is needed to compute deformation
        tensor and ```full_system``` is need to evaluate gradient of the deformation tensor needed to estimate Nye tensor !

        Parameters
        ----------

        system : Atoms 
            System to compute the Dislocation properties

        extended_system : Atoms 
            First buffer system needed to evaluate Nye tensor

        full_system : Atoms 
            Second buffer system needed to evaluate deformation tensor

        rcut : float 
            Cutoff raduis to compute neightborhood (in AA)

        structure : reference_structure
            Dictionnary needed to build the deformation tensor based on reference lattice. 
            Two key are needed in this dictionnary : (i) ```structure``` the cristallographic reference structure and (ii) ```unit_cell``` unit cell of the cristallographic reference 

        """

        self.system = system
        self.extended_system = extended_system
        self.full_system = full_system
        self.rcut = rcut
        self.structure = structure
        self.convex_hull = self.get_convex_hull()
        #self.OrganizeSystem()

        self.dislocations : Dict[int,ClusterDislo] = {}
        self.starting_point_line = None

        # reference cristallographic neighbours vector (unit coordinate) for traditionnal lattices 
        self.p_vectors_fcc = np.array([[ 0.5, 0.5, 0.0], [ 0.5, 0.0, 0.5], [ 0.0, 0.5, 0.5],
                                     [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5], [ 0.0,-0.5, 0.5],
                                     [ 0.5,-0.5, 0.0], [ 0.5, 0.0,-0.5], [ 0.0, 0.5,-0.5],
                                     [-0.5,-0.5, 0.0], [-0.5, 0.0,-0.5], [ 0.0,-0.5,-0.5]])

        self.p_vector_bcc = np.array([ [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],  [0.5, -0.5, 0.5], [0.5, 0.5, 0.5],
                                  [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5],  [0.5, -0.5, -0.5], [0.5, 0.5, -0.5],
                                  [0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [-1.,  0.,  0.], [ 0., -1.,  0.], [ 0.,  0., -1.]])

        self.p_vector_hcp = np.array([ [-0.57735027, 0.0, 0.815], [-5.77350269e-01, 0.0, -8.15000000e-01], [0.28867513, 0.5 , 0.815 ], [ 0.28867513, -0.5 , 0.815 ], 
                                  [ 0.28867513,  0.5 , -0.815 ], [ 0.28867513, -0.5 , -0.815 ], [-0.8660254,  0.5 , 0.0] , [-0.8660254, -0.5 ,  0.0], [ 0., -1.,  0.0],
                                  [0.8660254, 0.5 , 0.0], [ 0.8660254, -0.5 ,  0.0], [-0.57735027, 1.0 ,  0.815 ], [-0.57735027,  1.0 , -0.815 ], [ 1.15470054, 0.0 , -0.815 ], 
                                  [1.15470054, 0.0, 0.815]  ])

    def OrganizeSystem(self) -> None : 
        """Build the starting point local dislocation line for each ```ClusterDislo```. 
        Starting point of each cluster is chosen to be the farest of the center of the cluster"""
        barycenter = np.mean(self.system.positions, axis =0)
        composite_id_distance = [(idx, np.linalg.norm(pos - barycenter)) for idx,pos in enumerate(self.system.positions) ]
        sorting =  sorted(composite_id_distance, key=lambda x: x[1], reverse=True)
        self.system : Atoms = self.system[ [s[0] for s in sorting ] ]
        return  

    def get_convex_hull(self) -> ConvexHull :
        """Build the 3D convex hull for the whole system 
        
        Returns:
        --------

        ConvexHull
            Associated ConvexHull object
        """
        return ConvexHull(self.system.positions)

    def BuildSamplingLine(self, rcut_line : float = 3.5, 
                          rcut_cluster : float = 4.5, 
                          scale_cluster : float = 1.5) -> List[int] : 
        
        """Build a sample line to generate the dislocation line. In order to build an efficient an accurate line, 
        selected atoms have to be sufficiently far to the previous one
        
        Parameters
        ----------

        rcut_line : float 
            Rcut in to be added in a ```ClusterDislo```

        rcut_cluster : float 
            Initial size of the each ```ClusterDislo``` (see the documentation)

        scale_custer : float 
            Scale parameter for  ```ClusterDislo``` (see the documentation)

        Returns:
        --------

        List[int]
            Indexes of atoms in the sample line (used for debug)
        """
        if rcut_cluster < rcut_line : 
            raise ValueError(f'Impossible to build lines with rcut_cluster {rcut_cluster} < rcut_line {rcut_line}')

        #here is the initialisation step. We first select the farest atom to the system barycenter
        idx_dislo = 0
        barycenter = np.mean(self.system.positions, axis=0)
        starting_id = np.argmax( np.linalg.norm(self.system.positions - barycenter, axis=1)  )
        
        
        self.starting_point_line = starting_id
        self.dislocations[idx_dislo]  = ClusterDislo(LocalLine( self.system[starting_id].position, self.system[starting_id].symbol ), starting_id, rcut_cluster)  

        #debug
        tmp_list = [starting_id]

        for id_atom_i, at_i in enumerate(self.system) : 
            if id_atom_i == starting_id : 
                continue
            else : 
                #Find the nearest ClusterDislo to the at_i
                min_dist_array = [ np.amin(np.linalg.norm(at_i.position - cluster.atoms_dfct.positions, axis=1)) for _,cluster in self.dislocations.items()]
                index_min_key = np.argmin(min_dist_array)
                closest_cluster = list(self.dislocations.values())[index_min_key]

                #at_i have to be sufficiently far from the cluster to build an accurate dislocation line 
                if min_dist_array[index_min_key] > rcut_line :
                    tmp_list.append(id_atom_i)
                    if closest_cluster.get_elliptic_distance(at_i) > scale_cluster :
                            idx_dislo += 1 
                            self.dislocations[idx_dislo]  = ClusterDislo(LocalLine( at_i.position, at_i.symbol ), idx_dislo, rcut_cluster)                        
                    
                    #otherwise we aggregate at_i to the closest ClusterDislo
                    else : 
                        closest_cluster.append_dislo(LocalLine(at_i.position, at_i.symbol), id_atom_i)

                else : 
                    continue 
        
        return tmp_list

    def AggregateTwoClusterDislo(self, slave_cluster : ClusterDislo, master_cluster : ClusterDislo) -> ClusterDislo :
        """Aggregate two ```ClusterDislo``` in one 
        
        Parameters
        ----------

        slave_cluster : ```ClusterDislo```     
            ```ClusterDislo``` to delete

        master_cluster : ```ClusterDislo```
            Aggregation ```ClusterDislo```

        Returns
        -------

        ```ClusterDislo```
            Master ```ClusterDislo``` after aggregation

        """
        for id_s, l_s in slave_cluster.local_lines.items() :
            master_cluster.append_dislo(l_s, id_s)
        return master_cluster

    def RefineSamplingLine(self, scale : float = 2.0) -> None : 
        """Aggregate the closest ```ClusterDislo``` before build the ordering line
        
        Parameters
        ----------

        scale : float
            Scaling factor to consider if two ```ClusterDislo``` are overlaped

        """
        
        dic_aggre : Dict[int, List[int]] = {}
        for id_i, cluster_i in self.dislocations.items() : 
            for id_j, cluster_j in self.dislocations.items() : 
                if id_i > id_j :
                    r_ij = np.linalg.norm(cluster_i.center - cluster_j.center)
                    size_ij = cluster_i.size + cluster_j.size
                    
                    # Aggregation between the two clusters !
                    if r_ij < scale*size_ij :
                        if id_i in dic_aggre.keys() : 
                            if id_j not in dic_aggre[id_i] :
                                dic_aggre[id_i].append(id_j)
                        elif id_j in dic_aggre.keys() : 
                            if id_i not in dic_aggre[id_j] :
                                dic_aggre[id_j].append(id_i)

                        else :
                            
                            # check if we have to add a new key
                            bool_check = True
                            for _, val in dic_aggre.items() : 
                                if (id_j in val) and (id_i not in val) : 
                                    val.append(id_i)
                                    bool_check = False 
                                    break
                                elif (id_i in val) and (id_j not in val) :
                                    val.append(id_j)
                                    bool_check = False
                                    break

                            if bool_check :
                                dic_aggre[id_i] = [id_j]
                
                else :
                    continue
        
        #print(dic_aggre)
        # reaggreate ...
        for key_id, list_to_aggr in dic_aggre.items() : 
            for id_to_aggr in list_to_aggr : 
                try : 
                    cluster_master = self.dislocations[key_id]
                    cluster_slave = self.dislocations[id_to_aggr]

                    # aggregate on master
                    self.AggregateTwoClusterDislo(cluster_slave, cluster_master)

                    # del old key
                    del self.dislocations[id_to_aggr]
                except : 
                    warnings.warn(f'Something went wrong for {key_id} (master) / {id_to_aggr} (slave) aggregation')

        return 

    def StartingPointCluster(self) -> None :
        """Build the starting point local dislocation line for each ```ClusterDislo```. 
        Starting point of each cluster is chosen to be the farest of the center of the cluster"""
        for _, cluster in self.dislocations.items(): 
            composite_id_distance = [(idx, np.linalg.norm(line.center - cluster.center)) for idx, line in cluster.local_lines.items() ]
            max_id, _ =  sorted(composite_id_distance, key=lambda x: x[1], reverse=True)[0]
            cluster.starting_point = max_id
        return 


    def BuildOrderingLine(self, neighbour : np.ndarray, 
                          descriptor : np.ndarray = None, 
                          idx_neighbor : np.ndarray = None) -> List[Atoms] :
        """This method allows to order the ```LocalLine``` by explicitely construct the dislocation line. 
        The dislocation line is so computed at each  ```LocalLine``` object. If descriptor are not None, they are used 
        to compute realistic barycenter for the dislocation : 
            
            $q_{local_line,i} = \sum_{j \in neigh(i) q_{local_line,j}} \frac{e^{\Vert D_{local_line,j} \Vert}}{ sum_{j' \in neigh(i) q_{local_line,j'} e^{\Vert D_{local_line,j'} \Vert}}  }  $


        Parameters
        ----------

        neighour : np.ndarray 
            array of neighbours for EXTENDED_SYSTEM (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)
        
        descriptor : np.ndarray
            descriptor vector associated to the FULL_SYSTEM

        idx_neighbor : np.ndarray 
            array of index of neighbours for EXTENDED_ATOMS SYSTEM (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}
        
        Returns:
        --------

        Atoms 
            Coordinate of the whole point part of the line (used for debug)
        """
        
        barycenter = []
        for _, cluster in self.dislocations.items():
            barycenter_loc = Atoms()
            weights = np.ones(neighbour.shape[1])
            if descriptor is not None :
                subset_descriptor = descriptor[idx_neighbor[self.starting_point_line] , :]
                weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])

            starting_point = np.average( neighbour[cluster.starting_point,:,:] + cluster.local_lines[cluster.starting_point].center, axis=0, weights=weights)
            
            #debug 
            barycenter_loc.append(Atom('W',starting_point))

            # updating data in local lines for the starting point!
            cluster.local_lines[cluster.starting_point].update_center(starting_point)
            explored_idx = [cluster.starting_point]
            cluster.order_line = [cluster.starting_point]

            #starting_point_idx = cluster.starting_point

            for _ in range(len(cluster.local_lines) - 1 ) : 
                #build the sub_local_line dictionnary which excluded the previously explored points
                sub_local_line = {key:val for key, val in cluster.local_lines.items() if key not in explored_idx}
                
                #composite_id_distance = [(idx, np.amin(np.linalg.norm(line.center - starting_point - neighbour[starting_point_idx,:,:], axis=1))) for idx, line in sub_local_line.items() 
                #                         if np.linalg.norm(line.center - starting_point) > 0.0 ]                

                composite_id_distance = [(idx, np.linalg.norm(line.center - starting_point)) for idx, line in sub_local_line.items() if np.linalg.norm(line.center - starting_point) > 0.0 ]
                min_id, _ =  sorted(composite_id_distance, key=lambda x: x[1])[0]

                if descriptor is not None : 
                    subset_descriptor = descriptor[idx_neighbor[min_id] , :]
                    weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])

                #find the next previous point of the local line
                explored_idx.append(min_id)
                cluster.order_line.append(min_id)
                # build the new starting point for the iterative skim...
                new_starting_point = np.average( neighbour[min_id,:,:] + cluster.local_lines[min_id].center, axis=0, weights=weights)
                new_starting_point_idx = min_id

                #debug
                barycenter_loc.append(Atom('W',new_starting_point))

                #updating the local normal from the previous starting point and next id 
                cluster.local_lines[explored_idx[-1]].update_normal( (new_starting_point - starting_point)/np.linalg.norm(new_starting_point - starting_point) )
                cluster.local_lines[explored_idx[-1]].update_norm_normal( np.linalg.norm(new_starting_point - starting_point) )
                cluster.local_lines[explored_idx[-1]].update_next( min_id )
                
                #updating the new starting point ! 
                starting_point = new_starting_point
                
                #starting_point_idx = new_starting_point_idx
            
            barycenter.append(barycenter_loc)

        return barycenter


    def LineSmoothing(self, nb_averaging_window : int = 2) -> List[Atoms] : 
        """Allows to smooth the dislocation line based on BuildOrderingLine method. 
        The previous dislocation line is smoothed by averaging window (in cartesian space) with other point which are part of the previous line
        This method create a new ```ClusterDislo``` attribute called ```smooth_local_lines``` which have the same structure than ```local_lines```


        Parameters
        ----------

        nb_averaging_window : int
            Size of the averaging window 
        
        Returns:
        --------

        Atoms 
            Coordinate of the whole point part of the smooth line (used for debug)
        """
        
        barycenter = []
        for _, cluster in self.dislocations.items():
            
            # debug
            barycenter_loc = Atoms()

            #build the smoothed local line !INDEXES ARE CHANGED AT THIS POINT!
            for k in range(len(cluster.order_line) - nb_averaging_window) : 
                species = cluster.local_lines[cluster.order_line[k]].species
                all_window_positions = np.array([ cluster.local_lines[cluster.order_line[i]].center for i in range(k,k+nb_averaging_window) ])
                av_localline_obj = LocalLine( np.mean(all_window_positions, axis=0), species)
                
                #debug
                barycenter_loc.append(Atom( symbol=species, position= np.mean(all_window_positions, axis=0)))
                #smooth line is updated
                cluster.smooth_local_lines[k] = av_localline_obj
                cluster.smooth_order_line.append( cluster.order_line[int(0.5*(2*k+nb_averaging_window))] )

            #time to update every normal ...
            for k in range(len(cluster.smooth_local_lines) -1 ) :
                local_smooth_normal = (cluster.smooth_local_lines[k+1].center - cluster.smooth_local_lines[k].center)/np.linalg.norm(cluster.smooth_local_lines[k+1].center - cluster.smooth_local_lines[k].center)
                cluster.smooth_local_lines[k].update_normal( local_smooth_normal )
                cluster.smooth_local_lines[k].update_norm_normal(np.linalg.norm(cluster.smooth_local_lines[k+1].center - cluster.smooth_local_lines[k].center))
                cluster.smooth_local_lines[k].update_next(k+1)
            
            #last one to update ! 
            previous_normal = cluster.smooth_local_lines[len(cluster.smooth_local_lines)-2].local_normal
            previous_norm = cluster.smooth_local_lines[len(cluster.smooth_local_lines)-2].norm_normal
            cluster.smooth_local_lines[len(cluster.smooth_local_lines)-1].update_normal( previous_normal )
            cluster.smooth_local_lines[len(cluster.smooth_local_lines)-1].update_norm_normal( previous_norm )
            cluster.smooth_local_lines[len(cluster.smooth_local_lines)-1].update_next(None)

            barycenter.append(barycenter_loc)

        return barycenter
            
    def ComputeBurgerOnLineDraft(self, rcut_burger : float, 
                                 nye_tensor : np.ndarray, 
                                 descriptor : np.ndarray = None) -> None : 
        """Compute the local burger vector for each point of the ```local_line```. If descriptor is not None, the 
        same reweighting procedure than in BuildOrderingLine is used to compute the local average Nye tensor
        
        Parameters:
        ----------

        rcut_burger : float 
            Cutoff raduis to average Nye tensor (in AA)

        nye_tensor : np.ndarray 
            Nye tensor associated to the whole ```system```

        descriptor : np.ndarray 
            descriptor vector associated to the ```system```

        """
        for _, cluster in self.dislocations.items():
            for id_line in cluster.local_lines.keys() : 
                center_line = cluster.local_lines[id_line].center
                id_at2keep = [id for id, atom in enumerate(self.system) if np.linalg.norm(atom.position - center_line) < rcut_burger]
                if len(id_at2keep) == 0 : 
                    raise ValueError(f'Number of atom in rcut for id line {id_line} is zero !')

                weights = np.ones(len(id_at2keep))
                if descriptor is not None :
                    subset_descriptor = descriptor[id_at2keep , :]
                    weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])   

                nye_tensor_id_line = np.average( nye_tensor[id_at2keep], weights=weights, axis=0 )
                sub_convex_hull = ConvexHull(self.system.positions[id_at2keep])

                burger_vector_line = nye_tensor_id_line@cluster.local_lines[id_line].local_normal*sub_convex_hull.volume
                cluster.local_lines[id_line].update_burger(burger_vector_line)

        return 

    def ComputeBurgerOnLineSmooth(self, rcut_burger : float, nye_tensor : np.ndarray, descriptor : np.ndarray = None) -> None : 
        """Compute the local burger vector for each point of the ```smooth_local_line```. If descriptor is not None, the 
        same reweighting procedure than in BuildOrderingLine is used to compute the local average Nye tensor
        
        Parameters:
        ----------

        rcut_burger : float 
            Cutoff raduis to average Nye tensor (in AA)

        nye_tensor : np.ndarray 
            Nye tensor associated to the whole ```system```

        descriptor : np.ndarray 
            descriptor vector associated to the ```system```

        """
        
        for _, cluster in self.dislocations.items():
            for id_line in cluster.smooth_local_lines.keys() : 
                center_line = cluster.smooth_local_lines[id_line].center
                id_at2keep = [id for id, atom in enumerate(self.system) if np.linalg.norm(atom.position - center_line) < rcut_burger ]
                
                if len(id_at2keep) < 4 : 
                    raise ValueError(f'''Number of atom in rcut for id line {id_line} < 5 ({len(id_at2keep)}) 
                                       rcut_bruger has to be increased !''')

                weights = np.ones(len(id_at2keep))
                if descriptor is not None :
                    subset_descriptor = descriptor[id_at2keep , :]
                    weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])   

                nye_tensor_id_line = np.average( nye_tensor[id_at2keep], weights=weights, axis=0)
                sub_convex_hull = ConvexHull(self.system.positions[id_at2keep])

                burger_vector_line = nye_tensor_id_line@cluster.smooth_local_lines[id_line].local_normal*sub_convex_hull.volume
                cluster.smooth_local_lines[id_line].update_burger(burger_vector_line)
        return 


    def NyeTensor(self, theta_max: float = 27) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
        """Computes strain properties and Nye tensor for ```system``` object. As described in the ```__init__``` documentation, 
        Nye tensor calculation need to used two buffer region : (i) ```extended_system``` and (ii) ```full_atoms```. 

        !ADD THE REFERENCE FOR THE ROUTINE!

        Parameters
        ----------
        
        theta_max : float
            Number of angle used to compute Nye tensor 

        Returns:
        --------
        
        np.ndarray 
            Nye tensor for the whole ```system```

        np.ndarray 
            array of neighbours for ```system``` (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

        np.ndarray 
            array of index of neighbours for ```system``` (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}

        np.ndarray 
            array of neighbours for ```extended_system``` (M,N,3) for each line i -> { r_{neigh_i,n} - r_i }_{1 \leq n \leq N} (cartesian)

        np.ndarray 
            array of index of neighbours for ```extended_system``` (M,N) for each line i -> {idx_{neigh_i,n}}_{1 \leq n \leq N}

        """

        dictionnary_p_vectors = {'bcc':self.p_vector_bcc,
                                 'fcc':self.p_vectors_fcc,
                                 'hcp':self.p_vector_hcp}
        if not self.structure['structure'] in dictionnary_p_vectors.keys() : 
            raise NotImplementedError(f'This is not implemented : {self.structure}')
        else :
            p_vectors = dictionnary_p_vectors[self.structure['structure']]
            if len(p_vectors) == 1 :
                p_vectors = np.broadcast_to( p_vectors@self.structure['unit_cell'], (len(self.extended_system), p_vectors.shape[0], 3))
            elif len(p_vectors) != len(self.system) :
                p_vectors = np.broadcast_to( p_vectors@self.structure['unit_cell'], (len(self.extended_system), p_vectors.shape[0], 3))

        # Neighbor list setup
        if self.rcut is None :
            raise ValueError('neighbors or cutoff is required')
        else : 

            array_neighbour, index_array, array_neighbour_ext, index_array_ext = get_N_neighbour(self.system,
                                                self.extended_system, 
                                                self.full_system, 
                                                self.rcut,
                                                len(dictionnary_p_vectors[self.structure['structure']]),
                                                pbc=(True, True, True))

        # Get cos of theta_max
        cos_theta_max = np.cos(theta_max * np.pi / 180)

        # Define epsilon array
        eps = np.array([[[ 0, 0, 0],[ 0, 0, 1],[ 0,-1, 0]],
                        [[ 0, 0,-1],[ 0, 0, 0],[ 1, 0, 0]],
                        [[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 0]]])

        # Identify largest number of nearest neighbors
        nmax = array_neighbour.shape[1]

        # Initialize variables
        nye = np.empty((len(self.system), 3, 3))
        P = np.zeros((nmax, 3))
        Q = np.zeros((nmax, 3))
        G = np.empty((len(self.extended_system), 3, 3))
        gradG = np.empty((3, 3, 3))

        # Calculate correspondence tensor, G, and strain data for each atom
        for i in range(len(self.extended_system)):
            p = np.asarray(p_vectors[i])
            if p.ndim == 1:
                p = np.array([p])
            p_mags = np.linalg.norm(p, axis=1)
            r1 = p_mags.min()

            # Calculate radial neighbor vectors, q
            q = array_neighbour_ext[i,:,:] 
            if q.shape[0] == 1:
                q = np.array([q])
            q_mags = np.linalg.norm(q, axis=1)

            # Calculate cos_thetas between all p's and q's.
            cos_thetas = (np.dot(p, q.T) /q_mags ).T / p_mags

            # Identify best index matches
            index_pairing = cos_thetas.argmax(1)

            # Exclude values where theta is greater than theta_max
            index_pairing[cos_thetas.max(1) < cos_theta_max] = -1

            # Search for duplicate index_pairings
            #u, u_count = np.unique(index_pairing, return_counts=True)

            # Check if the particular p has already been assigned to another q
            for n in range(len(q)):
                if index_pairing[n] >=0:
                    for k in range(n):
                        if index_pairing[n] == index_pairing[k]:
                            nrad = abs(r1 - q_mags[n])
                            krad = abs(r1 - q_mags[k])
                            # Remove the p-q pair that is farther from r1
                            if nrad < krad:
                                index_pairing[k]=-1
                            else:
                                index_pairing[n]=-1

            # Construct reduced P, Q matrices from p-q pairs
            c = 0
            for n in range(len(q)):
                if index_pairing[n] >= 0:
                    Q[c] = q[n]
                    P[c] = p[index_pairing[n]]
                    c+=1

            # Compute lattice correspondence tensor, G, from P and Q
            if c == 0:
                G[i] = np.identity(3)
                warnings.warn('An atom lacks pair sets. Check neighbor list size')
            else:
                G[i] = np.linalg.lstsq(Q[:c], P[:c], rcond=None)[0]

        # Construct the gradient tensor of G, gradG for each atom
        for i in range(len(self.system)):
            Q = array_neighbour[i,:,:]
            if Q.shape[0] == 1:
                Q = np.array([Q])
            
            dG = G[index_array[i]] - G[i]
            for x in range(3):
                gradG[x,:] = np.linalg.lstsq(Q, dG[:,x,:], rcond=None)[0].T

            # Use gradG to calculate the nye tensor
            nye[i] = -1*np.einsum('ijm,ikm->jk', eps, gradG)

        return nye, array_neighbour, index_array, array_neighbour_ext, index_array_ext


    def BuildDislocations(self, rcut_line : float, 
                         rcut_burger : float, 
                         rcut_cluster : float,
                         scale_cluster : float,
                         descritpor : np.ndarray = None, 
                         smoothing_line : dict = None) -> None :
        """Full procedure method to build all the local dislocation properties...
        
        Parameters
        ----------

        rcut_line : float 
            Rcut in to be added in a ```ClusterDislo```

        rcut_burger : float 
            Cutoff raduis to average Nye tensor (in AA)
        
        rcut_cluster : float 
            Initial size of the each ```ClusterDislo``` (see the documentation)

        scale_custer : float 
            Scale parameter for  ```ClusterDislo``` (see the documentation)
        
        descriptor : np.ndarray 
            descriptor vector associated to the ```system```

        smoothing_line : dict 
            If smoothing line is not None, smoothing procedure is used and need nb_averaging_window key (see LineSmoothing method)
        """

        Nye_tensor, _, _, array_neighbour_ext, index_neighbour_ext = self.NyeTensor()
        self.BuildSamplingLine(rcut_line=rcut_line, rcut_cluster=rcut_cluster, scale_cluster=scale_cluster)
        self.StartingPointCluster()
        self.RefineSamplingLine(scale=scale_cluster)
        self.BuildOrderingLine(array_neighbour_ext, descriptor=descritpor, idx_neighbor=index_neighbour_ext)
        if smoothing_line is not None : 
            self.LineSmoothing(nb_averaging_window=smoothing_line['nb_averaging_window'])
            self.ComputeBurgerOnLineSmooth(rcut_burger, Nye_tensor, descriptor=descritpor)
        else :
            self.ComputeBurgerOnLineDraft(rcut_burger, Nye_tensor, descriptor=descritpor)
        
        return 
