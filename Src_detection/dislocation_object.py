import warnings
from typing import Dict, Optional, Tuple

from ase import Atoms 
from cluster import Cluster, ClusterDislo, LocalLine

import numpy as np
from scipy.spatial import ConvexHull
from neighbour import get_N_neighbour

class DislocationObject : 
    def __init__(self, system : Atoms, rcut : float, structure : str = 'bcc') -> None : 
        self.system = system
        self.rcut = rcut
        self.structure = structure
        self.convex_hull = self.get_convex_hull()
        
        self.dislocations : Dict[str,ClusterDislo] = {}
        self.starting_point_line = None

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

    def get_convex_hull(self) -> ConvexHull :
        return ConvexHull(self.system.positions)

    def BuildSamplingLine(self, rcut_line : float = 3.5, rcut_cluster : float = 4.5, scale_cluster : float = 1.5) -> None : 
        if rcut_cluster < rcut_line : 
            raise ValueError(f'Impossible to build lines with rcut_cluster {rcut_cluster} < rcut_line {rcut_line}')

        idx_dislo = 0
        starting_id = np.argmax( np.linalg.norm(self.system.positions - np.mean(self.system.positions, axis=0), axis=1)  )
        self.starting_point_line = starting_id

        self.dislocations[idx_dislo]  = ClusterDislo(LocalLine( self.system[starting_id].position ), starting_id, rcut_cluster)  

        for id_atom_i, at_i in enumerate(self.system) : 
            if id_atom_i == starting_id : 
                continue
            else : 
                
                #index_min_key = np.argmin([ np.linalg.norm(at_i.position - val.size ) for _, val in self.dislocations.items() ])
                min_dist_array = [ np.amin(np.linalg.norm(at_i.position - cluster.positions, axis=1)) for _,cluster in self.dislocations.items()]
                index_min_key = np.argmin(min_dist_array)
                closest_key = [key for key in self.dislocations.keys()][index_min_key]
                if min_dist_array[index_min_key] > rcut_line :
                    if self.dislocations[closest_key].get_elliptic_distance(at_i.position) > scale_cluster*self.dislocations[closest_key].size :
                            idx_dislo += 1 
                            self.dislocations[idx_dislo]  = ClusterDislo(LocalLine( self.system[idx_dislo].position ), idx_dislo, rcut_cluster)                        
                    else : 
                        self.dislocations[index_min_key].append(LocalLine(at_i.position), id_atom_i)

                else : 
                    continue 
                
        return 
    
    def StartingPointCluster(self) -> None :
        for _, cluster in self.dislocations.items(): 
            starting_id = np.argmax( np.linalg.norm(cluster.positions - cluster.center), axis=1)
            cluster.starting_point = starting_id
        return 


    def BuildOrderingLine(self, neighbour : np.ndarray, descriptor : np.ndarray = None, idx_neighbor : np.ndarray = None) -> None :
        for _, cluster in self.dislocations.items():
            weights = np.ones(neighbour.shape[1])
            if descriptor is not None :
                subset_descriptor = descriptor[idx_neighbor[self.starting_point_line] , :]
                weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])

            starting_point = np.average( neighbour[cluster.starting_point,:,:], axis=0, weights=weights)
            # updating data in local lines for the starting point!
            cluster.local_lines[cluster.starting_point].update_center(starting_point)
            #self.local_lines[self.starting_point_line].update_neighbour(neighbour[self.starting_point_line,:,:])

            explored_idx = [self.starting_point_line]
            for _ in range(len(cluster.local_lines) - 1 ) : 
                sub_local_line = {key:val for key, val in cluster.local_lines if key not in explored_idx}
                composite_id_distance = [(idx, np.linalg.norm(positions - starting_point)) for idx, positions in sub_local_line if np.linalg.norm(positions - starting_point) > 0.0 ]
                min_id, _ =  sorted(composite_id_distance, key=lambda x: x[1])[0]

                if descriptor is not None : 
                    subset_descriptor = descriptor[idx_neighbor[min_id] , :]
                    weights = np.array([ np.exp(desc)/np.exp(np.sum(subset_descriptor,axis=0)) for desc in subset_descriptor ])

                explored_idx.append(min_id)
                # build the new starting point for the iterative skim...
                new_starting_point = np.average( neighbour[min_id,:,:], axis=0, weights=weights)

                #updating the local normal from the previous starting point and next id 
                cluster.local_lines[explored_idx[-1]].update_normal( (new_starting_point - starting_point)/np.linalg.norm(new_starting_point - starting_point) )
                cluster.local_lines[explored_idx[-1]].update_next( min_id )

                #updating the new starting point ! 
                starting_point = new_starting_point

            return

    def ComputeBurgerOnLine(self, rcut_burger : float, nye_tensor : np.ndarray, descriptor : np.ndarray = None) -> None : 
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

                nye_tensor_id_line = np.average( nye_tensor, weights=weights, axis=0 )
                sub_convex_hull = ConvexHull(self.convex_hull.vertices[id_at2keep,:])

                burger_vector_line = nye_tensor_id_line@cluster.local_lines[id_line].local_normal*sub_convex_hull.volume
                cluster.local_lines[id_line].update_burger(burger_vector_line)

        return 

    #def get_local_normal(self,positions : np.ndarray) -> Tuple[np.ndarray,np.ndarray] : 
    #    centered_positions = positions - np.mean(positions, axis=0)
    #    _, _, vh = np.linalg.svd(positions - centered_positions, full_matrices=False)
    #    return vh[-1], centered_positions

    def NyeTensor(self, theta_max: float = 27) -> Tuple[np.ndarray,np.ndarray,np.ndarray] :

        """
        Computes strain properties and Nye tensor for a defect containing system.

        Parameters
        TO DO
        """
        dictionnary_p_vectors = {'bcc':self.p_vector_bcc,
                                 'fcc':self.p_vectors_fcc,
                                 'hcp':self.p_vector_hcp}
        if not self.structure in dictionnary_p_vectors.keys() : 
            raise NotImplementedError(f'This is not implemented : {self.structure}')
        else :
            p_vectors = dictionnary_p_vectors[self.structure]

        # Neighbor list setup
        if self.rcut is None :
            raise ValueError('neighbors or cutoff is required')
        else : 
            #neighbors_ij, neighbors_shift, _ = get_neighborhood(system.positions,
            #                                                                        cutoff,
            #                                                                        [True,True,True],
            #                                                                        system.cell[:],
            #                                                                        true_self_interaction=False)


            array_neighbour, index_array = get_N_neighbour(self.system, 
                                                self.rcut,
                                                dictionnary_p_vectors[self.structure],
                                                pbc=[True, True, True])

        # Get cos of theta_max
        cos_theta_max = np.cos(theta_max * np.pi / 180)

        # Define epsilon array
        eps = np.array([[[ 0, 0, 0],[ 0, 0, 1],[ 0,-1, 0]],
                        [[ 0, 0,-1],[ 0, 0, 0],[ 1, 0, 0]],
                        [[ 0, 1, 0],[-1, 0, 0],[ 0, 0, 0]]])

        # Identify largest number of nearest neighbors
        #counter = Counter(list(neighbors_ij[0,:]))
        #nmax = counter[max(counter, key=counter.get)]
        nmax = array_neighbour.shape[1]

        # Initialize variables
        nye = np.empty((len(self.system), 3, 3))
        P = np.zeros((nmax, 3))
        Q = np.zeros((nmax, 3))
        G = np.empty((len(self.system), 3, 3))
        gradG = np.empty((3, 3, 3))

        # Calculate correspondence tensor, G, and strain data for each atom
        for i in range(len(self.system)):
            p = np.asarray(p_vectors[i])
            if p.ndim == 1:
                p = np.array([p])
            p_mags = np.linalg.norm(p, axis=1)
            r1 = p_mags.min()

            # Calculate radial neighbor vectors, q
            #mask_i = neighbors_ij[0,:] == i
            #q = system.positions[ neighbors_ij[mask_i][1,:] ] - system.positions[ neighbors_ij[mask_i][0,:] ] + neighbors_shift[mask_i]
            q = array_neighbour[i,:,:] - self.system.positions[i,:]
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
            #mask_i = neighbors_ij[0,:] == i
            #Q = system.positions[ neighbors_ij[mask_i][1,:] ] - system.positions[ neighbors_ij[mask_i][0,:] ] + neighbors_shift[mask_i]
            #Q = system.dvect(i, neighbors[i])
            Q = array_neighbour[i,:,:] - self.system.positions[i,:]
            if Q.shape[0] == 1:
                Q = np.array([Q])
            dG = G[list(index_array[i])] - G[i]
            for x in range(3):
                gradG[x,:] = np.linalg.lstsq(Q, dG[:,x,:], rcond=None)[0].T

            # Use gradG to calculate the nye tensor
            nye[i] = -1*np.einsum('ijm,ikm->jk', eps, gradG)

        return nye, array_neighbour, index_array



    def BuildDislocation(self, rcut_line : float, rcut_burger : float, descritpor : np.ndarray = None) : 
        Nye_tensor, array_neighbour, index_neighbour = self.NyeTensor()
        self.BuildSamplingLine(rcut_line=rcut_line)
        self.StartingPointCluster()
        self.BuildOrderingLine(array_neighbour, descriptor=descritpor, idx_neighbor=index_neighbour)
        self.ComputeBurgerOnLine(rcut_burger, Nye_tensor, descriptor=descritpor)
        return 
