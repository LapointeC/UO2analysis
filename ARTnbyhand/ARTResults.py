from __future__ import annotations

import os
import sys
import numpy as np
import pickle

from ase import Atom, Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp
from typing import List, Dict, Tuple, TypedDict


class AllInstance:
    """All event object to store all the exploration without connectivity reduction

    Attributes 
    ----------

    all_event : List[Parameters_reduced]
    List which contains all the dictonnaries of instance of the ART procedure, these dictionnaries have the 
    following key : 
        - 'name' : str => name of the event 
        - 'configuration' : Atoms => ASE object containing the system associated to the event 
        - 'type' : str => type of instance i.e saddle point or minimum
        - 'energy' : float => energy associated to the instance 
        - 'descriptor' : np.ndarray => inertia moment tensor associated to the instance

    """
    def __init__(self, name : str, system: Atoms, energy: float, type: str, descriptor: np.ndarray):
        self.all_instance : List[Parameters_reduced] = []
        self.add_AllInstance(name,system,energy,type,descriptor)

    def add_AllInstance(self, name : str, system: Atoms, energy: float, type: str, descriptor: np.ndarray):
        new_reducted_instance : Parameters_reduced = {'name': name,
                                                  'configuration': system,
                                                  'type': type,
                                                  'energy': energy,
                                                  'descriptor': descriptor}

        self.all_instance.append(new_reducted_instance)

    def dump_pickles_AllInstantce_object(self, path_pickle : str) : 
        """Dump a pickle archive for ARTResults object 

        Parameters 
        ----------

        path_pickle : str 
            path to write the pickle archive

        """

        pickle.dump(self, open('%s/dump_ARTn_AllInstance.sav' %
                    (path_pickle), 'wb'))     


    def dump_AllInstance_vasp_config(self,path_poscar: str):
        """Dump system in POSCAR format

        Parameters 
        ----------

        path_poscar : str 
            path to write dump files

        """
        for instance in self.all_instance : 
            system = instance['configuration']
            name_poscar = '%s_%s'%(instance['type'],instance['name'])
            write('%s/%s'%(path_poscar, name_poscar), system,sort=True, direct=True, vasp5=True, format='vasp')


class Parameters_reduced(TypedDict):
    name : str
    configuration: Atoms
    type: str
    energy: float
    descriptor: np.ndarray


class Event:
    """Event object to store connectivity and data about exploration...

    Parameters
    ----------

    name : str
        Name of a given mimimum 

    minimum : Atoms 
        ASE object containing minimum configuration

    energy : float 
        Energy associated to the minimum

    descritpor : np.ndarray 
        Inertia moment tensor of the minimum

    last : bool 
        boolean for linear update


    Attributes
    ----------

    last : boolean for linear update

    parameters : Typed dictonnary with following keys :
        - 'name' : str => Name of a given mimimum 
        - 'minimum' : Atoms => ASE object containing 'minimum' configuration
        - 'energy' : float => Energy associated to the 'minimum'
        - 'descritpor' : np.ndarray => Inertia moment tensor of the 'minimum'

    connectivity : Typed dictionnary with following keys :
        - 'neigh' : List[Atoms] => list of connected mimima to 'minimum' configuration
        - 'saddle' : List[Atoms] => list of saddle points which connect with minima
        - 'barrier_in' : List[float] => list of barriers to go from given connected minimum to the 'mimimum' configuration
        - 'barrier_out' : List[float] =>list of barriers to go from the 'mimimum' configuration to a given connected minimum

    """

    def __init__(self, name: str, minimum: Atoms, energy: float, descriptor: np.ndarray, last: bool = False):
        self.parameters: Parameters = {'name': name,
                                       'minimum': minimum,
                                       'energy': energy,
                                       'descriptor': descriptor}
        self.connectivity: Connectivity = {'neigh': [],
                                           'saddle': [],
                                           'barrier_in': [],
                                           'barrier_out': []}
        self.last = last

    def update_connectivity(self, neigh: List[Event], saddle: List[Atom], barrier_in: List[float], barrier_out: List[float]) -> None:
        """Update the connectivity attribute

        Parameters 
        ----------

        neigh : List[Events] 
            List of Events connected to the mimimum

        saddle : List[Atoms] 
            List of ASE objects containing saddle points which connected the mimima

        barrier_in : List[float]
            List of energy barriers to go in 'minimum' configuration

        barrier_out : List[float]
            List of energy barriers to go out 'minimum' configuration

        """
        self.connectivity['neigh'] += neigh
        self.connectivity['saddle'] += saddle
        self.connectivity['barrier_in'] += barrier_in
        self.connectivity['barrier_out'] += barrier_out


class Parameters(TypedDict):
    name: str
    minimum: Atoms
    energy: float
    descriptor: np.ndarray


class Connectivity(TypedDict):
    neigh: List[Event] | None
    saddle: List[Atoms] | None
    barrier_in: List[float] | None
    barrier_out: List[float] | None


class ARTResults:
    def __init__(self, init_system: Atoms, init_energy: float, init_tensor: np.ndarray, tol_energy: float = 1e-3, tol_inertia: float = 1e-2):
        self.tol_energy = tol_energy
        self.tol_inertia = tol_inertia

        eig_val_init_tensor = self.diagonalise_inertia_tensor(init_tensor)

        """Linear update for events"""
        self.events = [
            Event('00000', init_system, init_energy, eig_val_init_tensor, last=True)]

        """Recursive update for graph events"""
        self.events_graph = Event(
            '00000', init_system, init_energy, eig_val_init_tensor, last=True)

    def dump_pickle(self, path_pickle: str):
        """Dump a pickle archive for ARTResults object 

        Parameters 
        ----------

        path_pickle : str 
            path to write the pickle archive
        """

        pickle.dump(self, open('%s/dump_ARTn.sav' %
                    (path_pickle), 'wb'))

    def dump_vasp_config(self, system: Atoms, path_poscar: str, name_poscar: str):
        """Dump system in POSCAR format

        Parameters 
        ----------

        system : Atoms 
            ASE object containing the system

        path_poscar : str 
            path to write dump files

        name_poscar : str 
            name of the poscar
        """
        write('%s/%s' % (path_poscar, name_poscar), system,
              sort=True, direct=True, vasp5=True, format='vasp')

    def dump_log_file(self, name: str, minimum_energy: float, saddle_energy: float, descriptor: np.ndarray, log_file: str):
        """Writing the log file the follow the exploration

        Parameters 
        ----------

        name : str
            name of the minimum

        minimum_energy : float 
            energy associated to the minimum

        saddle_nergy : float 
            energy of the last saddle point

        eig_val : np.ndarray 
            descritpor of the mimimum 

        log_file : str
            path to the log file 
        """
        w = open('%s/ARTnPython.log'%(log_file), 'a')
        w.write('--> %s, e_min : %3.5f eV, e_sad : %3.5f eV \n' %
                (name, minimum_energy, saddle_energy))
        str_desc = 'descriptor : '
        for val in descriptor:
            str_desc += '%3.5f ' % (val)
        w.write('%s\n' % (str_desc))

    def diagonalise_inertia_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Diagonalise the inertia tensor and return its ordered eigen values

        Parameters 
        ----------

        tensor : np.ndarray
            Inertia tensor

        Returns
        -------

        np.ndarray 
            Ordered eigen values 

        """
        eig_val, _ = np.linalg.eig(tensor)
        return np.sort(eig_val)

    def update_last(self):
        """Update the boolean to identify the last explored event"""
        for compt, event in enumerate(self.events):
            if compt == len(self.events) - 1:
                event.last = False
            else:
                event.last = True

    def add_event_linear(self, name: str, system: Atoms, energy: float, next_energy: float, saddle_system: Atoms, energy_saddle: float, descriptor: np.ndarray, next_descritpor: np.ndarray) -> None:
        """Update the events list and fill the connectivity attribute for each event

        Parameters
        ----------

        name : str 
            Name of the new minimum

        system : Atoms 
            ASE object containing the mimimum configuration associated to the event

        energy : float
            Energy at the mimimum

        next_energy : float 
            Energy of the arriving mimimum to compare with others

        saddle_system : Atoms 
            ASE object containing the saddle configuration associated to the event

        energy_saddle : float 
            Energy at the saddle point

        descritpor : np.ndarray
            Inertia moment tensor of the minimum configuration

        next_descritpor : np.ndarray
            Inertia moment tensor of the arriving minimum to compare with others
        """

        eig_val = self.diagonalise_inertia_tensor(descriptor)
        next_eig_val = self.diagonalise_inertia_tensor(next_descritpor)
        self.update_last()

        tmp_neigh, tmp_barrier_in, tmp_barrier_out, tmp_saddle = [], [], [], []
        id_event2update = []

        for id_event, event in enumerate(self.events):
            if not event.last:
                delta_energy = abs(next_energy - event.parameters['energy'])
                delta_norm_tensor = np.linalg.norm(next_eig_val - event.parameters['descriptor'])
                if delta_energy < self.tol_energy and delta_norm_tensor < self.tol_inertia:
                    tmp_neigh.append(event)
                    tmp_barrier_out.append(energy_saddle-event.parameters['energy'])
                    tmp_barrier_in.append(energy_saddle-energy)
                    tmp_saddle.append(saddle_system)
                    id_event2update.append(id_event)

            else:
                tmp_neigh.append(event)
                tmp_barrier_out.append(energy_saddle-event.parameters['energy'])
                tmp_barrier_in.append(energy_saddle-energy)
                tmp_saddle.append(saddle_system)
                id_event2update.append(id_event)

        new_event = Event(name, system, energy, eig_val, last=True)
        new_event.update_connectivity(
            tmp_neigh, tmp_saddle, tmp_barrier_in, tmp_barrier_out)

        for compt, id in enumerate(id_event2update):
            self.events[id].update_connectivity(
                new_event, [saddle_system], tmp_barrier_out[compt], tmp_barrier_in[compt])

    def add_event_graphs(self, name: str, system: Atoms, energy: float, next_energy: float, saddle_system: Atoms, energy_saddle: float, descriptor: np.ndarray, next_descriptor: np.ndarray, init: bool = True, event_graph: Event = None) -> None:
        """Update the events list and fill the connectivity attribute for each event

        Parameters
        ----------

        name : str 
            Name of the new minimum

        system : Atoms 
            ASE object containing the mimimum configuration associated to the event

        energy : float
            Energy at the mimimum

        next_energy : float 
            Energy of the arriving mimimum to compare with others

        saddle_system : Atoms 
            ASE object containing the saddle configuration associated to the event

        energy_saddle : float 
            Energy at the saddle point

        descritpor : np.ndarray
            Inertia moment tensor of the minimum configuration

        next_descritpor : np.ndarray
        Inertia moment tensor of the arriving minimum to compare with others
        """

        if init:
            eig_val = self.diagonalise_inertia_tensor(descriptor)
            next_eig_val = self.diagonalise_inertia_tensor(next_descriptor)
            event_graph_0 = self.events_graph

            """Updating !"""
            if len(event_graph_0.connectivity['neigh']) == 0:
                new_event = Event(name, system, energy, descriptor)
                event_graph_0.update_connectivity([new_event], [saddle_system], [
                                                  energy_saddle-event_graph_0.parameters['energy']], [energy_saddle-energy])
                return

            else:
                delta_energy = abs(
                    next_energy - event_graph_0.parameters['energy'])
                delta_norm_tensor = np.linalg.norm(next_eig_val - event_graph_0.parameters['descriptor'])
                if delta_energy < self.tol_energy and delta_norm_tensor < self.tol_inertia:
                    new_event = Event(name, system, energy, descriptor)
                    event_graph_0.update_connectivity([new_event], [saddle_system], [
                                                      energy_saddle-energy], [energy_saddle-event_graph_0.parameters['energy']])
                    return

            for event_graph_k in event_graph_0.connectivity['neigh']:
                self.add_event_graphs(name, system, energy, saddle_system,
                                      energy_saddle, descriptor, init=False, event_graph=event_graph_k)

        else:
            if len(event_graph.connectivity['neigh']) == 0:
                new_event = Event(name, system, energy, descriptor)
                event_graph.update_connectivity([new_event], [saddle_system], [
                                                energy_saddle-energy], [energy_saddle-event_graph.parameters['energy']])
                return

            else:
                delta_energy = abs(
                    next_energy - event_graph.parameters['energy'])
                delta_norm_tensor = np.linalg.norm(
                    next_eig_val - event_graph.parameters['descriptor'])
                if delta_energy < self.tol_energy and delta_norm_tensor < self.tol_inertia:
                    new_event = Event(name, system, energy, descriptor)
                    event_graph.update_connectivity([new_event], [saddle_system], [
                                                    energy_saddle-energy], [energy_saddle-event_graph.parameters['energy']])
                    return

            for event_graph_k in event_graph.connectivity['neigh']:
                self.add_event_graphs(name, system, energy, saddle_system,
                                      energy_saddle, descriptor, init=False, event_graph=event_graph_k)
