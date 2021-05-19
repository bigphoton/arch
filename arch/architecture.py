from typing import List, Iterable

from arch.simulations.simulations import Simulator
from arch.block import Block
from arch.connectivity import Connectivity


class Architecture(object):

    def __init__(self,
                 blocks: Iterable[Block] = None,
                 connections: Connectivity = None,
                 simulation: Simulator = None) -> None:

        self.__blocks = blocks
        self.__connections = connections
        self.__simulations = [simulation]

    @property
    def connections(self) -> Connectivity:
        return self.__connections

    @connections.setter
    def connections(self, new) -> None:
        self.__connections = new

    @property
    def simulations(self) -> List[Simulator]:
        return self.__simulations

    @connections.setter
    def simulations(self, new: List[Simulator]) -> None:
        self.__simulations = new

    @property
    def blocks(self) -> Iterable[Block]:
        return self.__blocks

    def add_connections(self, connections) -> None:
        self.__connections.append(connections)

    def add_simulation(self, simulation: Simulator) -> None:
        self.__simulations.append(simulation)
