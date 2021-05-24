from typing import List, Iterable, Union

from arch.simulations.simulations import Simulator
from arch.block import Block


class Architecture(object):

    def __init__(self,
                 blocks: Union[Iterable[Block], None] = None,
                 connections: Union[List, None] = None,
                 simulation: Union[Simulator, None] = None) -> None:

        self.__blocks = blocks
        self.__connections = connections
        self.__simulations = [simulation]

    @property
    def connections(self) -> List:
        return self.__connections

    @connections.setter
    def connections(self, new) -> None:
        self.__connections = new

    @property
    def simulations(self) -> List[Simulator]:
        return self.__simulations

    @simulations.setter
    def simulations(self, new: List[Simulator]) -> None:
        self.__simulations = new

    @property
    def blocks(self) -> Iterable[Block]:
        return self.__blocks

    def add_connections(self, connections) -> None:
        self.__connections.append(connections)

    def add_simulation(self, simulation: Simulator) -> None:
        self.__simulations.append(simulation)
