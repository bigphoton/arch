"""
Functions and properties describing quantum 
and classical states which are acted upon 
by blocks.

State
 * Represents 'abstract' or 'prototype' state
 * Specifies possible attributes/parameters
 * Specifies possible state methods
 * Designed to be subclassed by users

"""

from __future__ import annotations
import abc
from typing import Any, Optional, List, Union, Set


import arch.port as port
from arch.port import var
from arch.models import Model
from arch.connectivity import Connectivity


class State(abc.ABC):
    """
    * Represents 'abstract' or 'prototype' state
    * Specifies possible attributes/parameters
    * Specifies possible state methods
    * Designed to be subclassed by users

    Subclasses must:
     - Define the reference_prefix attribute
     - Implement the define method
     """

    # Dictionary of reference designators for all blocks
    names = dict()

    # To be overridden by subclasses:
    reference_prefix = "_"

    def __init__(self, _copy: bool = False, **kwargs: Optional[Any]) -> None:

        # Handle reference designator generation
        self._setup_new_name()

        # Store the init kwargs for use by copier
        self.__init_kwargs = kwargs

        # Run subclass define routine
        if not _copy:
            self.define(**kwargs)

        # Prevent post-define() modification
        self._inited = True

    

    @abc.abstractmethod
    def define(self, **kwargs: Optional[Any]) -> None:
        """
        Method overridden by subclasses to implement the block. kwargs are
        passed directly from __init__.

        This is an abstract method which does not do anything, but allows
        the use of this as a baseclass where these methods are inherited.

        i.e can inherit from this block and do not need to have a define
        method in order to work.
        """
        pass



    def _setup_new_name(self) -> None:

        """
        Ensure the reference prefix attribute is overwritten.

        """

        # if reference prefix is blank, raise error
        assert self.reference_prefix != "_", \
            "reference_prefix must be set by all Block subclasses."

        #if error raised then try
        try:
            existing_indices = Block.names[self.reference_prefix]
            self.reference_index = max(existing_indices) + 1

        except KeyError:
            self.reference_index = 0
            Block.names.update({self.reference_prefix: set()})

        Block.names[self.reference_prefix].add(self.reference_index)

        self.name = self.reference_prefix + str(self.reference_index)


    def __copy__(self) -> None:
        """Copy routine. Copying is forbidden."""
        raise RuntimeError(
            "Shallow copying (using copy.copy) is not allowed for objects "
            "of type Block. Use Block.copy instead.")


    def __deepcopy__(self) -> None:
        """Deep copy routine. Copying is forbidden."""
        raise RuntimeError(
            "Deep copying (using copy.deepcopy) is not allowed for "
            "objects of type Block. Use Block.copy instead.")


    def copy(self) -> Block:
        #TODO: Update this method for the state class SC/QP 07/07/21
        pass
        """Routine to copy this block."""
        """  cls = self.__class__
        new_self = cls.__new__(cls)
        new_self.__init__(_copy=True)

        port_map = dict()
        for p in self.ports:
            port_map[p] = new_self.add_port(name=p.local_name, kind=p.kind,
                                            direction=p.direction)

        for m in self.models:
            print("This needs to have its ports changed")
            new_self.add_model(m.copy(port_map=port_map))

        return new_self """

    #For the sake of generality at the moment there are no 
    #required properties. This may change.