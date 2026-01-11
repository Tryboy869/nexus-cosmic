"""
Nexus-Cosmic Core Module

Contains the main engine, entities, topologies, and force laws
"""

from nexus_cosmic.core.engine import NexusCosmic
from nexus_cosmic.core.entity import UniversalEntity
from nexus_cosmic.core.topology import Topology
from nexus_cosmic.core.laws import ForceLaw
from nexus_cosmic.core.base import CustomLaw, CustomTopology, CustomFreeze

__all__ = [
    'NexusCosmic',
    'UniversalEntity',
    'Topology',
    'ForceLaw',
    'CustomLaw',
    'CustomTopology',
    'CustomFreeze',
]
