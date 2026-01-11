"""
Nexus-Cosmic: Universal Emergent Computation Engine

A distributed computing framework based on emergent physics principles.

Author: Daouda Abdoul Anzize (Nexus Studio)
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Daouda Abdoul Anzize"
__email__ = "nexusstudio100@gmail.com"
__license__ = "MIT"

from nexus_cosmic.core.engine import NexusCosmic
from nexus_cosmic.core.base import CustomLaw, CustomTopology, CustomFreeze
from nexus_cosmic.validation.validator import validate_law

__all__ = [
    'NexusCosmic',
    'CustomLaw',
    'CustomTopology', 
    'CustomFreeze',
    'validate_law',
    '__version__',
]
