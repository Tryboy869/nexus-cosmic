"""
Base Classes for Extensibility

Users can extend these to create custom laws, topologies, and freeze conditions
"""

from typing import List


class CustomLaw:
    """
    Base class for custom force laws
    
    Example:
        >>> class MyLaw(CustomLaw):
        ...     def compute_force(self, e1, e2):
        ...         distance = abs(e2.state - e1.state)
        ...         return distance * 0.1
        ...
        >>> system = NexusCosmic(force_law=MyLaw())
    """
    
    def compute_force(self, entity1, entity2) -> float:
        """
        Compute force between two entities
        
        Args:
            entity1: Source entity (has .state, .mass, .charge, etc.)
            entity2: Target entity
        
        Returns:
            float: Force magnitude
                  Positive = towards entity2
                  Negative = away from entity2
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Custom laws must implement compute_force(entity1, entity2)"
        )


class CustomTopology:
    """
    Base class for custom topologies
    
    Example:
        >>> class HexGrid(CustomTopology):
        ...     def get_neighbors(self, entity_id, n_entities):
        ...         # Return 6 hexagonal neighbors
        ...         return [...]
        ...
        >>> system = NexusCosmic(topology=HexGrid())
    """
    
    def get_neighbors(self, entity_id: int, n_entities: int) -> List[int]:
        """
        Get neighbors for an entity
        
        Args:
            entity_id: ID of the entity (0 to n_entities-1)
            n_entities: Total number of entities
        
        Returns:
            List[int]: List of neighbor entity IDs
        
        Rules:
            - No self-loops (don't include entity_id)
            - Valid indices (0 to n_entities-1)
            - Can return empty list (isolated entity)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Custom topologies must implement get_neighbors(entity_id, n_entities)"
        )


class CustomFreeze:
    """
    Base class for custom freeze conditions
    
    Example:
        >>> class EnergyFreeze(CustomFreeze):
        ...     def should_freeze(self, entity):
        ...         energy = 0.5 * entity.velocity ** 2
        ...         return energy < 0.001
        ...
        >>> system = NexusCosmic(freeze_condition=EnergyFreeze())
    """
    
    def should_freeze(self, entity) -> bool:
        """
        Determine if entity should freeze
        
        Args:
            entity: Entity with .state, .velocity, .is_frozen, etc.
        
        Returns:
            bool: True if entity should freeze
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            "Custom freeze conditions must implement should_freeze(entity)"
        )


# Error classes for better user feedback

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class ConvergenceError(Exception):
    """Raised when system fails to converge"""
    pass


class ValidationError(Exception):
    """Raised when validation fails"""
    pass
