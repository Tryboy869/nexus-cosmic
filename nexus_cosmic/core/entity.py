"""
Universal Entity - Atomic unit of the emergent system
"""


class UniversalEntity:
    """
    Atomic entity in the Nexus-Cosmic system
    
    Attributes:
        id: Unique entity identifier
        state: Current position/value in continuous space
        velocity: Momentum (inertia)
        mass: Physical mass property
        charge: Electric charge property
        is_frozen: Whether entity is frozen (no computation)
        value: Original value (for sorting mode)
    """
    
    def __init__(self, entity_id: int, initial_state: float):
        self.id = entity_id
        self.state = initial_state
        self.velocity = 0.0
        
        # Physical properties
        self.mass = 1.0
        self.charge = 1.0 if entity_id % 2 == 0 else -1.0
        
        # Freeze mechanism
        self.is_frozen = False
        self.stability_counter = 0
        self.previous_state = initial_state
        
        # Sorting
        self.value = None
    
    def apply_force(self, force: float, momentum_factor: float):
        """
        Apply force with momentum
        
        Args:
            force: Force magnitude
            momentum_factor: Momentum coefficient (0-1)
        """
        if self.is_frozen:
            return
        
        self.velocity = momentum_factor * self.velocity + (1 - momentum_factor) * force
        self.state += self.velocity
    
    def check_stability(self, threshold: float, stability_steps: int):
        """
        Check stability for freeze mechanism
        
        Args:
            threshold: Change threshold for stability
            stability_steps: Required stable steps before freeze
        """
        change = abs(self.state - self.previous_state)
        
        if change < threshold:
            self.stability_counter += 1
            if self.stability_counter >= stability_steps:
                if not self.is_frozen:
                    self.is_frozen = True
                    self.velocity = 0.0
        else:
            self.stability_counter = 0
            if self.is_frozen:
                self.is_frozen = False
        
        self.previous_state = self.state
    
    def unfreeze(self):
        """Unfreeze the entity"""
        self.is_frozen = False
        self.stability_counter = 0
    
    def __repr__(self):
        return f"Entity(id={self.id}, state={self.state:.2f}, frozen={self.is_frozen})"
