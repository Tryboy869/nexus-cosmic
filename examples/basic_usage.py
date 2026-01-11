"""
Basic Usage Examples

Simple examples demonstrating core features
"""

from nexus_cosmic import NexusCosmic
import random


# ============================================================================
# EXAMPLE 1: CONSENSUS
# ============================================================================

print("=" * 80)
print("EXAMPLE 1: DISTRIBUTED CONSENSUS")
print("=" * 80)
print()

# Create system with 50 entities
system = NexusCosmic(mode='consensus', n_entities=50)

print(f"Initial variance: {system.variance():.4f}")

# Run until convergence
result = system.run(verbose=True)

print(f"\nConsensus value: {system.get_consensus():.4f}")
print(f"Converged: {result['converged']}")
print(f"Steps: {result['steps']}")
print(f"Frozen entities: {result['final_freeze_ratio']*100:.0f}%")


# ============================================================================
# EXAMPLE 2: SORTING
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 2: EMERGENT SORTING")
print("=" * 80)
print()

values = [8, 3, 9, 1, 5, 2, 7, 4, 6]
print(f"Unsorted: {values}")

system = NexusCosmic(mode='sorting', values=values)
result = system.run()

sorted_values = system.get_sorted_values()
print(f"Sorted:   {sorted_values}")
print(f"Steps: {result['steps']}")


# ============================================================================
# EXAMPLE 3: CUSTOM FORCE LAW
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 3: CUSTOM FORCE LAW")
print("=" * 80)
print()

from nexus_cosmic.core.base import CustomLaw
import math

class ExponentialDecayLaw(CustomLaw):
    """Force decays exponentially with distance"""
    
    def __init__(self, decay_rate=0.5):
        self.decay = decay_rate
    
    def compute_force(self, entity1, entity2):
        distance = abs(entity2.state - entity1.state)
        force_magnitude = math.exp(-self.decay * distance)
        direction = 1 if entity2.state > entity1.state else -1
        return force_magnitude * direction * 0.1

# Use custom law
system = NexusCosmic(
    n_entities=30,
    force_law=ExponentialDecayLaw(decay_rate=0.3),
    topology='small_world'
)

result = system.run()

print(f"Custom law converged: {result['converged']}")
print(f"Steps: {result['steps']}")
print(f"Final variance: {result['final_variance']:.4f}")


# ============================================================================
# EXAMPLE 4: DYNAMIC CHANGES
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 4: DYNAMIC CHANGES (FREEZE/UNFREEZE)")
print("=" * 80)
print()

system = NexusCosmic(mode='consensus', n_entities=30)
result = system.run()

print(f"Initial convergence: {result['steps']} steps")
print(f"Frozen: {result['final_freeze_ratio']*100:.0f}%")

# Inject local change
print("\nInjecting change at entity 15...")
system.inject_change(entity_id=15, new_state=100.0)

print(f"Active entities after injection: {system.count_active()}")

# Re-converge
result = system.run()

print(f"Re-convergence: {result['steps']} steps")
print(f"Frozen again: {result['final_freeze_ratio']*100:.0f}%")


# ============================================================================
# EXAMPLE 5: VALIDATION
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 5: VALIDATE CUSTOM LAW")
print("=" * 80)
print()

from nexus_cosmic import validate_law

class TestLaw(CustomLaw):
    def compute_force(self, e1, e2):
        return (e2.state - e1.state) * 0.15

results = validate_law(TestLaw(), n_runs=5)

print(f"Validation Results:")
print(f"  Success rate: {results['success_rate']:.0f}%")
print(f"  Avg steps: {results['avg_steps']:.1f}")
print(f"  Avg variance: {results['avg_variance']:.4f}")
print(f"  Verdict: {results['verdict']}")

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE!")
print("=" * 80)
