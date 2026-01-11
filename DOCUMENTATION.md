# Nexus-Cosmic Developer Documentation

**Version:** 1.0.0  
**Language:** English | [Français](DOCUMENTATION_FR.md)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [API Reference](#api-reference)
4. [Patterns & Use Cases](#patterns--use-cases)
5. [Force Laws Explained](#force-laws-explained)
6. [Topologies Explained](#topologies-explained)
7. [Freeze Mechanism](#freeze-mechanism)
8. [Extensibility](#extensibility)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
pip install git+https://github.com/Tryboy869/nexus-cosmic.git
```

### Basic Usage

```python
from nexus_cosmic import NexusCosmic

# Distributed consensus
system = NexusCosmic(mode='consensus', n_entities=100)
result = system.run()
print(f"Consensus: {system.get_consensus()}")
print(f"Steps: {result['steps']}")

# Emergent sorting
system = NexusCosmic(mode='sorting', values=[8,3,9,1,5])
system.run()
print(system.get_sorted_values())  # [1,3,5,8,9]
```

---

## Core Concepts

### What is Nexus-Cosmic?

Nexus-Cosmic is a **distributed computing framework** based on **emergent physics principles**. Instead of centralized algorithms, it uses simple local interactions between entities that naturally converge to optimal solutions.

### Key Principles

1. **Emergence**: Complex global behavior from simple local rules
2. **Small-World Networks**: O(log N) information propagation
3. **Momentum**: Physics-inspired inertia for stability
4. **Freeze Mechanism**: Computational economy (100% in stable state)
5. **Discrete Attractors**: Guaranteed convergence for sorting

### Architecture

```
NexusCosmic System
├── Entities (UniversalEntity)
│   ├── state (position in solution space)
│   ├── velocity (momentum)
│   ├── mass (inertia)
│   └── frozen (freeze state)
├── Topology (neighbor connections)
│   ├── small_world
│   ├── full
│   ├── ring
│   └── grid
├── Force Law (interaction rules)
│   ├── adaptive_consensus
│   └── discrete_attractor
└── Freeze Mechanism (economy)
    ├── threshold
    └── stability_steps
```

---

## API Reference

### NexusCosmic Class

#### Constructor

```python
NexusCosmic(
    mode=None,                    # 'consensus', 'sorting', or None (custom)
    n_entities=None,              # Number of entities (required for consensus/custom)
    values=None,                  # Values to sort (required for sorting)
    topology='small_world',       # Network topology
    shortcuts_per_node=2,         # For small_world topology
    momentum=0.8,                 # Momentum factor [0, 1]
    strength=None,                # Force strength (auto if None)
    force_law=None,               # Custom force law
    freeze_enabled=True,          # Enable freeze mechanism
    freeze_threshold=0.01,        # Stability threshold
    freeze_stability_steps=5,     # Steps before freezing
    seed=None                     # Random seed
)
```

#### Parameters Explained

**mode** (str): Operating mode
- `'consensus'`: Distributed consensus (all entities converge to same value)
- `'sorting'`: Emergent sorting (entities organize by value)
- `None`: Custom mode (define your own force law)

**n_entities** (int): Number of entities in system
- Consensus: 10-1000 (performance degrades >1000)
- Custom: Any positive integer

**values** (list): Values for sorting mode
- Must be numeric (int or float)
- Any length (tested up to 10,000)

**topology** (str): Network structure
- `'small_world'`: Best performance (O(log N) diameter)
- `'full'`: Fastest for small N (<50)
- `'ring'`: Experimental (slow but reliable)
- `'grid'`: Experimental (2D lattice)

**momentum** (float): Inertia factor [0, 1]
- `0.0`: No momentum (reactive)
- `0.8`: Recommended (balanced)
- `1.0`: Maximum inertia (slow to change)

**strength** (float or callable): Force magnitude
- `None`: Auto-adapts to N (recommended)
- `float`: Fixed strength
- `callable`: Function of N, e.g., `lambda n: 0.1 * log(n)`

**freeze_threshold** (float): Variance threshold for freezing
- `0.01`: Recommended (tight convergence)
- `0.1`: Looser (faster but less precise)

**freeze_stability_steps** (int): Steps stable before freezing
- `5`: Recommended (reliable)
- `1`: Aggressive (risk oscillations)
- `10`: Conservative (slower freeze)

#### Methods

##### run()

Execute simulation until convergence.

```python
result = system.run(
    max_steps=100,           # Maximum iterations
    convergence_threshold=1.0,  # Variance threshold
    verbose=False            # Print progress
)
```

**Returns:** dict with keys:
- `converged` (bool): Whether system converged
- `steps` (int): Number of steps taken
- `final_variance` (float): Final variance
- `final_freeze_ratio` (float): Ratio of frozen entities [0, 1]
- `active_entities` (int): Number of active entities

##### get_consensus()

Get consensus value (average of all entity states).

```python
consensus = system.get_consensus()  # float
```

##### get_sorted_values()

Get sorted values (for sorting mode).

```python
sorted_vals = system.get_sorted_values()  # list
```

##### inject_change()

Inject local change (unfreezes affected zone).

```python
system.inject_change(
    entity_id=15,      # Entity to modify
    new_state=100.0    # New state value
)
```

**Use case:** Test resilience, simulate external events

##### reset()

Reset system to initial state.

```python
system.reset()
```

##### variance()

Compute current variance.

```python
var = system.variance()  # float
```

##### count_active()

Count active (non-frozen) entities.

```python
active = system.count_active()  # int
```

##### get_frozen_ratio()

Get ratio of frozen entities.

```python
ratio = system.get_frozen_ratio()  # float [0, 1]
```

---

## Patterns & Use Cases

### Pattern 1: Distributed Consensus

**Problem:** N nodes need to agree on a value without central coordinator.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Initialize with random states
system = NexusCosmic(mode='consensus', n_entities=100)

# Run until convergence
result = system.run()

# Get agreed value
consensus = system.get_consensus()

print(f"All 100 nodes agree on: {consensus}")
print(f"Converged in {result['steps']} steps")
```

**Real-world applications:**
- Blockchain consensus protocols
- Distributed cache synchronization
- Multi-agent coordination
- Sensor network fusion

### Pattern 2: Emergent Sorting

**Problem:** Sort values using distributed entities.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Unsorted data
values = [42, 17, 91, 3, 58, 24, 76]

# Create sorting system
system = NexusCosmic(mode='sorting', values=values)

# Run sorting
system.run()

# Get sorted result
sorted_values = system.get_sorted_values()

print(f"Sorted: {sorted_values}")
# Output: [3, 17, 24, 42, 58, 76, 91]
```

**Real-world applications:**
- Dynamic priority queues
- Task scheduling
- Leaderboard systems
- Resource allocation

### Pattern 3: Custom Force Law

**Problem:** Need custom interaction rules.

**Solution:**

```python
from nexus_cosmic import NexusCosmic, CustomLaw
import math

class GravityLaw(CustomLaw):
    """Entities attract proportionally to mass difference"""
    
    def compute_force(self, entity1, entity2):
        # Force magnitude
        distance = abs(entity2.state - entity1.state)
        if distance < 0.01:
            return 0.0
        
        force = (entity2.mass * entity1.mass) / (distance ** 2)
        
        # Direction
        direction = 1 if entity2.state > entity1.state else -1
        
        return force * direction * 0.01

# Use custom law
system = NexusCosmic(
    n_entities=50,
    force_law=GravityLaw(),
    topology='small_world'
)

result = system.run()
```

**Real-world applications:**
- Particle simulations
- Optimization problems
- Game AI (flocking, swarming)
- Network routing

### Pattern 4: Dynamic Updates

**Problem:** System must adapt to changes.

**Solution:**

```python
from nexus_cosmic import NexusCosmic

# Initial system
system = NexusCosmic(mode='consensus', n_entities=50)
system.run()

initial_consensus = system.get_consensus()
print(f"Initial consensus: {initial_consensus}")

# Inject external change
system.inject_change(entity_id=25, new_state=100.0)

# System re-adapts
system.run()

new_consensus = system.get_consensus()
print(f"New consensus: {new_consensus}")
```

**Real-world applications:**
- Real-time systems
- Adaptive networks
- Fault tolerance
- Self-healing systems

### Pattern 5: Weak Hardware Computing

**Problem:** Limited computational resources.

**Solution:**

```python
from nexus_cosmic import NexusCosmic
import random

class WeakHardwareEngine:
    def __init__(self, n_workers=100):
        self.system = NexusCosmic(
            mode='consensus',
            n_entities=n_workers,
            freeze_enabled=True  # Critical for economy
        )
    
    def distributed_average(self, large_dataset):
        # Each worker processes small chunk
        chunk_size = len(large_dataset) // self.system.n
        
        for i, entity in enumerate(self.system.entities):
            start = i * chunk_size
            end = start + chunk_size
            chunk = large_dataset[start:end]
            entity.state = sum(chunk) / len(chunk)
        
        # Emergent global average
        self.system.run()
        
        return self.system.get_consensus()

# Use on mobile device
engine = WeakHardwareEngine(n_workers=100)
big_data = [random.random() for _ in range(1_000_000)]

# Each worker: 10K values only
avg = engine.distributed_average(big_data)
```

**Real-world applications:**
- Mobile distributed computing
- IoT edge processing
- Federated learning
- Resource-constrained environments

---

## Force Laws Explained

### Adaptive Consensus Law

**Purpose:** Make entities converge to average value.

**Formula:**
```
force = (neighbor_state - entity_state) × strength
```

**Behavior:**
- Entities pull towards neighbors
- Strength adapts to system size N
- Guarantees convergence

**Parameters:**
- `strength`: Auto or `0.1 × log(N) / log(10)`

**Use when:**
- Need distributed agreement
- Want average/median value
- Require fault tolerance

**Code:**
```python
from nexus_cosmic.core.laws import ForceLaw

law = ForceLaw.adaptive_consensus(strength=0.2)
```

### Discrete Attractor Law

**Purpose:** Sort entities by creating discrete positions.

**Formula:**
```
target_position = entity_rank
force = (target_position - current_position) × strength
```

**Behavior:**
- Each entity attracted to ranked position
- Creates sorted order
- Discrete attractors prevent overlap

**Parameters:**
- `strength`: `0.3` (fixed)
- `method`: `'discrete_attractors'`

**Use when:**
- Need sorted output
- Want guaranteed ordering
- Priority-based systems

**Code:**
```python
from nexus_cosmic.core.laws import ForceLaw

law = ForceLaw.discrete_attractor()
```

### Custom Laws

Create your own interaction rules.

**Template:**

```python
from nexus_cosmic import CustomLaw

class MyLaw(CustomLaw):
    def __init__(self, param1=1.0):
        self.param1 = param1
    
    def compute_force(self, entity1, entity2):
        """
        Compute force on entity1 from entity2
        
        Args:
            entity1: Entity receiving force
            entity2: Entity exerting force
        
        Returns:
            float: Force magnitude (positive = pull right, negative = pull left)
        """
        # Your logic here
        diff = entity2.state - entity1.state
        force = diff * self.param1
        return force
```

**Examples:**

#### Exponential Decay

```python
class ExponentialLaw(CustomLaw):
    def __init__(self, decay=0.5):
        self.decay = decay
    
    def compute_force(self, e1, e2):
        import math
        distance = abs(e2.state - e1.state)
        magnitude = math.exp(-self.decay * distance)
        direction = 1 if e2.state > e1.state else -1
        return magnitude * direction * 0.1
```

#### Spring Force

```python
class SpringLaw(CustomLaw):
    def __init__(self, k=0.1, damping=0.9):
        self.k = k
        self.damping = damping
    
    def compute_force(self, e1, e2):
        # Hooke's law with damping
        displacement = e2.state - e1.state
        spring_force = self.k * displacement
        damping_force = -self.damping * e1.velocity
        return spring_force + damping_force
```

#### Threshold Activation

```python
class ThresholdLaw(CustomLaw):
    def __init__(self, threshold=1.0, strength=0.2):
        self.threshold = threshold
        self.strength = strength
    
    def compute_force(self, e1, e2):
        diff = e2.state - e1.state
        if abs(diff) < self.threshold:
            return 0.0  # No force if within threshold
        return diff * self.strength
```

---

## Topologies Explained

### Small-World Topology

**Structure:** Local clusters + long-range shortcuts

**Diameter:** O(log N)

**Performance:** Best (27x speedup)

**Visual:**
```
Entity 0: [1, 2, 15, 42]  (neighbors: local + 2 random)
Entity 1: [0, 2, 3, 28]
Entity 2: [0, 1, 3, 7]
...
```

**Use when:**
- Default choice (almost always)
- Need fast convergence
- Have >20 entities

**Code:**
```python
system = NexusCosmic(
    n_entities=100,
    topology='small_world',
    shortcuts_per_node=2  # Number of random shortcuts
)
```

### Full Connectivity

**Structure:** Every entity connected to all others

**Diameter:** 1

**Performance:** Best for N < 50, expensive for large N

**Visual:**
```
Entity 0: [1, 2, 3, 4, ..., 99]  (all others)
Entity 1: [0, 2, 3, 4, ..., 99]
...
```

**Use when:**
- Very small systems (N < 50)
- Need absolute fastest convergence
- Don't care about scalability

**Code:**
```python
system = NexusCosmic(
    n_entities=30,
    topology='full'
)
```

### Ring Topology

**Structure:** Each entity connected to 2 neighbors (circular)

**Diameter:** N/2

**Performance:** Slow but reliable (experimental)

**Visual:**
```
Entity 0: [99, 1]  (previous, next)
Entity 1: [0, 2]
Entity 2: [1, 3]
...
Entity 99: [98, 0]
```

**Use when:**
- Testing/research
- Want predictable structure
- Need fault isolation

**Code:**
```python
system = NexusCosmic(
    n_entities=100,
    topology='ring'
)
```

### Grid Topology

**Structure:** 2D lattice (4 neighbors except edges)

**Diameter:** 2×sqrt(N)

**Performance:** Slow but stable (experimental)

**Visual:**
```
Grid 5×5:
0  1  2  3  4
5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24

Entity 12: [7, 11, 13, 17]  (up, left, right, down)
Entity 0: [1, 5]            (right, down only)
```

**Use when:**
- Spatial problems
- Image processing
- Cellular automata

**Code:**
```python
system = NexusCosmic(
    n_entities=25,  # Will create 5×5 grid
    topology='grid'
)
```

---

## Freeze Mechanism

### What is Freezing?

When an entity's state becomes stable (variance below threshold for N steps), it **freezes** and stops computing. This achieves **100% computational economy** in stable state.

### How It Works

```python
class UniversalEntity:
    def check_stability(self, threshold=0.01, stability_steps=5):
        # Compare to previous state
        change = abs(self.state - self.previous_state)
        
        if change < threshold:
            self.stability_counter += 1
            if self.stability_counter >= stability_steps:
                self.frozen = True  # Freeze!
        else:
            self.stability_counter = 0
```

### Configuration

```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    freeze_enabled=True,          # Enable freeze
    freeze_threshold=0.01,        # Stability threshold
    freeze_stability_steps=5      # Steps stable before freezing
)
```

### Monitoring Freeze

```python
# During simulation
result = system.run()
print(f"Frozen: {result['final_freeze_ratio']*100:.0f}%")
print(f"Active: {result['active_entities']}")

# Get freeze ratio
ratio = system.get_frozen_ratio()  # float [0, 1]

# Count active
active = system.count_active()  # int
```

### Unfreezing

Entities automatically unfreeze when:
1. They receive external change (inject_change)
2. Their neighbors change significantly
3. System is reset

```python
# Inject change (unfreezes zone)
system.inject_change(entity_id=50, new_state=100.0)

# Check unfrozen
active_after = system.count_active()  # Higher than before
```

### Benefits

1. **Computational Economy**: 100% savings in stable state
2. **Energy Efficiency**: Critical for IoT/mobile
3. **Scalability**: Large systems auto-optimize
4. **Fault Tolerance**: Local changes only affect local zone

---

## Extensibility

### Custom Topologies

```python
from nexus_cosmic import CustomTopology

class HexagonalTopology(CustomTopology):
    """Hexagonal grid (6 neighbors)"""
    
    def get_neighbors(self, entity_id, n_entities):
        """
        Return list of neighbor IDs
        
        Args:
            entity_id: ID of current entity
            n_entities: Total number of entities
        
        Returns:
            list of int: Neighbor IDs
        """
        # Your topology logic
        neighbors = []
        
        # Example: hexagonal neighbors
        row_size = int(n_entities ** 0.5)
        row = entity_id // row_size
        col = entity_id % row_size
        
        # Add 6 neighbors (hex pattern)
        # ... (implementation)
        
        return neighbors

# Use
system = NexusCosmic(
    n_entities=100,
    topology=HexagonalTopology()
)
```

### Custom Freeze Conditions

```python
from nexus_cosmic import CustomFreeze

class EnergyFreeze(CustomFreeze):
    """Freeze based on energy level"""
    
    def __init__(self, energy_threshold=0.1):
        self.threshold = energy_threshold
    
    def should_freeze(self, entity):
        """
        Check if entity should freeze
        
        Args:
            entity: UniversalEntity instance
        
        Returns:
            bool: True if should freeze
        """
        # Compute kinetic energy
        energy = 0.5 * entity.mass * (entity.velocity ** 2)
        
        return energy < self.threshold

# Use
from nexus_cosmic.core.engine import NexusCosmic

system = NexusCosmic(n_entities=50)
system.freeze_condition = EnergyFreeze(energy_threshold=0.05)
```

### Validation Tool

Test your custom laws before deployment.

```python
from nexus_cosmic import validate_law, CustomLaw

class MyLaw(CustomLaw):
    def compute_force(self, e1, e2):
        return (e2.state - e1.state) * 0.15

# Validate
results = validate_law(
    MyLaw(),
    n_runs=5,         # Number of test runs
    max_steps=100,    # Max steps per run
    n_entities=30     # System size
)

print(f"Success rate: {results['success_rate']}%")
print(f"Avg steps: {results['avg_steps']}")
print(f"Verdict: {results['verdict']}")
# Output: ✅ VALIDÉ, ⚠️ EXPERIMENTAL, or ❌ REJETÉ
```

---

## Performance Optimization

### NumPy Acceleration (Optional)

```bash
pip install numpy
```

Nexus-Cosmic automatically detects and uses NumPy for **2-3x speedup** on large systems.

### Choosing Optimal Parameters

**For speed:**
```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    topology='small_world',    # Best topology
    shortcuts_per_node=3,      # More shortcuts = faster
    momentum=0.9,              # High momentum = stability
    freeze_threshold=0.1       # Looser = faster freeze
)
```

**For precision:**
```python
system = NexusCosmic(
    mode='consensus',
    n_entities=100,
    topology='small_world',
    shortcuts_per_node=2,
    momentum=0.7,              # Lower = more reactive
    freeze_threshold=0.001,    # Tighter = more precise
    freeze_stability_steps=10  # More steps = reliable
)
```

### Scalability

| N Entities | Topology | Avg Steps | Time (ms) |
|------------|----------|-----------|-----------|
| 10         | full     | 8         | 0.5       |
| 50         | small_world | 12     | 2.1       |
| 100        | small_world | 15     | 5.3       |
| 500        | small_world | 22     | 45        |
| 1000       | small_world | 28     | 180       |

**Recommendation:** Use `small_world` for N > 20

---

## Troubleshooting

### System doesn't converge

**Symptom:** `result['converged'] == False`

**Solutions:**
1. Increase `max_steps`
2. Check force law (use `validate_law`)
3. Try different topology
4. Adjust `strength` parameter

```python
# Debug
result = system.run(verbose=True)  # See step-by-step
```

### Slow convergence

**Symptom:** Takes >100 steps

**Solutions:**
1. Use `topology='small_world'`
2. Increase `shortcuts_per_node`
3. Adjust `momentum` to 0.8-0.9
4. Install NumPy

### Oscillations

**Symptom:** Variance oscillates, never converges

**Solutions:**
1. Increase `momentum` (0.8-0.9)
2. Decrease `strength`
3. Use adaptive strength (set to `None`)

### Freeze not working

**Symptom:** `freeze_ratio` always 0%

**Solutions:**
1. Check `freeze_enabled=True`
2. Loosen `freeze_threshold` (0.01 → 0.1)
3. Reduce `freeze_stability_steps`

### Import errors

**Symptom:** `ModuleNotFoundError`

**Solution:**
```bash
pip uninstall nexus-cosmic
pip install git+https://github.com/Tryboy869/nexus-cosmic.git
```

---

## Advanced Examples

### Example 1: Blockchain Consensus

```python
from nexus_cosmic import NexusCosmic
import random

class BlockchainNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.proposed_value = random.uniform(0, 100)

# Create 100 blockchain nodes
nodes = [BlockchainNode(i) for i in range(100)]

# Each node proposes a value
system = NexusCosmic(mode='consensus', n_entities=100)

for i, node in enumerate(nodes):
    system.entities[i].state = node.proposed_value

# Reach consensus
result = system.run()

agreed_value = system.get_consensus()

print(f"All nodes agree on: {agreed_value:.2f}")
print(f"Consensus reached in {result['steps']} steps")
```

### Example 2: Task Scheduler

```python
from nexus_cosmic import NexusCosmic

class Task:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

# Tasks with priorities
tasks = [
    Task("Critical Bug", 90),
    Task("Feature Request", 30),
    Task("Documentation", 20),
    Task("Security Patch", 95),
    Task("Refactoring", 50),
]

# Sort by priority
priorities = [t.priority for t in tasks]

system = NexusCosmic(mode='sorting', values=priorities)
system.run()

sorted_priorities = system.get_sorted_values()

# Reorder tasks
task_order = [priorities.index(p) for p in sorted_priorities]
sorted_tasks = [tasks[i] for i in task_order]

print("Task execution order:")
for i, task in enumerate(sorted_tasks, 1):
    print(f"{i}. {task.name} (priority {task.priority})")
```

### Example 3: Sensor Network Fusion

```python
from nexus_cosmic import NexusCosmic
import random

# 50 temperature sensors with noise
sensors = [22.0 + random.gauss(0, 2) for _ in range(50)]

# Consensus to filter noise
system = NexusCosmic(mode='consensus', n_entities=50)

for i, reading in enumerate(sensors):
    system.entities[i].state = reading

result = system.run()

true_temperature = system.get_consensus()

print(f"Individual sensors: {sensors[:5]}...")
print(f"True temperature (consensus): {true_temperature:.2f}°C")
print(f"Noise reduced in {result['steps']} steps")
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Daouda Abdoul Anzize** (Nexus Studio)
- GitHub: [@Tryboy869](https://github.com/Tryboy869)
- Email: nexusstudio100@gmail.com

---

**Happy Computing with Nexus-Cosmic!** 🚀
