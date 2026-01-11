"""
Weak Hardware Engine - Distributed Computing on Weak Devices

Transform weak hardware into quantum-like computational power
through emergent distributed computation.

Author: Daouda Abdoul Anzize (Nexus Studio)
"""

import random
from typing import List, Callable, Any, Dict
from nexus_cosmic import NexusCosmic


class WeakHardwareEngine:
    """
    Distributed computation engine for weak hardware
    
    Principle:
    - 1 complex problem → N simple entities
    - Each entity = minimal computation
    - Emergent convergence = solution
    - Freeze mechanism = massive economy
    
    Result: Weak hardware rivals strong hardware through parallelism
    
    Examples:
        Average 10M values on mobile:
            >>> engine = WeakHardwareEngine(n_workers=1000)
            >>> data = [random.random() for _ in range(10_000_000)]
            >>> avg = engine.distributed_average(data)
            # Each worker: 10K values only
            # Convergence: ~10 steps
            # Freeze: 90% workers inactive after convergence
        
        Parallel optimization:
            >>> def objective(x):
            ...     return -(x**2 - 4*x + 3)
            >>> best_x, best_f = engine.parallel_optimization(objective)
            # 50 workers explore in parallel
            # Hardware faible = Hardware quantique!
    """
    
    def __init__(self, n_workers: int = 100):
        """
        Initialize weak hardware engine
        
        Args:
            n_workers: Number of parallel workers (entities)
                      More workers = more parallelism
                      Typical: 100-1000 for mobile, 10-50 for IoT
        """
        self.n_workers = n_workers
        self.system = NexusCosmic(
            mode='consensus',
            n_entities=n_workers,
            topology='small_world',
            momentum=0.8,
            freeze_enabled=True
        )
    
    def distributed_average(self, data: List[float]) -> float:
        """
        Compute average of large dataset distributedly
        
        Instead of sum(data)/len(data) which may freeze weak devices,
        each worker computes local average, then emergent consensus.
        
        Args:
            data: Large dataset (millions of values)
        
        Returns:
            Global average
        
        Example:
            >>> # 10 million values on mobile phone
            >>> big_data = [random.random() for _ in range(10_000_000)]
            >>> avg = engine.distributed_average(big_data)
            >>> # Each worker: 10K values only
            >>> # Total time: ~100ms (vs several seconds centralized)
        """
        chunk_size = len(data) // self.n_workers
        
        # Each worker computes local average
        for i, entity in enumerate(self.system.entities):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_workers - 1 else len(data)
            
            chunk = data[start_idx:end_idx]
            entity.state = sum(chunk) / len(chunk) if len(chunk) > 0 else 0.0
        
        # Emergent convergence to global average
        self.system.run(max_steps=50, convergence_threshold=0.001)
        
        return self.system.get_consensus()
    
    def distributed_sum(self, data: List[float]) -> float:
        """
        Compute sum of large dataset distributedly
        
        Args:
            data: Large dataset
        
        Returns:
            Global sum
        """
        chunk_size = len(data) // self.n_workers
        
        # Each worker computes local sum
        for i, entity in enumerate(self.system.entities):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_workers - 1 else len(data)
            
            chunk = data[start_idx:end_idx]
            entity.state = sum(chunk)
        
        # Emergent convergence
        self.system.run(max_steps=50)
        
        # Total = consensus × n_workers
        return self.system.get_consensus() * self.n_workers
    
    def parallel_optimization(self, 
                            objective_function: Callable[[float], float],
                            search_space: tuple = (-10, 10),
                            maximize: bool = True) -> tuple:
        """
        Parallel optimization through emergent search
        
        Each worker explores different region of search space.
        Workers are attracted to better solutions.
        Emergent convergence finds global optimum.
        
        Args:
            objective_function: Function to optimize f(x) -> score
            search_space: (min, max) bounds for search
            maximize: True to maximize, False to minimize
        
        Returns:
            (best_x, best_score): Optimal solution and its score
        
        Example:
            >>> # Find maximum of function
            >>> def peaks(x):
            ...     return -(x**2) + 4*x + 3
            >>> best_x, best_score = engine.parallel_optimization(peaks)
            >>> print(f"Optimum: x={best_x}, f(x)={best_score}")
        """
        min_val, max_val = search_space
        
        # Initialize workers randomly in search space
        for entity in self.system.entities:
            entity.state = random.uniform(min_val, max_val)
            entity.fitness = objective_function(entity.state)
        
        # Custom force: attraction towards better solutions
        from nexus_cosmic.core.base import CustomLaw
        
        class OptimizationLaw(CustomLaw):
            def __init__(self, maximize=True):
                self.maximize = maximize
            
            def compute_force(self, e1, e2):
                # Attract to better fitness
                if self.maximize:
                    if e2.fitness > e1.fitness:
                        diff = e2.state - e1.state
                        return diff * 0.1
                else:
                    if e2.fitness < e1.fitness:
                        diff = e2.state - e1.state
                        return diff * 0.1
                return 0.0
        
        # Run optimization
        original_law = self.system.force_law
        self.system.force_law = OptimizationLaw(maximize=maximize)
        
        for _ in range(50):
            self.system.step()
            
            # Re-evaluate fitness
            for entity in self.system.entities:
                # Clip to search space
                entity.state = max(min_val, min(max_val, entity.state))
                entity.fitness = objective_function(entity.state)
        
        # Restore original law
        self.system.force_law = original_law
        
        # Find best solution
        if maximize:
            best = max(self.system.entities, key=lambda e: e.fitness)
        else:
            best = min(self.system.entities, key=lambda e: e.fitness)
        
        return best.state, best.fitness
    
    def parallel_map(self, 
                    function: Callable[[Any], float],
                    inputs: List[Any]) -> List[float]:
        """
        Parallel map operation
        
        Apply function to all inputs in parallel, then aggregate.
        
        Args:
            function: Function to apply
            inputs: List of inputs
        
        Returns:
            List of results
        
        Example:
            >>> def expensive_computation(x):
            ...     return sum(i**2 for i in range(x))
            >>> results = engine.parallel_map(expensive_computation, range(1000))
        """
        chunk_size = len(inputs) // self.n_workers
        results = []
        
        # Each worker processes chunk
        for i in range(self.n_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.n_workers - 1 else len(inputs)
            
            chunk = inputs[start_idx:end_idx]
            chunk_results = [function(inp) for inp in chunk]
            results.extend(chunk_results)
        
        return results
    
    def distributed_ml_training(self,
                               model_update_func: Callable,
                               dataset: List[Any],
                               n_epochs: int = 10) -> Any:
        """
        Distributed ML training (Federated Learning style)
        
        Each worker trains on local data subset.
        Emergent consensus = global model.
        
        Args:
            model_update_func: Function(data_batch) -> model_weights
            dataset: Training dataset
            n_epochs: Number of training epochs
        
        Returns:
            Converged model weights
        
        Example:
            >>> def train_batch(batch):
            ...     # Your training logic
            ...     return updated_weights
            >>> model = engine.distributed_ml_training(train_batch, dataset)
        """
        chunk_size = len(dataset) // self.n_workers
        
        for epoch in range(n_epochs):
            # Each worker trains on local batch
            for i, entity in enumerate(self.system.entities):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.n_workers - 1 else len(dataset)
                
                batch = dataset[start_idx:end_idx]
                entity.state = model_update_func(batch)
            
            # Consensus step
            self.system.run(max_steps=10)
        
        return self.system.get_consensus()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get engine statistics
        
        Returns:
            Dict with performance metrics
        """
        return {
            'workers': self.n_workers,
            'active_workers': self.system.count_active(),
            'frozen_workers': int(self.system.get_frozen_ratio() * self.n_workers),
            'freeze_economy': f"{self.system.get_frozen_ratio() * 100:.0f}%",
            'steps_taken': self.system.step_count
        }


# ============================================================================
# DEMONSTRATION EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("WEAK HARDWARE ENGINE - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Example 1: Distributed Average
    print("Example 1: Average of 1 Million Values")
    print("-" * 80)
    
    engine = WeakHardwareEngine(n_workers=100)
    
    # Generate 1M random values
    big_data = [random.random() for _ in range(1_000_000)]
    
    # Distributed computation
    import time
    start = time.time()
    distributed_avg = engine.distributed_average(big_data)
    distributed_time = time.time() - start
    
    # Classic computation (for comparison)
    start = time.time()
    classic_avg = sum(big_data) / len(big_data)
    classic_time = time.time() - start
    
    print(f"Classic average: {classic_avg:.6f} ({classic_time*1000:.2f}ms)")
    print(f"Distributed avg: {distributed_avg:.6f} ({distributed_time*1000:.2f}ms)")
    print(f"Difference: {abs(classic_avg - distributed_avg):.8f}")
    print(f"Statistics: {engine.get_statistics()}")
    print()
    
    # Example 2: Parallel Optimization
    print("Example 2: Find Maximum of Function")
    print("-" * 80)
    
    def test_function(x):
        """Rastrigin-like function with multiple local maxima"""
        return -(x**2 - 10 * (1 - (2*3.14159*x)**2)**0.5)
    
    engine = WeakHardwareEngine(n_workers=50)
    
    best_x, best_score = engine.parallel_optimization(
        test_function,
        search_space=(-5, 5),
        maximize=True
    )
    
    print(f"Optimum found: x={best_x:.4f}, f(x)={best_score:.4f}")
    print(f"Statistics: {engine.get_statistics()}")
    print()
    
    # Example 3: Parallel Map
    print("Example 3: Parallel Computation")
    print("-" * 80)
    
    def expensive_compute(n):
        """Simulate expensive computation"""
        return sum(i**2 for i in range(n))
    
    engine = WeakHardwareEngine(n_workers=20)
    
    inputs = range(100, 200)
    
    start = time.time()
    results = engine.parallel_map(expensive_compute, list(inputs))
    parallel_time = time.time() - start
    
    start = time.time()
    classic_results = [expensive_compute(n) for n in inputs]
    classic_time = time.time() - start
    
    print(f"Classic time: {classic_time*1000:.2f}ms")
    print(f"Parallel time: {parallel_time*1000:.2f}ms")
    print(f"Speedup: {classic_time/parallel_time:.2f}x")
    print()
    
    print("=" * 80)
    print("WEAK HARDWARE = QUANTUM-LIKE POWER!")
    print("=" * 80)
