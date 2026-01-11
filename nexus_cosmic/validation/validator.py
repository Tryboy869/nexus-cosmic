"""
Validation Tools

Validate custom laws, topologies, and configurations
"""

from typing import Dict, Any
from nexus_cosmic.core.engine import NexusCosmic
from nexus_cosmic.core.base import CustomLaw


def validate_law(law: CustomLaw,
                 n_entities: int = 30,
                 n_runs: int = 5,
                 max_steps: int = 100,
                 convergence_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Validate a custom force law
    
    Tests law for:
    - Convergence success rate
    - Average convergence time
    - Final variance
    
    Args:
        law: CustomLaw instance to validate
        n_entities: Number of entities for test
        n_runs: Number of test runs
        max_steps: Maximum steps per run
        convergence_threshold: Variance threshold
    
    Returns:
        Dict with validation results:
            - success_rate: Percentage of successful runs
            - avg_steps: Average convergence steps
            - avg_variance: Average final variance
            - verdict: '✅ VALIDÉ', '⚠️ EXPERIMENTAL', or '❌ REJETÉ'
            - details: List of individual run results
    
    Example:
        >>> class MyLaw(CustomLaw):
        ...     def compute_force(self, e1, e2):
        ...         return (e2.state - e1.state) * 0.1
        >>> results = validate_law(MyLaw())
        >>> print(results['verdict'])
        ✅ VALIDÉ
    """
    convergence_steps = []
    final_variances = []
    success_count = 0
    run_details = []
    
    for run in range(n_runs):
        system = NexusCosmic(
            n_entities=n_entities,
            force_law=law,
            topology='small_world'
        )
        
        result = system.run(max_steps=max_steps, convergence_threshold=convergence_threshold)
        
        convergence_steps.append(result['steps'])
        final_variances.append(result['final_variance'])
        
        success = result['converged'] and result['final_variance'] < convergence_threshold
        if success:
            success_count += 1
        
        run_details.append({
            'run': run + 1,
            'steps': result['steps'],
            'variance': result['final_variance'],
            'converged': result['converged'],
            'success': success
        })
    
    # Statistics
    avg_steps = sum(convergence_steps) / len(convergence_steps)
    avg_var = sum(final_variances) / len(final_variances)
    success_rate = (success_count / n_runs) * 100
    
    # Verdict
    if success_rate >= 80 and avg_steps <= 50 and avg_var < 1.0:
        verdict = "✅ VALIDÉ"
    elif success_rate >= 50:
        verdict = "⚠️ EXPERIMENTAL"
    else:
        verdict = "❌ REJETÉ"
    
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_variance': avg_var,
        'verdict': verdict,
        'details': run_details
    }


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary
    
    Args:
        config: Configuration dict
    
    Returns:
        Dict with validation results
    
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    warnings = []
    
    # Check required fields for mode
    mode = config.get('mode')
    if mode == 'consensus':
        if 'n_entities' not in config:
            errors.append("'n_entities' required for consensus mode")
    
    elif mode == 'sorting':
        if 'values' not in config:
            errors.append("'values' required for sorting mode")
    
    # Check momentum
    momentum = config.get('momentum', 0.8)
    if momentum < 0 or momentum >= 1:
        errors.append(f"momentum must be in [0, 1), got {momentum}")
    elif momentum > 0.95:
        warnings.append(f"High momentum ({momentum}) may cause slow convergence")
    
    # Check freeze params
    freeze_threshold = config.get('freeze_threshold', 0.01)
    if freeze_threshold < 0:
        errors.append(f"freeze_threshold must be positive, got {freeze_threshold}")
    elif freeze_threshold > 0.1:
        warnings.append(f"High freeze_threshold ({freeze_threshold}) may cause premature freezing")
    
    # Check topology
    topology = config.get('topology', 'small_world')
    valid_topologies = ['small_world', 'ring', 'grid', 'full']
    if isinstance(topology, str) and topology not in valid_topologies:
        warnings.append(f"Unknown topology '{topology}', valid: {valid_topologies}")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
