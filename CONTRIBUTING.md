# Contributing to Nexus-Cosmic

Thank you for your interest in contributing to Nexus-Cosmic!

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS)

### Suggesting Features

Feature suggestions are welcome! Please create an issue describing:
- The feature and its use case
- Why it would be useful
- Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python tests/test_all.py`)
5. Commit with clear message (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8
- Add docstrings to functions/classes
- Include type hints where appropriate
- Write tests for new features

### Testing

All contributions must pass existing tests:

```bash
python tests/test_all.py
```

### Validation for Custom Laws

If adding a custom law, validate it:

```python
from nexus_cosmic import validate_law

results = validate_law(YourLaw(), n_runs=5)
print(results['verdict'])  # Should be ✅ VALIDÉ or ⚠️ EXPERIMENTAL
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/Tryboy869/nexus-cosmic.git
cd nexus-cosmic

# Install in development mode
pip install -e .

# Run tests
python tests/test_all.py
```

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Questions?

Open an issue or contact: nexusstudio100@gmail.com

---

**Thank you for contributing to Nexus-Cosmic!**
