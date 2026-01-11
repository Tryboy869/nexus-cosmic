"""
Nexus-Cosmic Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="nexus-cosmic",
    version="1.0.0",
    author="Daouda Abdoul Anzize",
    author_email="nexusstudio100@gmail.com",
    description="Universal Emergent Computation Engine - Distributed computing based on emergent physics principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tryboy869/nexus-cosmic",
    project_urls={
        "Bug Tracker": "https://github.com/Tryboy869/nexus-cosmic/issues",
        "Documentation": "https://tryboy869.github.io/nexus-cosmic",
        "Source Code": "https://github.com/Tryboy869/nexus-cosmic",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "numpy": [
            "numpy>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nexus-cosmic=nexus_cosmic.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="distributed-computing emergent-systems consensus sorting optimization weak-hardware",
)
