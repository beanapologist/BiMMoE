from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qdt-bimmoe",
    version="1.0.0",
    author="QDT Research Team",
    author_email="research@qdt-framework.org",
    description="Quantum Duality Theory (QDT) Bidirectional Multi-Modal Multi-Expert Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beanapologist/BiMMoE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    keywords="quantum, multi-modal, machine learning, physics, energy, tokenization",
    project_urls={
        "Bug Reports": "https://github.com/beanapologist/BiMMoE/issues",
        "Source": "https://github.com/beanapologist/BiMMoE",
        "Documentation": "https://github.com/beanapologist/BiMMoE#readme",
    },
) 