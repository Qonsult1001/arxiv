#!/usr/bin/env python3
"""
LAM - Linear Associated Memory
==============================

Drop-in replacement for sentence-transformers with O(n) linear complexity.

Installation:
    pip install lam
    
Usage:
    from lam import LAM
    
    model = LAM('LAM-base-v1')
    embeddings = model.encode(['Hello world', 'How are you?'])
"""

from setuptools import setup, find_packages

setup(
    name="lam",
    version="1.0.0",
    author="LAM Research",
    author_email="research@lam.ai",
    description="Linear Associated Memory - Drop-in replacement for sentence-transformers",
    long_description=open("LAM-base-v1/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lam-research/lam",
    packages=find_packages(),
    package_data={
        "LAM-base-v1": [
            "*.json",
            "*.txt",
            "*.bin",
            "*.md",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "numpy",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="sentence-transformers embeddings nlp machine-learning linear-attention",
)


