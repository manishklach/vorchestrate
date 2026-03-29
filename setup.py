"""Setuptools configuration for vOrchestrate."""

from setuptools import find_packages, setup


setup(
    name="vorchestrate",
    version="0.1.0",
    author="Manish Lachwani",
    description="Predictive multi-tier weight residency orchestration for transformer inference",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
