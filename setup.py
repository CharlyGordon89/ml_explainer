# setup.py

from setuptools import setup, find_packages

setup(
    name="ml_explainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "shap",
        "lime",
        "scikit-learn",
        "numpy",
        "pandas"
    ],
    author="Ruslan Mamedov",
    description="Reusable explanation module for SHAP, LIME, and permutation importance.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
