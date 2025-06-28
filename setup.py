from setuptools import setup, find_packages

setup(
    name="plicp_slam",  
    version="0.0.1", 
    packages=find_packages(),  
    install_requires=[
        "numpy>=1.26.4", 
        "g2o-python>=0.0.12",
        "scipy>=1.14.1",
        "scikit-image>=0.25.0"
    ],
    python_requires=">=3.6",
)