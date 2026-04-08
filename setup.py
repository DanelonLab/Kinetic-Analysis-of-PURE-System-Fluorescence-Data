from setuptools import setup, find_packages

setup(
    name="pure_kinetics",
    version="1.0.0",
    description="Kinetic analysis of PURE system fluorescence time-course data",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "openpyxl",
    ],
)
