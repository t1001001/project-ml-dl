from setuptools import setup, find_packages

setup(
    name="project_ml_dl",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
