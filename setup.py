from setuptools import setup, find_packages

setup(
    name="pyspawn",
    version="1.0.0",
    description="Full Multiple Spawning in Python",
    author="Benjamin G. Levine",
    url="https://github.com/blevine37/pySpawn17",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21,<1.24",
        "h5py>=3.0,<4",
    ],
)