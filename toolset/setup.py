from distutils.core import setup
from setuptools import find_packages

setup(
    name='manip_dataset_toolset',
    version='1.0.0',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
)