from distutils.core import setup
from setuptools import find_packages

setup(
    name='toto_benchmark',
    version='0.0.1',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
)