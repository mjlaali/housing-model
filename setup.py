#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="housing-model",
    version="1.0",
    description="A toolkit to build models for predicting house prices.",
    author="Majid Laali",
    author_email="mjlaali@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
