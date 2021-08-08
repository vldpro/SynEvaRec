#!/usr/bin/env python
from setuptools import setup, find_packages


NAME = "modules"
DESCRIPTION = "modules"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = "0.1.0"
AUTHOR = "Vladimir Provalov"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "notebooks"]),
)