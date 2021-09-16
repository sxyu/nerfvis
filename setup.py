import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

__version__ = None
exec(open('nerfvis/version.py', 'r').read())

setup(
    name='nerfvis',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='NeRF visualization library',
    long_description='NeRF visualization library based on PlenOctrees. See https://github.com/sxyu/nerfvis',
    packages=['nerfvis'],
    include_package_data=True,
    package_data={'nerfvis': ['volrend.zip']},
)
