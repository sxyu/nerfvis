import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


setup(
    name='nerfvis',
    version='0.0.1',
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='Web NeRF viewer',
    long_description='',
    packages=['nerfvis'],
    include_package_data=True,
    package_data={'nerfvis': ['volrend.zip']},
)
