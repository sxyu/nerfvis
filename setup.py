import logging
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import numpy
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, SystemExit)

__version__ = None
exec(open('nerfvis/version.py', 'r').read())

logging.basicConfig()
log = logging.getLogger(__file__)

cython_args = {}
cython_args["ext_modules"] = [Extension(
                                "nerfvis.utils._rotation",
                                ["nerfvis/utils/_rotation.c"],
                                include_dirs=[numpy.get_include()],
                            )]
cython_args["cmdclass"] = {'build_ext': build_ext}

try:
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
        **cython_args,
    )
except ext_errors as ex:
    log.warn(ex)
    log.warn("The C extension could not be compiled")
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
        cmdclass = {}
    )
