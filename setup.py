from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# fortran extension module
ext = Extension(name='limetr.futils',
                sources=['limetr/utils.f90'],
                library_dirs=['/usr/local/lib'],
                libraries=['lapack', 'blas'])

setup(name='limetr',
      version='0.0.0',
      description='linear mixed effects model with trimming',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['limetr'],
      ext_modules=[ext],
      install_requires=['numpy', 'scipy', 'ipopt'],
      zip_safe=False)
