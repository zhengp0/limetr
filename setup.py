from pathlib import Path
from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

# fortran extension module
ext = Extension(name='limetr.special_mat',
                sources=['src/limetr/special_mat.f90'],
                library_dirs=['./lib'],
                libraries=['lapack', 'blas'])


if __name__ == '__main__':
    base_dir = Path(__file__).parent
    src_dir = base_dir/'src'

    about = {}
    with (src_dir/'limetr'/'__about__.py').open() as f:
        exec(f.read(), about)

    with (base_dir/'README.md').open() as f:
        long_description = f.read()

    install_requirements = ['numpy', 'scipy']
    test_requirements = ['pytest']
    doc_requirements = []
    unresolved_requirements = ['ipopt']

    setup(name=about['__title__'],
          version=about['__version__'],

          description=about['__summary__'],
          long_description=long_description,
          license=about['__license__'],
          url=about['__uri__'],

          author=about['__author__'],
          author_email=about['__email__'],

          package_dir={'': 'src'},
          packages=find_packages(where='src'),
          include_package_data=True,

          install_requires=install_requirements,
          tests_require=test_requirements,
          extras_require={
              'docs': doc_requirements,
              'test': test_requirements,
              'dev': doc_requirements + test_requirements
          },
          ext_modules=[ext],
          zip_safe=False)
