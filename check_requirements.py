# check if the requried the packages are installed
import os
import importlib


# installation of the required packages
required_modules = [('numpy',
                     'conda install -y numpy'),
                    ('scipy',
                     'conda install -y scipy'),
                    ('ipopt',
                     'conda install -y -c conda-forge cyipopt')]

def check_module(module_name, install_command):
    try:
        importlib.import_module(module_name)
    except:
        os.system(install_command)

for module_name, install_command in required_modules:
    check_module(module_name, install_command)

# create the library of blas and lapack
os.mkdir("./lib")
os.system("cp -L $CONDA_PREFIX/lib/libblas.so ./lib")
os.system("cp -L $CONDA_PREFIX/lib/libblas.so.3 ./lib")
os.system("cp -L $CONDA_PREFIX/lib/liblapack.so ./lib")
os.system("cp -L $CONDA_PREFIX/lib/liblapack.so.3 ./lib")