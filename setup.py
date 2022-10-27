from setuptools import setup

setup(name='limetr',
      version='0.0.2',
      description='linear mixed effects model with trimming',
      url='https://github.com/zhengp0/limetr',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['limetr'],
      package_dir={'limetr': 'src/limetr'},
      install_requires=['numpy', 'scipy', 'ipopt', 'spmat'],
      zip_safe=False)
