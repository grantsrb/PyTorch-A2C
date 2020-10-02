from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='a2c',
      packages=find_packages(),
      version="0.1.0",
      description='A project that implements A2C',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/PyTorch-A2C.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm"],
      py_modules=['a2c'],
      long_description='''
            A project that implements Advantage Actor Critic using
            multiple processes.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
