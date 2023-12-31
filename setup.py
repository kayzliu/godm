from setuptools import find_packages, setup

# read the contents of README file
from os import path

# get __version__ from _version.py
ver_file = path.join('godm', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(name='godm',
      version=__version__,
      description='GODM',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author='kayzliu',
      author_email='zliu234@uic.edu',
      url='https://github.com/kayzliu/godm',
      download_url='https://github.com/kayzliu/godm/archive/main.zip',
      keywords=['outlier detection', 'data augmentation', 'diffusion models',
                'graph neural networks', 'graph generative model'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements,
      setup_requires=['setuptools>=58.2.0'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Education',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only'
      ],
)