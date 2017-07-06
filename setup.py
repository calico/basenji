from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='basenji',
    version='0.0.1',
    description='Sequential regulatory activity machine learning',
    long_description=readme,
    author='David Kelley',
    author_email='drk@calicolabs.com',
    url='https://github.com/calico/basenji',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['h5py','joblib','matplotlib','numpy','pandas','pyBigWig','pysam','scipy','seaborn','sklearn','statsmodels','tensorflow']
)
