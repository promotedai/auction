from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='auction',
    version='0.1.0',
    description='Ad auction in Python',
    long_description=readme,
    author='Andrew Yates',
    author_email='ayates@promoted.ai',
    url='https://github.com/promotedai/auction',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
