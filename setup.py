from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='torchutil',
    description='PyTorch utilities for developing deep learning frameworks',
    version='0.0.5',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torchutil',
    install_requires=['accelerate', 'apprise', 'torch'],
    packages=find_packages(),
    package_data={'torchutil': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pytorch', 'utility', 'training'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
