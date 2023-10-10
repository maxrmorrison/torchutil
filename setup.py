from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='torchutil',
    description='PyTorch utilities for developing deep learning frameworks',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/torchutil',
    extras_require={
        'train': [
            'apprise',
            'accelerate',
            'matplotlib'
        ]
    },
    install_requires=['gdown', 'torch'],
    packages=['torchutil'],
    package_data={'torchutil': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pytorch', 'utility', 'training'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
