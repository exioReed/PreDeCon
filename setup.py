#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fp:
    long_description = fp.read()

setuptools.setup(
    name='predecon',
    version='0.1',
    author='Maximilian Kaulmann',
    author_email='exioreed@ownlink.eu',
    description='PreDeCon - An Implementation in Python, Compatible With Scikit-Learn',
    packages=setuptools.find_packages(),
    install_requires=[
        'joblib>=0.14.0',
        'numpy>=1.15.4',
        'scikit-learn>=0.22.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.6',
)