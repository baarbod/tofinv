# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='tofinv',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tofinv = tofinv.cli:main',
        ],
    }
)