from setuptools import setup, find_packages

setup(
    name='winter_corrections',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'winter_corrections=winter_corrections.example:example_function',
        ],
    },
)
