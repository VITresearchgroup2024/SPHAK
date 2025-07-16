from setuptools import setup, find_packages
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name='sphak',
    version='0.1',
    description='SPHAK: Sequence-based Prediction of Host Analyis of k-mers',
    author='Vibin Ipe Thomas, Vinni N G, Ananya Prakash, Kavya S'
    author_email='vibin@cmscollege.ac.in, inning372@gmail.com , ananyaprakash0105@gmail.co, kavyasree6424@gmail.com'
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sphak=cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', 
        'Operating System :: OS Independent',
    ],
)

