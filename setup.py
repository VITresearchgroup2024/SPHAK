from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path

# Try reading requirements.txt
try:
    requirements = Path("requirements.txt").read_text().splitlines()
except FileNotFoundError:
    requirements = []

# Custom install command to show message
class CustomInstallCommand(install):
    def run(self):
        super().run()
        print("\n" + "="*50)
        print("✅ Successfully installed SPHAK!")
        print("="*50 + "\n")

setup(
    name='sphak',
    version='0.1',
    description='SPHAK: Sequence-based Prediction of Host Analysis using k-mers',
    author='Vibin Ipe Thomas, Vinni N G, Ananya Prakash, Kavya S',
    author_email='vibin@cmscollege.ac.in',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'sphak': ['data/*.pkl'],
    },
    entry_points={
        'console_scripts': [
            'sphak=sphak.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    cmdclass={
        'install': CustomInstallCommand,
    }
)
