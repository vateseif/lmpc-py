from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="lmpc",
    version="0.0.1",
    description="Multi agent control with communication",
    packages=["lmpc", "mpe"],
    package_dir={'':'src'},
    install_requires=requirements
)