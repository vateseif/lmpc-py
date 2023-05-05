# lmpc-py

## Setup


### Create env
Download anaconda and create the new enviroment
~~~
conda create --name lmpc python=3.9
~~~


### Activate env
You need to activate the environment and install all requierements
~~~
conda activate lmpc
pip install -r requirements.txt
~~~


### Setup pip
This will install your project and its dependencies in the current Python environment. 
~~~
python setup.py sdist bdist_wheel
pip install .
~~~
If you make any changes to your project, you can run `pip install .` again to reinstall it.


### Install Mosek
Follow [this](https://docs.mosek.com/latest/install/installation.html) tutorial for the installation of Mosek.