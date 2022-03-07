cetra_project
==============================


conda update -n base -c defaults conda


# how to create developing environment [env]

conda env create -f cetra-environment.yml 
NOTE: it may take some time to solve all the rependencies!

conda env remove -n  cetra-env


conda activate cetra-env
cd cetra_code
pip install -e .


conda activate kg-env
conda install tensorflow-gpu==1.14

conda install tensorflow==1.15.0
conda install tensorflow==1.14
conda install tensorflow-gpu==1.13.1



 anaconda-navigator

conda activate ampligraph

# install AmpliGraph

cd AmpliGraph

pip install -e .



# Start Tensorboard

conda activate kg-env
tensorboard --logdir=./visualizations

tensorboard --logdir=./GoT_embeddings




# on server
conda activate kg-env ; spyder --new-instance &


# Extra
conda update conda
conda update anaconda
conda update python
conda update --all

conda clean --all





# Tips for developers
pip install pep8
pip install pylint


advice in 2020, to use Visual Studio Code

Install  Python support


VS Code Quick Open (Ctrl+P)

ext install ms-python.python


