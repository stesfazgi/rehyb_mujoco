# Set up Server

This ReadMe shows how to run the Monte Carlo sampler on the LSR-ITR server from scratch.

## Clone repository

Run from home folder `~`:

> ~$ git clone https://github.com/ronansgd/rehyb_mujoco.git

## Install MuJoCo

Create dedicated folder:

> \$ mkdir ~/.mujoco\

Download and unpack:

> \$ cd ~/.mujoco
> \$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
> \$ tar -xf mujoco210-linux-x86_64.tar.gz

Download license key:

> $ wget https://www.roboti.us/file/mjkey.txt

## Add environment variables required by the project

Initialize `.bashrc`:

> \$ touch ~/.bashrc

Append the following lines to the .bashrc:

> export REHYB_MUJOCO_PATH=/home/rsangouard/rehyb_mujoco\
> export MUJOCO_BIN_PATH=/home/rsangouard/.mujoco/mujoco210/bin\
> export PYTHONPATH=$REHYB_MUJOCO_PATH/src_py\

## Define useful notebook command alias

Append the following line to the .bashrc:

> alias nbx='jupyter nbconvert --execute --to notebook --inplace'

## Add environment variables required by mujoco-py

Append the following lines to the .bashrc:

> export MUJOCO_PY_MUJOCO_PATH=/home/rsangouard/.mujoco/mujoco210\
> export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$MUJOCO_BIN_PATH\
> export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia

Source .bashrc:

> \$ source ~/.bashrc

NB: Everytime you open a new terminal/modify the .bashrc, you have to source the `.bashrc` file

## Install patchelf

Download and install patchelf via GitHub:

> \$ cd ~/ && git clone https://github.com/NixOS/patchelf.git \
> \$ cd patchelf && ./bootstrap.sh\
> \$ ./configure --prefix=$HOME/.local\
> \$ make\
> \$ make install\
> \$ rm -rf ~/patchelf

Add the following line to your .bashrc. Replace <user> with your user name!

> export PATH=$PATH:/home/<user>/.local/bin

Source .bashrc:

> \$ source ~/.bashrc

## Create Python virtual env with conda

> \$ conda create --name rehyb_mujoco_py --file requirements.txt

## Generate the arm supports

Don't forget to activate the environment and source the .bashrc beforehand!

> \$ cd ~/rehyb_mujoco/src_py/xml_generation/arm_support/\
> \$ nbx generate_arm_support.ipynb

NB: The alias `nbx` was created previously and should be in your .bashrc file.

## Run Monte Carlo sampling

> \$ cd ~/rehyb_mujoco/src_py/monte_carlo/\
> \$ python sample_models.py

If you see no error, you are done with this tutorial!

## Credits

Credits to Martin Schuck for the section related to patchelf installation.
