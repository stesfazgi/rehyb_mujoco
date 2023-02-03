# Setup guide

## I. Linux

This solution was tested on Ubuntu 20.04 LTS.

### A. Install and set up MuJoCo

1. Go to https://www.roboti.us/index.html, download `mujoco210` and extract it into `~/.mujoco/mujoco210`:

   > \$ mkdir ~/.mujoco\
   > \$ cd ~/.mujoco\
   > ~/.mujoco$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
   > ~/.mujoco$ tar -xf mujoco210-linux-x86_64.tar.gz

2. Place `mjkey.txt` in the folder `~/.mujoco`.
   > \$ wget https://www.roboti.us/file/mjkey.txt && mv mjkey.txt ~/.mujoco
3. (Optional but recommended) Compile MuJoCo code samples.
   > ~/.mujoco/mujoco210/sample$ make
4. (Test) Run the built-in 'simulate' executable on the 'humanoid.xml'. A simulation of an orange humanoid should appear!
   > ~/path/to/mujoco210/bin$ ./simulate ../model/humanoid.xml

### B. Clone rehyb_mujoco and carry out a first test

1. Clone the project on your device.
   > ~/path/to/desired_folder$ git clone https://github.com/ronansgd/rehyb_mujoco.git
2. (Test) Simulate 'setup_guide/model_1.xml' from MuJoCo's bin. If you see no error and a simulation of the exoskeleton appears, you are on the good track!
   > ~/.mujoco/mujoco210/bin$ ./simulate relative/path/to/rehyb_mujoco/setup_guide/model_1.xml

### C. Configurate `simulate.sh` (Highly recommended)

On the long run, it may be tedious to type the relative paths to your models from the directory of MuJoCo's `simulate` executable.

Hopefully, `simulate.sh` is there to do this job for you! You just have to provide the path to your model relatively to the root folder of the project.

1. Add the following lines to your `.bashrc` file (Don't forget to adapt the paths!):

   > export REHYB_MUJOCO_PATH=/absolute/path/to/rehyb_mujoco \
   > export MUJOCO_BIN_PATH=/absolute/path/to/mujoco210/bin

2. Make `simulate.sh` executable.
   > ~/path/to/rehyb_mujoco$ chmod +x simulate.sh
3. (Test) Simulate `setup_guide/model_1.xml` using `simulate.sh`. You should see the same simulation that was launched at step B.2.!
   > ~/path/to/rehyb_mujoco$ ./simulate.sh setup_guide/model_1.xml

### D. Set up the python environment

A specific python environment needs to be set up for this project. The documented procedure of installation relies on `virtualenv` and `virtualenvwrapper`. The XML generation option of `simulate.sh` also relies on those two libraries.

#### Recommended way (with virtualenvwrapper)

1. Install `virtualenv` and `virtualenvwrapper` if they are not yet installed on your device.

> \$ pip install virtualenv\
> $ pip install virtualenvwrapper

2. Add the following lines to your `.bashrc` (the `WORKON_HOME` directory will store the environments).

> export VIRTUALENVWRAPPER_PYTHON=$(which python3) \
> export WORKON_HOME=$HOME/.virtualenvs \
> export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv \
> export VIRTUALENVWRAPPER_SH=~/.local/bin/virtualenvwrapper.sh \
> source $VIRTUALENVWRAPPER_SH

3. Create a new environment with all required libraries using `requirements.txt`. A new folder named 'rehyb_mujoco_py' should be created in the `WORKON_HOME` folder.

> ~/path/to/rehyb_mujoco$ mkvirtualenv -r requirements.txt rehyb_mujoco_py

4. You can easily activate ("work on") and deactivate this new environment.

> \$ workon rehyb_mujoco_py \
> (rehyb_mujoco_py) $ deactivate

#### Alternative methods:

A minimal yaml file is provided to create the environment with `conda` (only conda is available on LSR-ITR remote server). However, it has not yet been tested extensively on the project code.

### E. Set environment variables used by the python packages of this project

1. Add the following line to your `.bashrc` to allow relative imports between packages (Don't forget to adapt the path!):

> export PYTHONPATH=$REHYB_MUJOCO_PATH/src_py

2. (Test) Generate the arm supports of the exoskeleton by executing the notebook `src_py/xml_generation/arm_support/generate_arm_support.ipynb` (Don't forget to activate the environment beforehand!). \
   Then, simulate `set-up/model_2.xml`.
   Model 2 is very similar to model 1. However, you may notice additional geoms around the meshes of the arm supports: they have been generated by the notebook (the generated XML files should be located in the `bin/models/setup_guide` folder).

> ~/path/to/rehyb_mujoco$ ./simulate.sh setup_guide/model_2.xml

### F. Set environment variables required by mujoco-py

1. Add the following lines to your `.bashrc` file (Again don't forget to adapt the paths!).

> export MUJOCO_PY_MUJOCO_PATH=/absolute/path/to/mujoco210 \
> export MUJOCO_PY_MJKEY_PATH=/absolute/path/to/mjkey.txt \
> export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$MUJOCO_BIN_PATH

2. (Test) Run the demonstration script located in `setup_guide/demo_grasping` (Again remember to activate the environment!). \
   After a while, if you see a hand trying to catch an horizontal bar in a very robotic style, then you've succeeded in setting up your environment, congratulations!

> (rehyb_mujoco_py) ~/path/to/rehyb_mujoco/setup_guide/demo_grasping$ python demo_script.py

### G. Simulate `nesm_with_patient.xacro`

`nesm_with_patient.xacro` is the main model of this project. In this model, the patient and the exoskeleton are interacting.

It is written in a `.xacro` file, and not a `.xml` file (see [xacro documentation](http://wiki.ros.org/xacro)). Hence, it cannot be directly simulated; it has to be first parsed into a `.xml` file. To do so, use the `xacro.sh` script located at the root:

> ~/path/to/rehyb_mujoco$ chmod +x xacro.sh \
> ~/path/to/rehyb_mujoco$ ./xacro.sh models/exo_with_patient/nesm_with_patient.xacro

The script generates a new file located at `bin/models/exo_with_patient/nesm_with_patient.xml`.

We are not yet ready to simulate the model. Indeed, similarly to point [E.2.](#e-set-environment-variables-used-by-the-python-packages-of-this-project), we need to generate the arm support files. To do so, execute the last cells of the `src_py/xml_generation/arm_support/generate_arm_support.ipynb` notebook. Two files should be generated at `bin/models/arm_supports/upper_arm_support.xml` and `bin/models/arm_supports/lower_arm_support.xml` (it is possible that you had already generated them at step E.2. if you executed the whole notebook).

You are now able to simulate the model:

> ~/path/to/rehyb_mujoco$ ./simulate.sh bin/models/exo_with_patient/nesm_with_patient.xml

Congratulations, you reached the end of this tutorial! You can go further by reading the READMEs of the different subfolders :)

## Additional resources for trouble shooting:

- MuJoCo Documentation: https://mujoco.readthedocs.io/en/latest/overview.html \
- virtualenvwrapper Documentation: https://virtualenvwrapper.readthedocs.io/en/latest/ \
- mujoco-py Documentation: https://github.com/openai/mujoco-py \
- The almighty Google