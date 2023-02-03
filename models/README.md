This folder contains all the MuJoCo models of this project.

Xml models can be simulated out of the box using the _simulate.sh_ script located at the project root:

> path/to/rehyb_mujoco\$ ./simulate.sh models/exoskeleton/exoskeleton_alone.xml

Xacro models must first be parsed using the _xacro.sh_ script located at the project root. The xml output should be saved in the _bin/models_ folder, where the structure of this folder is replicated. The output file can finally be simulated thanks to _simulate.sh_:

> path/to/rehyb_mujoco\$ ./xacro.sh models/exo_with_patient/nesm_with_patient_simple.xacro \
> path/to/rehyb_mujoco\$ ./simulate.sh bin/models/exo_with_patient/nesm_with_patient.xml
