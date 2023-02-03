def add_joint_pos(sim_data, pos_list, joint_name):
    pos_list.append(sim_data.get_joint_qpos(joint_name))


def add_joint_vel(sim_data, vel_list, joint_name):
    vel_list.append(sim_data.get_joint_qvel(joint_name))


def add_musc_force(sim_data, force_list, joint_index):
    force_list.append(sim_data.qfrc_actuator[joint_index])
