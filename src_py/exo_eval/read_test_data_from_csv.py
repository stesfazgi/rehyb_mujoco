'''
This file defines functions allowing to read data from the "20210208_EFE" experiment

"read_test_data_from_csv" returns all data as a tuple of numpy arrays
'''

import os
import re
import pandas as pd
import numpy as np

from shared_utils import get_project_root


def get_test_data_filename(test_name, test_number, data_folder="data/20210208_EFE"):
    abs_data_folder = os.path.join(get_project_root(), data_folder)
    assert os.path.isdir(abs_data_folder)

    data_filename = os.path.join(
        abs_data_folder, f"{test_name}{test_number}.csv")
    assert os.path.isfile(data_filename)

    return data_filename


def read_test_data_from_csv(test_name, test_number, data_folder="data/20210208_EFE"):
    '''
    Returns a tuple: (Time (N), JointAngles (4, N), DesiredPositions (4, N), ActualTorques (4, N),
        Currents (4, N), MotorCounts (4,N), SpringAngles (4,N))
    with:
        4 the number of actuators (S_AA, S_FE, S_IE, E_FE)
        N the number of data points
    '''

    df = pd.read_csv(get_test_data_filename(
        test_name, test_number, data_folder), sep=",", header=None).T

    # manually extracted: prone to bug
    df.columns = ["iteration",
                  "S_AA_DesiredTorque", "S_AA_SpringAngle", "S_AA_JointAngle", "S_AA_ActualTorque", "S_AA_SetCurrent",
                  "S_FE_DesiredTorque", "S_FE_SpringAngle", "S_FE_JointAngle", "S_FE_ActualTorque", "S_FE_SetCurrent",
                  "S_IE_DesiredTorque", "S_IE_SpringAngle", "S_IE_JointAngle", "S_IE_ActualTorque", "S_IE_SetCurrent",
                  "E_FE_DesiredTorque", "E_FE_SpringAngle", "E_FE_JointAngle", "E_FE_ActualTorque", "E_FE_SetCurrent",
                  "S_AA_DesiredPosition", "S_FE_DesiredPosition", "S_IE_DesiredPosition", "E_FE_DesiredPosition",
                  "S_AA_Gravity_Torque", "S_FE_Gravity_Torque", "S_IE_Gravity_Torque", "E_FE_Gravity_Torque",
                  "E_FE_MotorCounts", "S_IE_MotorCounts", "S_FE_MotorCounts", "S_AA_MotorCounts"
                  ]

    # convert iteration to time (100 Hz)
    df['Time'] = (df['iteration'] - df['iteration'][0]) * .01

    # convert unit of Currents col (tick to Ampere)
    regex_current = re.compile("^.+_SetCurrent$")
    for col_name in df.columns:
        if regex_current.match(col_name) is not None:
            # convert tick to Ampere
            df[col_name] = (2000 - df[col_name])/0.0625/1000

    # pack data
    returned_list = [df["Time"].to_numpy()]

    # stack columns with same suffix in prefix order
    # each stack is appended to the return list
    return_tuple_suffixes = ["_JointAngle", "_DesiredPosition",
                             "_ActualTorque", "_SetCurrent", "_MotorCounts", "_SpringAngle"]
    ordered_prefixes = ["S_AA", "S_FE", "S_IE", "E_FE"]

    for suffix in return_tuple_suffixes:
        measurement = np.zeros((0, df.shape[0]))
        for prefix in ordered_prefixes:
            measurement = np.vstack(
                [measurement, df[prefix+suffix].to_numpy()])

        returned_list.append(measurement)

    return tuple(returned_list)
