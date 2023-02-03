import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from shared_utils.general import get_project_root


matplotlib.use('TkAgg')

data_filename = "run1.csv"
data_path = os.path.join(get_project_root(), "data",
                         "4_dofs_eval", data_filename)
assert os.path.isfile(data_path)

df = pd.read_csv(data_path, sep=",")
torque_data = df.filter(regex="Torque$").to_numpy()

time = np.arange(len(torque_data))*0.01

plt.plot(time, torque_data[:, 0])
plt.show()
