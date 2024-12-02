import json
import numpy as np
from matplotlib import pyplot as plt

#Name of file represents the results of loss while training with the number of layers which are representesd in name of file
with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data128.json", mode="r", encoding="utf-8") as f1:
    data = json.load(f1)
    data_array1 = np.array(data["loss"])

# x axis is epoch, y axis is loss value
fig = plt.figure()
ax = plt.axes(xlim=(0.0, 1000.0), ylim=(0.0, 0.2))
line1 = ax.plot(range(1000), data_array1, lw=3)

plt.show()

f1.close()