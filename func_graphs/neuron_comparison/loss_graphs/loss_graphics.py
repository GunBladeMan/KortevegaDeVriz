import json
import numpy as np
from matplotlib import pyplot as plt

#Name of file represents the results of loss while training with the number of layers which are representesd in name of file
with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data8.json", mode="r", encoding="utf-8") as f1:
    data = json.load(f1)
    data_array1 = np.array(data["loss"])

with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data16.json", mode="r", encoding="utf-8") as f2:
    data = json.load(f2)
    data_array2 = np.array(data["loss"])

with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data32.json", mode="r", encoding="utf-8") as f3:
    data = json.load(f3)
    data_array3 = np.array(data["loss"])

with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data64.json", mode="r", encoding="utf-8") as f4:
    data = json.load(f4)
    data_array4 = np.array(data["loss"])

with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data96.json", mode="r", encoding="utf-8") as f5:
    data = json.load(f5)
    data_array5 = np.array(data["loss"])

with open("/func_graphs/neuron_comparison/loss_graphs/json/loss_data128.json", mode="r", encoding="utf-8") as f6:
    data = json.load(f6)
    data_array6 = np.array(data["loss"])

# x axis is epoch, y axis is loss value
fig = plt.figure()
ax = plt.axes(xlim=(0.0, 1000.0), ylim=(0.0, 0.2))
line1 = ax.plot(range(1000), data_array1, lw=3)
line2 = ax.plot(range(1000), data_array2, lw=3)
line3 = ax.plot(range(1000), data_array3, lw=3)
line4 = ax.plot(range(1000), data_array4, lw=3)
line5 = ax.plot(range(1000), data_array5, lw=3)
line6 = ax.plot(range(1000), data_array6, lw=3)

plt.show()

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()