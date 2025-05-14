#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

print("Hello")

data = np.load("experimental_recording_4/test_input_spike_ID.npy")

print(data)
print(len(data))


# Plot as a line graph
#plt.figure(figsize=(10, 6))
#plt.plot(data)
#plt.title('Synaptic Weights')
#plt.xlabel('Index')
#plt.ylabel('Weight')
#plt.show()
