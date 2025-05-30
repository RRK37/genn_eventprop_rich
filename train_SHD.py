#!/usr/bin/env python3

from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists


# Training parameters 
p["EVALUATION"] = "speaker"
p["N_EPOCH"] = 2
p["MIN_EPOCH_ETA_FIXED"] = 300
p["BALANCE_TRAIN_CLASSES"]= True
p["BALANCE_EVAL_CLASSES"]= True
p["TRAIN_DATA_SEED"] = 123
p["TEST_DATA_SEED"] = 321
p["TRIAL_MS"] = 1000.0
p["DT_MS"] = 0.1
p["AUGMENTATION"]= {
    "NORMALISE_SPIKE_NUMBER": True,
    "random_shift": 40.0,
    "blend": [0.5,0.5]
}
p["N_INPUT_DELAY"] = 10
p["INPUT_DELAY"] = 30

# Network parameters
p["ETA"] = 0.001
p["N_HID_LAYER"] = 1
p["NUM_HIDDEN"] = 1024
p["RECURRENT"] = True
p["TAU_MEM"] = 20
p["TAU_SYN"] = 5
p["INPUT_HIDDEN_MEAN"] = 0.03
p["INPUT_HIDDEN_STD"] = 0.01
p["HIDDEN_HIDDEN_MEAN"] = 0
p["HIDDEN_HIDDEN_STD"] = 0.02 
p["HIDDEN_OUTPUT_MEAN"] = 0 
p["HIDDEN_OUTPUT_STD"] = 0.03
p["PDROP_INPUT"] = 0
p["NU_UPPER"] = 14
p["REG_TYPE"] = "simple"

# p["GLB_UPPER"] = 10^(-9)

# Training parameters
p["LOSS_TYPE"] = "sum_weigh_exp"
p["TAU_0"] = 1
p["TAU_1"] = 100
p["ALPHA"] = 5*10^(-5)
p["N_BATCH"]= 64

# Recording parameters
# p["OUT_DIR"] = "experimental_recording_4"
p["OUT_DIR"] = "table_5_values_new_loss_bell_0_-02_44_04"

# p["DEBUG"]= True

p["REC_SPIKES"]= ["input","hidden0","output"]
p["REC_SPIKES_EPOCH_TRIAL"]= [[1,1]]
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])

# p["W_OUTPUT_EPOCH_TRIAL"] = [[0,25],[0,26],[0,27],[0,28],[0,29],[0,30],[0,31],[1,25],[1,26],[1,27],[1,28],[1,29],[1,30],[1,31]]


#p["REC_NEURONS"] = [("input","in_neuron"), ("hidden","hid_neuron"), ("output","out_neuron")]
#p["REC_NEURONS_EPOCH_TRIAL"] = [1,1]
#p["REC_SYNAPSES"] = [("in_to_hid","in_hi_synapse")]#, "hid_to_out"]
#p["REC_SYNAPSES_EPOCH_TRIAL"] = [1,7]
#p["W_OUTPUT_EPOCH_TRIAL"] = [1,7]
#p["TAU_OUTPUT_EPOCH_TRIAL"] = [1,7]

p["BUILD"] = True

jname= os.path.join(p["OUT_DIR"], p["NAME"]+".json")
jfile= open(jname,'w')
json.dump(p,jfile)
print(p)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
rname= os.path.join(p["OUT_DIR"], p["NAME"]+'.summary.txt')
sumfile= open(rname,'w')
sumfile.write("Training correct: {}, Valuation correct: {}".format(correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))
