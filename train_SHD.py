#!/usr/bin/env python3

from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists


p["EVALUATION"] = "speaker"
p["N_EPOCH"] = 200
p["LOSS_TYPE"] = "sum_weigh_exp"
p["MIN_EPOCH_ETA_FIXED"] = 100
p["BALANCE_TRAIN_CLASSES"]= True
p["BALANCE_EVAL_CLASSES"]= True

p["TRAIN_DATA_SEED"] = 123
p["TEST_DATA_SEED"] = 321
p["REG_TYPE"] = "simple"
p["PDROP_INPUT"] = 0.1
p["TRIAL_MS"] = 1400.0
p["AUGMENTATION"]= {
    "NORMALISE_SPIKE_NUMBER": True,
    "random_shift": 40.0,
    "blend": [0.5,0.5]
}
p["N_INPUT_DELAY"] = 10
p["INPUT_DELAY"] = 30

p["N_HID_LAYER"] = 1
p["NUM_HIDDEN"] = 1024
p["RECURRENT"] = True
p["HIDDEN_OUTPUT_MEAN"] = 0 
p["HIDDEN_OUTPUT_STD"] = 0.03
p["HIDDEN_HIDDEN_MEAN"] = 0
p["HIDDEN_HIDDEN_STD"] = 0.03 

p["BUILD"] = True

p["OUT_DIR"] = "recording_neurons"


p["REC_SPIKES"] = ["input"]#,"hidden","output"]
p["REC_SPIKES_EPOCH_TRIAL"] = [99,7]
p["REC_NEURONS"] = [("input","in_neuron")]#, "hidden", "output"]
p["REC_NEURONS_EPOCH_TRIAL"] = [99,7]
p["REC_SYNAPSES"] = [("in_to_hid","in_hi_synapse")]#, "hid_to_out"]
p["REC_SYNAPSES_EPOCH_TRIAL"] = [99,7]
p["W_OUTPUT_EPOCH_TRIAL"] = [99,7]
p["TAU_OUTPUT_EPOCH_TRIAL"] = [99,7]

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
