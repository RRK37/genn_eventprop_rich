from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import exists

p["OUT_DIR"]= "test111"
p["TRIAL_MS"]= 140
p["TRAIN_DATA_SEED"]= 372
p["TEST_DATA_SEED"]= 814
p["MODEL_SEED"]= 135
p["NAME"]= "test114"
p["NUM_HIDDEN"]= 1024 #256
p["N_MAX_SPIKE"]= 1500
p["DT_MS"]= 1
p["PDROP_INPUT"]= 0.0
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   
p["DEBUG_HIDDEN_N"]= True
p["LOAD_LAST"]= False
p["N_EPOCH"]= 300
p["N_BATCH"]= 32
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 7644 
p["N_VALIDATE"]= 512 
p["ETA"]= 5e-3 
p["SHUFFLE"]= True
p["RECURRENT"]= True
p["INPUT_HIDDEN_MEAN"]= 0.2 
p["INPUT_HIDDEN_STD"]= 0.1 
p["HIDDEN_OUTPUT_MEAN"]= 0.0 
p["HIDDEN_OUTPUT_STD"]= 0.03 
p["HIDDEN_HIDDEN_MEAN"]= 0.0
p["HIDDEN_HIDDEN_STD"]= 0.02 # 0.02
p["HIDDEN_HIDDENFWD_MEAN"]= 0.04 # only used when > 1 hidden layer
p["HIDDEN_HIDDENFWD_STD"]= 0.01 # only used when > 1 hidden layer
p["TAU_MEM"] = 8.0
p["TAU_MEM_OUTPUT"]= 8.0
p["TAU_SYN"] = 2.0 
p["REG_TYPE"]= "simple"
p["LBD_UPPER"]= 1e-7
p["LBD_LOWER"]= 1e-7
p["NU_UPPER"]= 6
p["NU_LOWER"]= 6
p["RHO_UPPER"]= 10000.0100
p["GLB_UPPER"]= 1e-8
p["TIMING"]= True
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["EVALUATION"]= "speaker"


p["REWIRE_SILENT"]= True
p["REWIRE_LIFT"]= 0.01
p["AVG_SNSUM"]= True

# "first_spike" loss function variables
p["TAU_0"]= 2
p["TAU_1"]= 25.0
p["ALPHA"]= 1e-3 

p["LOSS_TYPE"]= "sum_weigh_exp"

p["AUGMENTATION"]= {
    "random_shift": 4.0,
}

#p["REDUCED_CLASSES"]= [0]

p["SPEAKER_LEFT"]= 0
p["COLLECT_CONFUSION"]= False
p["TAU_ACCUMULATOR"]= 5.0

p["HIDDEN_NOISE"]= 0.0
p["RESCALE_T"]= 0.1
p["RESCALE_X"]= 1.0

p["TRAIN_TAUM"]= False
p["N_HID_LAYER"]= 1

# learning rate schedule depending on exponential moving average of performance
p["EMA_ALPHA1"]= 0.2
p["EMA_ALPHA2"]= 0.05
p["ETA_FAC"]= 0.5
p["MIN_EPOCH_ETA_FIXED"]= 20

p["REC_SPIKES"]= ["input"]
p["REC_SPIKES_EPOCH_TRIAL"]= [[0,0]]
p["TAUM_OUTPUT_EPOCH_TRIAL"]= [[2,0],[10,0],[20,0],[50,0],[100,0],[299,0]]

found= False
i= -1
while not found:
    i=i+1
    jname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.json')
    found= not exists(jname)

jfile= open(jname,'w')
json.dump(p,jfile)

mn= SHD_model(p)
spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train(p)
#spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.cross_validate_SHD(p)

rname= os.path.join(p["OUT_DIR"], p["NAME"]+'.'+str(i)+'.summary.txt')
sumfile= open(rname,'w')
sumfile.write("Training correct: {}, Valuation correct: {}".format(correct,correct_eval))

print("correct: {}".format(correct))
print("correct_eval: {}".format(correct_eval))

#plt.figure()
#plt.scatter(spike_t["input"], spike_ID["input"])
#plt.show()
