from simulator_SHD import *
import matplotlib.pyplot as plt
import numpy as np
import json



# Training Setup
p["EVALUATION"]             = "speaker"
p["N_EPOCH"]                = 1
p["BALANCE_TRAIN_CLASSES"]  = True
p["BALANCE_EVAL_CLASSES"]   = True
p["TRAIN_DATA_SEED"]        = 32
p["TEST_DATA_SEED"]         = 44
p["TRIAL_MS"]               = 1000.0
p["AUGMENTATION"]= {
    "NORMALISE_SPIKE_NUMBER": True,
    "random_shift": 40.0,
    "blend": [0.5,0.5]
}
p["N_INPUT_DELAY"]          = 10
p["INPUT_DELAY"]            = 30
p["N_BATCH"]= 32
p["DATASET"]= "SHD"
p["NAME"]= "tt1"
p["DT_MS"]= 1
p["DEBUG"]= False
p["DEBUG_HIDDEN_N"]= False
p["LOAD_LAST"]= False


# Learning parameters
p["ETA"]                    = 0.001
p["MIN_EPOCH_ETA_FIXED"]    = 1
p["LOSS_TYPE"]              = "sum_weigh_exp"
p["TAU_0"]                  = 1
p["TAU_1"]                  = 100
p["ALPHA"]                  = 5*10^(-5)


# Network parameters
p["REG_TYPE"]               = "simple"
p["TAU_MEM"]                = 20
p["TAU_SYN"]                = 5
p["N_HID_LAYER"]            = 1
p["NUM_HIDDEN"]             = 512
p["RECURRENT"]              = True
p["INPUT_HIDDEN_MEAN"]      = 0.03
p["INPUT_HIDDEN_STD"]       = 0.01
p["HIDDEN_HIDDEN_MEAN"]     = 0
p["HIDDEN_HIDDEN_STD"]      = 0.02 
p["HIDDEN_OUTPUT_MEAN"]     = 0 
p["HIDDEN_OUTPUT_STD"]      = 0.03
p["PDROP_INPUT"]            = 0
p["NU_UPPER"]               = 14
p["GLB_UPPER"]              = 1e-9
p["LBD_UPPER"]              = 2e-9
p["LBD_LOWER"]              = 2e-9
p["N_MAX_SPIKE"]= 1500
p["PDROP_HIDDEN"]= 0.0
p["ADAM_BETA1"]= 0.9
p["ADAM_BETA2"]= 0.999   



p["N_BATCH"]= 16
p["SUPER_BATCH"]= 1
p["N_TRAIN"]= 8156 # that is all of them
p["N_VALIDATE"]= 0 # no validation
p["SHUFFLE"]= True
p["W_REPORT_INTERVAL"] = 11000  # this should be at the end of the epoch (at first trial of evaluation)
# p["NU_LOWER"]= 5
# p["RHO_UPPER"]= 10000.0
# p["ETA_DECAY"]= 1.0      
p["ETA_FIDDELING"]= False
p["ETA_REDUCE"]= 0.5
p["ETA_REDUCE_PERIOD"]= 50
p["TIMING"]= False
p["SPK_REC_STEPS"]= int(p["TRIAL_MS"]/p["DT_MS"])
p["EVALUATION"]= "random"

p["RECURRENT"]= True

p["REWIRE_SILENT"]= True
p["AVG_SNSUM"]= True


if p["DEBUG"]:
    p["REC_SPIKES"]= ["input", "hidden"]
    #p["REC_NEURONS"]= [("output", "V"), ("output", "lambda_V"), ("output", "lambda_I")]
    #p["REC_SYNAPSES"]= [("hid_to_out", "w")]

with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as file:
    json.dump(p, file)

p["N_TRAIN"]= 8156
p["N_VALIDATE"]= 0
for i in range(10):
    mn= SHD_model(p)
    spike_t, spike_ID, rec_vars_n, rec_vars_s,correct,correct_eval= mn.train_test(p)
    with open(os.path.join(p["OUT_DIR"], p["NAME"]+'_traintest.txt'),'a') as f:
        f.write("{} {}\n".format(correct,correct_eval))
    p["TRAIN_DATA_SEED"]+= 31
