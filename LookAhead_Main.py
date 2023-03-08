import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os

from KF_Algorithms import LookAhead

if __name__ == "__main__":
    #parameters
    data= "C:/Users/Dima/Desktop/TestFolder/500_Samples_180Dense.csv"
    image_path = "C:/Users/Dima/Desktop/TestFolder/Images/"
    image_base_name = "LA_PH_FinalTest"
    data_frame = "C:/Users/Dima/Desktop/TestFolder/LA_PH_Final_Test.csv"
    parallel = True #True/False
    algorithm = 1 #0 = Brute-Force, 1 = Prediction Horizon
    p_val_threshold = 0.19
    horizon = 30
    cores = 2

    #read in and safe data
    df = pd.read_csv(data)

    #use json.loads(x) in order to make list of strings back to list of lists
    obs = [json.loads(x) for x in df.obs.to_list()]
    z = [json.loads(x) for x in df.indices.to_list()]
    obs_no_noise = [json.loads(x) for x in df.obs_no_noise.to_list()]
    sigmas = [json.loads(x) for x in df.sigmas.to_list()]

    #initialization
    true_strings = []
    predicted_strings = []
    matches = []
    mean_seqs = []
    distances = []
    pred_switching_indices = []
    true_switching_indices = []

    #run the algorithm unparallelized
    if not parallel:
        for i in range(len(obs)):
            if algorithm == 0:
                pred_switching_seq, true_switching_seq, distance, match, mean_seq, pred_switching_indices_,true_switching_indices_ = LookAhead.brute_force_evaluation(np.array(obs[i]),np.array(obs_no_noise[i]),p_val_threshold)
            else:
                pred_switching_seq, true_switching_seq, distance, match, mean_seq, pred_switching_indices_,true_switching_indices_ = LookAhead.prediction_horizon(np.array(obs[i]), np.array(obs_no_noise[i]),p_val_threshold,horizon)

            #safe variables
            predicted_strings.append(pred_switching_seq)
            true_strings.append(true_switching_seq)
            distances.append(distance)
            matches.append(match) 
            mean_seqs.append(mean_seq)
            pred_switching_indices.append(pred_switching_indices_)
            true_switching_indices.append(true_switching_indices_)

    #run the algorithm parallelized
    else:
        parallel_obs = []
        if algorithm == 0:
            for i in range(0,len(obs)):
                parallel_obs.append((np.array(obs[i]), np.array(obs_no_noise[i]),p_val_threshold))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(LookAhead.brute_force_evaluation, parallel_obs):
                        predicted_strings.append(results[0])
                        true_strings.append(results[1])
                        distances.append(results[2])
                        matches.append(results[3]) 
                        mean_seqs.append(results[4])
                        pred_switching_indices.append(results[5])
                        true_switching_indices.append(results[6])

        else:
            for i in range(0,len(obs)):
                parallel_obs.append((np.array(obs[i]), np.array(obs_no_noise[i]),p_val_threshold, horizon))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(LookAhead.prediction_horizon, parallel_obs):
                        predicted_strings.append(results[0])
                        true_strings.append(results[1])
                        distances.append(results[2])
                        matches.append(results[3]) 
                        mean_seqs.append(results[4])
                        pred_switching_indices.append(results[5])
                        true_switching_indices.append(results[6])


    #generate the image path if it doesn't exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    #save results
    plt.figure(figsize=(10,4))
    for i in range(len(obs)):
        plt.plot(obs[i], label='observations')
        plt.plot(obs_no_noise[i], c='r', label='no noise')
        plt.plot(mean_seqs[i], c="black", label="prediction")
        plt.ylabel('[measurement unit]')
        plt.xlabel('t')
        plt.legend()
        plt.savefig(image_path+image_base_name+"_"+ str(i) + ".jpg", bbox_inches="tight")
        plt.cla()    

    
    data_df = pd.DataFrame(list(zip(true_strings,predicted_strings,mean_seqs, matches, distances, pred_switching_indices, true_switching_indices)), columns=["true_string","predicted_string","mean_seq","match", "distance", "predicted_switching_indices","true_switching_indices"])
    data_df.to_csv(data_frame, index=False)