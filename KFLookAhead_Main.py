import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os

from KF_Algorithms import KFLookAhead

if __name__ == "__main__":
    #parameters
    data= "C:/Users/Dima/Desktop/TestFolder/real_data.csv"
    image_path = "C:/Users/Dima/Desktop/TestFolder/Images/"
    image_base_name = "real_data_KFLA_par"
    data_frame = "C:/Users/Dima/Desktop/TestFolder/real_data_KFLA_par.csv"
    parallelized = True
    cores = 1
    kmer_model_path = "C:/Users/Dima/Desktop/TestFolder/template_median69pA.model"
    data_type = "real" #synthetic/real

    #read in data
    KMER_MODEL = pd.read_csv(kmer_model_path, sep="\t", index_col="kmer_enc2")
    model_mus = np.array(KMER_MODEL.level_mean)
    model_sigmas = np.array(KMER_MODEL.level_stdv)
    df = pd.read_csv(data)

    #use json.loads(x) in order to make list of strings back to list of lists
    obs = [json.loads(x) for x in df.obs.to_list()]

    if data_type == "synthetic":
        z = [json.loads(x) for x in df.indices.to_list()]
        obs_no_noise = [json.loads(x) for x in df.obs_no_noise.to_list()]
        sigmas = [json.loads(x) for x in df.sigmas.to_list()]

    true_switching_seqs = []
    pred_switching_seqs = []
    most_likely_positions = []
    matches = []
    distances = []
    pred_switching_indices = []
    true_switching_indices = []
    most_likely_states = []


    #run the algorithm unparallelized
    if not parallelized:
        M = 1024
        name = np.arange(M)
        Q = 0

        for i in range(len(obs)):
            filter_set = []
            curr_state = 0

            if data_type == "real":
                #Find closest Mean if no Ground Truth provided
                mean_start = np.mean(obs[i][:6])
                threshold = np.inf
                for j, mu in enumerate(model_mus):
                    dist_f = abs(mean_start-mu)

                    if dist_f < threshold:
                        start = j
                        threshold = dist_f

                    filter_set.append(KFLookAhead.KalmanFilter.KalmanFilter_class(model_mus[j],model_mus[j],model_sigmas[j],Q,model_sigmas[j],name[j]))

                predicted_string_, mlp_, pred_switching_indices_, mls_ = KFLookAhead.LookAhead(filter_set,obs[i],curr_state)

                pred_switching_seqs.append(predicted_string_)
                most_likely_positions.append(mlp_)
                pred_switching_indices.append(pred_switching_indices_)
                most_likely_states.append(mls_)
            
            else:
                for j, mu in enumerate(model_mus):
                    if mu == obs_no_noise[i][0]:
                        start = j

                    filter_set.append(KFLookAhead.KalmanFilter.KalmanFilter_class(model_mus[j],model_mus[j],model_sigmas[j],Q,model_sigmas[j],name[j]))

                true_string_, predicted_string_, mlp_, match_, distance_, pred_switching_indices_, true_switching_indices_, mls_ = KFLookAhead.LookAhead(filter_set,obs[i],curr_state,obs_no_noise[i])

                true_switching_seqs.append(true_string_)
                pred_switching_seqs.append(predicted_string_)
                most_likely_positions.append(mlp_)
                matches.append(match_)
                distances.append(distance_)
                pred_switching_indices.append(pred_switching_indices_)
                true_switching_indices.append(true_switching_indices_)
                most_likely_states.append(mls_)
                
    #run the algorithm parallelized
    else:
        if data_type == "synthetic":
            parallel_obs = []
            for i in range(0,len(obs)):
                parallel_obs.append((i,obs,model_mus,model_sigmas,obs_no_noise))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(KFLookAhead.LookAhead_Manager, parallel_obs):
                    true_switching_seqs.append(results[0])
                    pred_switching_seqs.append(results[1])
                    most_likely_positions.append(results[2])
                    matches.append(results[3])
                    distances.append(results[4])
                    pred_switching_indices.append(results[5])
                    true_switching_indices.append(results[6])
                    most_likely_states.append(results[7])

        else:
            parallel_obs = []
            for i in range(0,len(obs)):
                parallel_obs.append((i,obs,model_mus,model_sigmas))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(KFLookAhead.LookAhead_Manager, parallel_obs):
                    pred_switching_seqs.append(results[0])
                    most_likely_positions.append(results[1])
                    pred_switching_indices.append(results[2])
                    most_likely_states.append(results[3])


    #generate the image path if it doesn't exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    #save results
    if data_type == "synthetic":

        plt.figure(figsize=(10,4))
        for i in range(len(obs)):
            plt.plot(obs[i], label='observations')
            plt.plot(obs_no_noise[i], c='r', label='no noise')
            plt.plot(most_likely_positions[i], c="black", label="prediction")
            plt.ylabel('[measurement unit]')
            plt.xlabel('t')
            plt.legend()
            plt.savefig(image_path+image_base_name+"_"+ str(i) + ".jpg", bbox_inches="tight")
            plt.cla()

        data_df = pd.DataFrame(list(zip(true_switching_seqs,pred_switching_seqs,distances,matches,pred_switching_indices,true_switching_indices,most_likely_states,most_likely_positions)), columns=["true_string","predicted_string","distance","match","predicted_switching_indices","true_switching_indices","most_likely_states","mean_seq"])
        data_df.to_csv(data_frame, index=False)  

    else:
        plt.figure(figsize=(10,4))
        for i in range(len(obs)):
            plt.plot(obs[i], label='observations')
            plt.plot(most_likely_positions[i], c="black", label="prediction")
            plt.ylabel('[measurement unit]')
            plt.xlabel('t')
            plt.legend()
            plt.savefig(image_path+image_base_name+"_"+ str(i) + ".jpg", bbox_inches="tight")
            plt.cla()

        data_df = pd.DataFrame(list(zip(pred_switching_seqs,pred_switching_indices,most_likely_states,most_likely_positions)), columns=["predicted_string","predicted_switching_indices","most_likely_states","mean_seq"])
        data_df.to_csv(data_frame, index=False)  


