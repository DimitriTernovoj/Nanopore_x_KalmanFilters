import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os

from KF_Algorithms import ForwardBackward

if __name__ == "__main__":
    data= "C:/Users/Dima/Desktop/TestFolder/500_Samples_180Dense.csv"
    image_path = "C:/Users/Dima/Desktop/TestFolder/Images/"
    image_base_name = "180Dense"
    data_frame = "C:/Users/Dima/Desktop/TestFolder/180Dense.csv"
    parallelized = False
    cores = 1
    kmer_model_path = "C:/Users/Dima/Desktop/TestFolder/template_median69pA.model"
    data_type = "synthetic" #synthetic/real
    reference_sequence = "GGGAGAGTGCTCCGCTCATCGACCGaCTGGATCCGTgCCACGCTCTGcACGCTGCGTCtGCCTCGGTAACATGGTCATAGCTGTTTCCTG"[::-1] #used only in real data

    #read in data
    KMER_MODEL = pd.read_csv(kmer_model_path, sep="\t", index_col="kmer_enc2")
    model_mus = np.array(KMER_MODEL.level_mean)
    model_sigmas = np.array(KMER_MODEL.level_stdv)
    df = pd.read_csv(data)

    #use json.loads(x) in order to make list of strings back to list of lists
    obs = [json.loads(x) for x in df.obs.to_list()]

    if data_type=="synthetic":
        z = [json.loads(x) for x in df.indices.to_list()]
        obs_no_noise = [json.loads(x) for x in df.obs_no_noise.to_list()]
        sigmas = [json.loads(x) for x in df.sigmas.to_list()]

        #generate transition matrix
        Z_f, Z_b = ForwardBackward.Transition_Matrix(KMER_MODEL, obs_no_noise[0])

    #initialization
    most_likely_states = []
    most_likely_positions = []
    FB_distributions = []
    pred_switching_seqs = []
    true_switching_seqs = []
    distances = []
    matches = []
    pred_switching_indices = []
    true_switching_indices = []

    #run the algorithm unparallelized
    if not parallelized:

        M = 1024
        name = np.arange(M)
        Q = 0

        for i in range(len(obs)):

            #infer starting position
            start_front = -1
            start_back = -1
            if data_type == "real":
                Z_f, Z_b = ForwardBackward.Transition_Matrix(KMER_MODEL, obs[i], reference_sequence)
                mean_front = np.mean(obs[i][:6])
                mean_back = np.mean(obs[i][-6:])
                threshold_front = np.inf
                threshold_back = np.inf

                for j in range(len(model_mus)):
                    dist_f = abs(mean_front-model_mus[j])
                    dist_b = abs(mean_back-model_mus[j])

                    if dist_f < threshold_front:
                        start_front = j
                        threshold_front = dist_f
                    
                    if dist_b < threshold_back:
                        start_back = j
                        threshold_back = dist_b

            else:  
                for index, mean in enumerate(model_mus):
                    if start_front != -1 and start_back != -1:
                        break
                    if mean == obs_no_noise[i][0]:
                        start_front = index
                    if mean == obs_no_noise[i][-1]:
                        start_back = index

            filter_set_forward = []
            filter_set_backward = []
            for k in range(len(model_mus)):
                if k == start_front:
                    filter_set_forward.append(ForwardBackward.KalmanFilter.KalmanFilter_class(model_mus[k],model_mus[k],model_sigmas[k],Q,model_sigmas[k],name[k],1))
                else:
                    filter_set_forward.append(ForwardBackward.KalmanFilter.KalmanFilter_class(model_mus[k],model_mus[k],model_sigmas[k],Q,model_sigmas[k],name[k],0))

                if k == start_back:
                    filter_set_backward.append(ForwardBackward.KalmanFilter.KalmanFilter_class(model_mus[k],model_mus[k],model_sigmas[k],Q,model_sigmas[k],name[k],1))
                else:
                    filter_set_backward.append(ForwardBackward.KalmanFilter.KalmanFilter_class(model_mus[k],model_mus[k],model_sigmas[k],Q,model_sigmas[k],name[k],0))
            
            if data_type == "synthetic":
                pred_switching_seq_, true_switching_seq_, distance_, match_, pred_switching_indices_, true_switching_indices_, most_likely_states_, most_likely_positions_, FB_dist_ = ForwardBackward.ForwardBackwardGPB2(filter_set_forward,filter_set_backward,Z_f,Z_b,obs[i],obs_no_noise[i])
                
                pred_switching_seqs.append(pred_switching_seq_) 
                true_switching_seqs.append(true_switching_seq_) 
                distances.append(distance_) 
                matches.append(match_) 
                pred_switching_indices.append(pred_switching_indices_) 
                true_switching_indices.append(true_switching_indices_) 
                most_likely_states.append(most_likely_states_)
                most_likely_positions.append(most_likely_positions_)
                FB_distributions.append(FB_dist_)

            else:
                pred_switching_seq_, pred_switching_indices_, most_likely_states_, most_likely_positions_, FB_dist_ = ForwardBackward.ForwardBackwardGPB2(filter_set_forward,filter_set_backward,Z_f,Z_b,obs[i])

                pred_switching_seqs.append(pred_switching_seq_)
                pred_switching_indices.append(pred_switching_indices_)
                most_likely_states.append(most_likely_states_)
                most_likely_positions.append(most_likely_positions_)
                FB_distributions.append(FB_dist_)

    #run the algorithm parallelized
    else:
        if data_type == "synthetic":
            parallel_obs = []
            for index in range(0,len(obs)):
                parallel_obs.append((index,[],[],model_mus,model_sigmas,Z_f,Z_b,obs,obs_no_noise))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(ForwardBackward.Forward_Backward_Loop, parallel_obs):
                    pred_switching_seqs.append(results[0])
                    true_switching_seqs.append(results[1])
                    distances.append(results[2])
                    matches.append(results[3])
                    pred_switching_indices.append(results[4])
                    true_switching_indices.append(results[5])
                    most_likely_states.append(results[6])
                    most_likely_positions.append(results[7])
                    FB_distributions.append(results[8])
        else:
            parallel_obs = []

            for index in range(0,len(obs)):
                Z_f, Z_b = ForwardBackward.Transition_Matrix(KMER_MODEL, obs[index], reference_sequence)
                parallel_obs.append((index,[],[],model_mus,model_sigmas,Z_f,Z_b,obs))

            with multiprocessing.Pool(processes=cores) as pool:
                for results in pool.starmap(ForwardBackward.Forward_Backward_Loop, parallel_obs):
                    pred_switching_seqs.append(results[0])
                    pred_switching_indices.append(results[1])
                    most_likely_states.append(results[2])
                    most_likely_positions.append(results[3])
                    FB_distributions.append(results[4])

    
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

        data_df = pd.DataFrame(list(zip(true_switching_seqs,pred_switching_seqs,distances,matches,pred_switching_indices,true_switching_indices,most_likely_states,most_likely_positions,FB_distributions)), columns=["true_string","predicted_string","distance","match","predicted_switching_indices","true_switching_indices","most_likely_states","mean_seq","FB_distributions"])
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

        data_df = pd.DataFrame(list(zip(pred_switching_seqs,pred_switching_indices,most_likely_states,most_likely_positions,FB_distributions)), columns=["predicted_string","predicted_switching_indices","most_likely_states","mean_seq","FB_distributions"])
        data_df.to_csv(data_frame, index=False) 







