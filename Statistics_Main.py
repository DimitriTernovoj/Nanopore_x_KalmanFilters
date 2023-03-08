import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from KF_Algorithms import Statistics

if __name__ == "__main__":
    #parameters to set
    samples = "C:/Users/Dima/Desktop/TestFolder/500_Samples_180Dense.csv"
    best_pred = "C:/Users/Dima/Desktop/TestFolder/180Dense_KFLA.csv"
    data_frame_path = "C:/Users/Dima/Desktop/TestFolder/Statistics/"
    data_frame_base_name = "500_Samples_180Dense_KFLA"
    switch_density = "Dense"
    
    #read in and concatenate data
    df_1 = pd.read_csv(samples,) 
    df_2 = pd.read_csv(best_pred, dtype="str")
    df = pd.concat([df_1,df_2],axis=1)
    
    true_strings = df.true_string.to_list()
    predicted_strings = df.predicted_string.to_list()
    true_switches = [json.loads(x) for x in df.true_switching_indices.to_list()]
    predicted_switches = [json.loads(x) for x in df.predicted_switching_indices.to_list()]
    true_mean_seqs = [json.loads(x) for x in df.obs_no_noise.to_list()]
    predicted_mean_seqs = [json.loads(x) for x in df.mean_seq.to_list()]
    sigmas = [json.loads(x) for x in df.sigmas.to_list()]
    obss = [json.loads(x) for x in df.obs.to_list()]

    #sequence length
    seqlen = len(obss[0])
    #number of true changepoints per sample
    num_switches = [len(x) for x in true_switches]
    #bounded classification
    FPs_bounded,TPs_bounded,FNs_bounded,TNs_bounded, TP_dists, TP_abs_dists = [],[],[],[],[],[]
    #exact classification
    FPs_exact,TPs_exact,FNs_exact,TNs_exact = [],[],[],[]
    #hamming distance
    hamming_dist = []
    #mean_seq distances
    L2_dist = []
    normed_L2_dist = []
    dtw_dist = []
    #prediction score
    prediction_score = []
    #switching_ratios
    switching_ratio = []
    correct_switching_ratio = []

    for i in range(len(true_strings)):
        #Bounded Classification
        FPs_bounded_, TPs_bounded_, FNs_bounded_, TNs_bounded_, TP_dists_, TP_abs_dists_ = Statistics.Bounded_Classification(true_switches[i],predicted_switches[i],len(true_strings[0]))
        FPs_bounded.append(FPs_bounded_)
        TPs_bounded.append(TPs_bounded_)
        FNs_bounded.append(FNs_bounded_)
        TNs_bounded.append(TNs_bounded_)
        TP_dists.append(TP_dists_)
        TP_abs_dists.append(TP_abs_dists_)
    

        #Exact Classification
        FPs_exact_, TPs_exact_, FNs_exact_, TNs_exact_ = Statistics.Exact_Classification(true_switches[i],predicted_switches[i],len(true_strings[0]))
        FPs_exact.append(FPs_exact_)
        TPs_exact.append(TPs_exact_)
        FNs_exact.append(FNs_exact_)
        TNs_exact.append(TNs_exact_)

        #Hamming Distance
        hamming_dist_ = sum(c1 != c2 for c1, c2 in zip(true_strings[i], predicted_strings[i]))
        hamming_dist.append(hamming_dist_)

        #Distance Metrics
        L2_dist_, normed_L2_dist_, dtw_dist_ = Statistics.Mean_Seq_Dists(np.array(true_mean_seqs[i]),np.array(predicted_mean_seqs[i]),np.array(sigmas[i]))
        L2_dist.append(L2_dist_)
        normed_L2_dist.append(normed_L2_dist_)
        dtw_dist.append(dtw_dist_)

        #Switching Ratios
        switching_ratio_, correct_switching_ratio_ = Statistics.Switching_Ratios(true_strings[i], predicted_strings[i])
        switching_ratio.append(switching_ratio_)
        correct_switching_ratio.append(correct_switching_ratio_)

        #Prediction Score
        if true_switches[i]:
            prediction_score_ = Statistics.Prediction_Score(obss[i],true_mean_seqs[i],predicted_mean_seqs[i])
        else:
            prediction_score_ = np.nan

        #in case of odd values
        if prediction_score_ < 0:
            prediction_score_ = np.nan 

        prediction_score.append(prediction_score_)

    #Compute Statistics for Classification results (bounded and exact)
    TPs_bounded = np.array(TPs_bounded)
    FPs_bounded = np.array(FPs_bounded)
    FNs_bounded = np.array(FNs_bounded)
    TNs_bounded = np.array(TNs_bounded)

    TPs_exact = np.array(TPs_exact)
    FPs_exact = np.array(FPs_exact)
    FNs_exact = np.array(FNs_exact)
    TNs_exact = np.array(TNs_exact)

    #BOUNDED
    #Sensitivity (True Positive Rate)
    Sensitivity_bounded = TPs_bounded/(TPs_bounded+FNs_bounded)
    #False Negative Rate
    FNR_bounded = FNs_bounded/(TPs_bounded+FNs_bounded)
    #False Positive Rate
    FPR_bounded = FPs_bounded/(FPs_bounded+TNs_bounded)
    #Specificity (True Negative Rate)
    Specificity_bounded = TNs_bounded/(TNs_bounded + FPs_bounded)
    #Precision, Positive Predictive Value (PPV)
    Precision_bounded = TPs_bounded/(TPs_bounded + FPs_bounded)
    #False Discovery rate (FDR)
    FDR_bounded = FPs_bounded/(TPs_bounded + FPs_bounded)
    #F1-Score
    F1_bounded = 2*TPs_bounded/(2*TPs_bounded + FPs_bounded + FNs_bounded)

    #EXACT
    #Sensitivity (True Positive Rate)
    Sensitivity_exact = TPs_exact/(TPs_exact+FNs_exact)
    #False Negative Rate
    FNR_exact = FNs_exact/(TPs_exact+FNs_exact)
    #False Positive Rate
    FPR_exact = FPs_exact/(FPs_exact+TNs_exact)
    #Specificity (True Negative Rate)
    Specificity_exact = TNs_exact/(TNs_exact + FPs_exact)
    #Precision, Positive Predictive Value (PPV)
    Precision_exact = TPs_exact/(TPs_exact + FPs_exact)
    #False Discovery rate (FDR)
    FDR_exact = FPs_exact/(TPs_exact + FPs_exact)
    #F1-Score
    F1_exact = 2*TPs_exact/(2*TPs_exact + FPs_exact + FNs_exact)

    #Make all np.arrays back to lists again
    TPs_bounded = TPs_bounded.tolist()
    FPs_bounded = FPs_bounded.tolist()
    FNs_bounded = FNs_bounded.tolist()
    TNs_bounded = TNs_bounded.tolist()
    TPs_exact = TPs_exact.tolist()
    FPs_exact = FPs_exact.tolist()
    FNs_exact = FNs_exact.tolist()
    TNs_exact = TNs_exact.tolist()

    Sensitivity_bounded = Sensitivity_bounded.tolist()
    FNR_bounded = FNR_bounded.tolist()
    FPR_bounded = FPR_bounded.tolist()
    Specificity_bounded = Specificity_bounded.tolist()
    Precision_bounded = Precision_bounded.tolist()
    FDR_bounded = FDR_bounded.tolist()
    F1_bounded = F1_bounded.tolist()
    Sensitivity_exact = Sensitivity_exact.tolist()
    FNR_exact = FNR_exact.tolist()
    FPR_exact = FPR_exact.tolist()
    Specificity_exact = Specificity_exact.tolist()
    Precision_exact = Precision_exact.tolist()
    FDR_exact = FDR_exact.tolist()
    F1_exact = F1_exact.tolist()

    #Normalize Distances by Sequence Length
    L2_dist = (np.array(L2_dist)/seqlen).tolist()
    normed_L2_dist = (np.array(normed_L2_dist)/seqlen).tolist()
    dtw_dist = (np.array(dtw_dist)/seqlen).tolist()
    hamming_dist = (np.array(hamming_dist)/seqlen).tolist()

#Statistics Per Sample
data_df_extensive = pd.DataFrame(list(zip([seqlen]*len(true_strings),num_switches,[switch_density]*len(true_strings),
                                          TPs_bounded,FPs_bounded,FNs_bounded,TNs_bounded, TP_dists, TP_abs_dists,TPs_exact,FPs_exact,FNs_exact,TNs_exact,
                                          Sensitivity_bounded, FNR_bounded, FPR_bounded, Specificity_bounded, Precision_bounded, FDR_bounded, F1_bounded,
                                          Sensitivity_exact, FNR_exact, FPR_exact, Specificity_exact, Precision_exact, FDR_exact, F1_exact,
                                          hamming_dist,L2_dist,normed_L2_dist,dtw_dist,prediction_score,switching_ratio,correct_switching_ratio)),
                                          columns=["Sequence Length","Number True Switches","Switching Density",
                                                   "TP Bounded","FP Bounded", "FN Bounded", "TN Bounded", "Relative TP Distance", "Absolute TP Distance", "TP Exact","FP Exact", "FN Exact", "TN Exact",
                                                   "Sensitivity Bounded", "FNR Bounded", "FPR Bounded", "Specificity Bounded", "Precision Bounded", "FDR Bounded", "F1 Bounded",
                                                   "Sensitivity Exact", "FNR Exact", "FPR Exact", "Specificity Exact", "Precision Exact", "FDR Exact", "F1 Exact",
                                                   "Hamming Distance", "L2 Distance", "Normed L2 Distance", "DTW Distance", "Prediction Score", "Switching Ratio", "Correct Switching Ratio"])


data_df_extensive.to_csv(data_frame_path + data_frame_base_name + "_Extensive.csv", index=False)

#Summary Statistics
data_df_summary = pd.DataFrame(list(zip([seqlen],[np.nanmean(num_switches)],[switch_density],
                                          [np.nanmean(TPs_bounded)],[np.nanmean(FPs_bounded)],[np.nanmean(FNs_bounded)],[np.nanmean(TNs_bounded)], [np.nanmean(TP_dists)], [np.nanmean(TP_abs_dists)],[np.nanmean(TPs_exact)],[np.nanmean(FPs_exact)],[np.nanmean(FNs_exact)],[np.nanmean(TNs_exact)],
                                          [np.nanmean(Sensitivity_bounded)], [np.nanmean(FNR_bounded)], [np.nanmean(FPR_bounded)], [np.nanmean(Specificity_bounded)], [np.nanmean(Precision_bounded)], [np.nanmean(FDR_bounded)], [np.nanmean(F1_bounded)],
                                          [np.nanmean(Sensitivity_exact)], [np.nanmean(FNR_exact)], [np.nanmean(FPR_exact)], [np.nanmean(Specificity_exact)], [np.nanmean(Precision_exact)], [np.nanmean(FDR_exact)], [np.nanmean(F1_exact)],
                                          [np.nanmean(hamming_dist)],[np.nanmean(L2_dist)],[np.nanmean(normed_L2_dist)],[np.nanmean(dtw_dist)],[np.nanmean(prediction_score)],[np.nanmean(switching_ratio)],[np.nanmean(correct_switching_ratio)])),
                                          columns=["Sequence Length","Number True Switches","Switching Density",
                                                   "Mean TP Bounded", "Mean FP Bounded", "Mean FN Bounded", "Mean TN Bounded", "Mean Relative TP Distance", "Mean Absolute TP Distance", "Mean TP Exact","FP Exact", "Mean FN Exact", "Mean TN Exact",
                                                   "Mean Sensitivity Bounded", "Mean FNR Bounded", "Mean FPR Bounded", "Mean Specificity Bounded", "Mean Precision Bounded", "Mean FDR Bounded", "Mean F1 Bounded",
                                                   "Mean Sensitivity Exact", "Mean FNR Exact", "Mean FPR Exact", "Mean Specificity Exact", "Mean Precision Exact", "Mean FDR Exact", "Mean F1 Exact",
                                                   "Mean Hamming Distance", "Mean L2 Distance", "Mean Normed L2 Distance", "Mean DTW Distance", "Mean Prediction Score", "Mean Switching Ratio", "Mean Correct Switching Ratio"])

data_df_summary.to_csv(data_frame_path + data_frame_base_name + "_Summary.csv", index=False)


    
                

    


    
