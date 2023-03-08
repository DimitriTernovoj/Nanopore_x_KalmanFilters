import numpy as np
from dtaidistance import dtw
from scipy.stats import linregress

def Bounded_Classification(true_switches,predicted_switches,seq_len):
    '''Computes the bounded Classification.\n
       Input: Indices of Ground Truth Switches, Indices of Predicted Switches, Sequence Length
       Output: False Positives, True Positives, False Negatives, True Negatives, Relative True Positive Distance, Absolute True Positive Distance 
    '''
    #Lower Bound is exclusive
    #Upper Bound is inclusive
    #special case: ground truth doesn't include switches
    if true_switches == []:
        TP_dists = np.nan
        TP_abs_dists = np.nan
        FPs = len(predicted_switches)
        TPs = np.nan
        TNs = seq_len-FPs
        FNs = np.nan

        return FPs, TPs, FNs, TNs, TP_dists, TP_abs_dists

    #special case: only one switch in the ground truth
    if len(true_switches) == 1:
        #option: no predicted switches
        if not predicted_switches:
            FPs = 0
            TPs = 0
            FNs = 1
            TNs = seq_len - 1
            TP_dists = np.nan
            TP_abs_dists = np.nan
            return FPs, TPs, FNs, TNs, TP_dists, TP_abs_dists
        
        #option: predicted switches
        else:
            true_switch = true_switches[0]
            TP_dists = np.inf
            TP_abs_dists = np.inf

            for j in predicted_switches:
                #switch was predicted before the actual switch
                if j < true_switch:
                    lb_denom = int(true_switch/2)-1 if true_switch % 2 == 0 else int(true_switch/2)
                    #outside of the window
                    if j < true_switch - lb_denom:
                        continue

                    temp_TP = 0.5/lb_denom * (true_switch-j)
                    temp_TP_abs = (true_switch-j)

                    if temp_TP < TP_dists:
                        TP_dists = temp_TP
                        TP_abs_dists = temp_TP_abs

                #switch was predicted after the actual switch
                else:
                    denom = int((seq_len - true_switch)/2)
                    #outside of the window
                    if j > true_switch+denom:
                        continue
                    
                    temp_TP = 0.5/denom * (j-true_switch)
                    temp_TP_abs = (j-true_switch)

                    if temp_TP < TP_dists:
                        TP_dists = temp_TP
                        TP_abs_dists = temp_TP_abs

            #option: no switch was found within the valid window
            if TP_dists == np.inf:
                FPs = len(predicted_switches)
                TPs = 0
                FNs = 1
                TNs = seq_len - FPs + FNs
                TP_dists = np.nan
                TP_abs_dists = np.nan

            #option: switch was found within the valid window
            else:
                FPs = len(predicted_switches)-1
                TPs = 1
                FNs = 0
                TNs = seq_len - len(predicted_switches)
                TP_dists = min(0.5,TP_dists)

            return FPs, TPs, FNs, TNs, TP_dists, TP_abs_dists

    #No Special Cases
    #initialize variables
    FPs = 0
    TPs = 0
    FNs = 0
    TNs = 0
    TP_dists = 0
    TP_abs_dists = 0
    
    outer_loop_finish = True
    temp_TP = 0
    temp_TP_abs = 0

    for count,i in enumerate(true_switches):
        #option: no more predicted switches, but still true switches left
        if not predicted_switches:
            FNs += 1
            outer_loop_finish = False
            TP_dists += temp_TP
            TP_abs_dists += temp_TP_abs
            temp_TP = 0
            temp_TP_abs = 0

            TP_dists += 0.5
            if i == true_switches[-1]:
                lb = int((true_switches[count] - true_switches[count-1])/2)-1 if (true_switches[count] - true_switches[count-1]) % 2 == 0 else int((true_switches[count] - true_switches[count-1])/2)
                ub = int((seq_len - true_switches[count])/2)
                TP_abs_dists += max(lb,ub)
            else:
                lb = int((true_switches[count] - true_switches[count-1])/2)-1 if (true_switches[count] - true_switches[count-1]) % 2 == 0 else int((true_switches[count] - true_switches[count-1])/2)
                ub = int((true_switches[count+1] - true_switches[count])/2)
                TP_abs_dists += max(lb,ub)
                
        else:
            temp_TP = np.inf
            temp_TP_abs = np.inf

            for j in predicted_switches:
                if i != true_switches[-1]:
                    #if the predicted switch exceeds the window that is currently looked at, move to the next true switch
                    bound = true_switches[count] + int((true_switches[count+1] - true_switches[count])/2)
                    if j > bound:
                        if temp_TP != np.inf:
                            TPs += 1
                            TP_dists += temp_TP
                            TP_abs_dists += temp_TP_abs
                        else:
                            FNs += 1
                            TP_dists += 0.5
                            if count != 0:
                                lb = int((true_switches[count] - true_switches[count-1])/2)-1 if (true_switches[count] - true_switches[count-1]) % 2 == 0 else int((true_switches[count] - true_switches[count-1])/2)
                                ub = int((true_switches[count+1] - true_switches[count])/2)
                                TP_abs_dists += max(lb,ub)

                            #when it happens at the very beginning, there is no count-1, which is why the start of the seq has to be used
                            else:
                                lb = int((true_switches[count])/2)-1 if true_switches[count] % 2 == 0 else int((true_switches[count])/2)
                                ub = int((true_switches[count+1] - true_switches[count])/2) 
                                TP_abs_dists += max(lb,ub)

                        break

                    #skip to the next iteration if the correct switch was inferred
                    elif temp_TP == 0:
                        FPs += 1
                        predicted_switches = predicted_switches[1:]
                        continue


                #first iteration
                if count == 0:
                    #if the predicted changepoint is not within half the distance between the beginning and the 1st changepoint
                    #it is assumed to be a False-Positive
                    bound = int(i/2)-1 if i % 2 == 0 else int(i/2)
                    if j < i-bound:
                        FPs += 1    

                    #the predicted switch is after the true switch
                    elif j >= i:
                        denom = int((true_switches[count+1] - true_switches[count])/2)
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (j-i)
                            temp_TP_abs = (j-i)
                        else: 
                            distance = 0.5/denom * (j-i)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (j-i)

                            FPs += 1

                    #the predicted switch is before the true switch
                    else:
                        denom = bound
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (i-j)
                            temp_TP_abs = (i-j)
                        else:
                            distance = 0.5/denom * (i-j)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (i-j)

                            FPs += 1        

                #last iteration
                elif count == (len(true_switches)-1):
                    val = int((seq_len - true_switches[count])/2)
                    if j > true_switches[-1] + val :
                        FPs += 1

                    #the predicted switch is after the true switch
                    elif j>=i:
                        denom = int((seq_len - true_switches[count])/2)
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (j-i)
                            temp_TP_abs = (j-i)
                        else:
                            distance = 0.5/denom * (j-i)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (j-i)
                            
                            FPs += 1

                    #the predicted switch is before the true switch
                    else:
                        denom = int((true_switches[count] - true_switches[count-1])/2)-1 if (true_switches[count] - true_switches[count-1]) % 2 == 0 else int((true_switches[count] - true_switches[count-1])/2)
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (i-j)
                            temp_TP_abs = (i-j)
                        else:
                            distance = 0.5/denom * (i-j)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (i-j)

                            FPs += 1 

                #main-lopp
                else:
                    #the switch is after the true switch
                    if j >= i:
                        denom = int((true_switches[count+1] - true_switches[count])/2)
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (j-i)
                            temp_TP_abs = (j-i)
                        else: 
                            distance = 0.5/denom * (j-i)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (j-i)

                            FPs += 1

                    #the switch is before the true switch
                    else:
                        denom = int((true_switches[count] - true_switches[count-1])/2)-1 if (true_switches[count] - true_switches[count-1]) % 2 == 0 else int((true_switches[count] - true_switches[count-1])/2)
                        if temp_TP == np.inf:
                            temp_TP = 0.5/denom * (i-j)
                            temp_TP_abs = (i-j)
                        else: 
                            distance = 0.5/denom * (i-j)
                            if distance < temp_TP:
                                temp_TP = distance
                                temp_TP_abs = (i-j)

                            FPs += 1   
                        
                predicted_switches = predicted_switches[1:]

    if outer_loop_finish:
        TPs += 1
        TP_abs_dists += temp_TP_abs
        TP_dists += temp_TP

    return FPs, TPs, FNs, seq_len-(FPs+TPs+FNs), TP_dists, TP_abs_dists

def Exact_Classification(true_switches,predicted_switches,seq_len):
    '''Computes the bounded Classification.\n
       Input: Indices of Ground Truth Switches, Indices of Predicted Switches, Sequence Length
       Output: False Positives, True Positives, False Negatives, True Negatives
    '''
    FPs = 0
    TPs = 0
    FNs = 0
    TNs = 0

    #special case: ground truth doesn't include switches
    if not true_switches:
        FPs = len(predicted_switches)
        TPs = np.nan
        FNs = np.nan
        TNs = seq_len - len(predicted_switches)

    #special case: no switches were predicted
    elif not predicted_switches:
        FPs = 0
        TPs = 0
        FNs = len(true_switches)
        TNs = seq_len - len(true_switches)

    else:
        matches = 0
        for i in true_switches:
            for j in predicted_switches:
                if i == j:
                    matches += 1
        
        FPs = len(predicted_switches)-matches
        TPs = matches
        FNs = len(true_switches)-matches
        TNs = seq_len - (FPs+TPs+FNs)
    
    return FPs, TPs, FNs, TNs


def Mean_Seq_Dists(true_means,predicted_means,sigmas):
    '''Computes Distance Metrics between Ground Truth and Predicted Sequence.\n
       Input: Ground Truth Sequence, Predicted Sequence, Ground Truth Sigmas per Segment \n
       Output: L2-Distance, Normalized L2-Distance, Dynamic Time Warping Distance
    '''
    #L2-Distance
    L2_dist = np.sqrt(np.sum((true_means - predicted_means)**2))

    #L2-Distance normalized by true standard deviation per segment
    normed_L2_dist = np.sqrt(np.sum(((true_means - predicted_means)/sigmas)**2))

    #Dynamic Time Warping Distance
    dtw_dist = dtw.distance_fast(true_means,predicted_means)

    return L2_dist, normed_L2_dist, dtw_dist

#Helper Function for Prediction Score
def MeansPerSegments(true_means, obs):
    '''Helper Function for the Computation of the Prediction Score.\n
       Computes the best case prediction (the Empirical Mean per True Segment).
    '''
    counter = 1
    values = obs[0]
    segment_means_emp = []
    full_segment_emp = []

    for i in range(1,len(true_means)):

        if true_means[i] == true_means[i-1]:
            values += obs[i]
            counter += 1
            if i == (len(true_means)-1):
                segment_means_emp.append(values/counter)
                full_segment_emp += [segment_means_emp[-1]]*counter
        else:
            segment_means_emp.append(values/counter)
            full_segment_emp += [segment_means_emp[-1]]*counter

            counter = 1
            values = obs[i]

    return full_segment_emp


def Prediction_Score(obs,true_means,predicted_means,switches = True):
    '''Computes the Prediction Score.\n
       Input: Observations, Ground Truth, Predicted Sequence\n
       Output: Prediction Score (Number between 0 and 1)
    '''
    
    if switches:
        #computing worst and best case scenarios
        full_segment_emp = MeansPerSegments(true_means,obs)
        worst_case = np.array([np.mean(obs)]*len(obs))
        true_means = np.array(true_means)

        #computing the score using linear interpolation
        x0 = np.linalg.norm(true_means-worst_case) # worst-score=0
        x1 = np.linalg.norm(true_means-full_segment_emp) # best-score=1

        a, b, _, _, _ = linregress([x0,x1], [0,1])

        x_pred = np.linalg.norm(true_means-predicted_means)

        prediction_score = a * x_pred + b
        #cap the prediction score to 1
        return min(prediction_score,1)
    
    else:
        return np.float("nan")

def Switching_Ratios(true_strings, predicted_strings):
    '''Computes Switching Ratios.\n
       Input: Ground Truth Switching Sequence, Predicted Switching Sequence
       Output: Switching Ratio (#Predicted Switches/#True Switches), Correct Switching Ratio (#Correctly Predicted Switches(#True Switches)
    '''

    true_switching_number = 0
    predicted_switching_number = 0
    correctly_inferred_switches = 0
    for j in range(len(true_strings)):

        #if there is a true_switch, increase true_switching_number and check if that switch was correctly predicted or not, increase the variables accordingly
        if true_strings[j] == "1":
            true_switching_number += 1
            if predicted_strings[j] == "1":
                correctly_inferred_switches += 1
                predicted_switching_number += 1

        #elif check if there was a switch predicted
        elif predicted_strings[j] == "1":
            predicted_switching_number += 1

    if true_switching_number != 0:
        #Number of Predicted Switches divided by Number of True Switches
        switching_ratio = predicted_switching_number/true_switching_number

        #Correctly Predicted Switches divided by Number of True Switches
        correct_switching_ratio = correctly_inferred_switches/true_switching_number
    else:
        switching_ratio = np.float("nan")
        correct_switching_ratio = np.float("nan")

    return switching_ratio, correct_switching_ratio
