import numpy as np
import scipy.stats as sc

def brute_force_evaluation(obs, true_means, p_val_threshold,horizon_size=None):
    '''Look-Ahead Algorithm - Brute-Force Version.\n
       Input: Observations, Ground Truth, P-Value Threshold \n
       If Part of Prediction Horizon: \n
       Output: All statistically significant Possibilities within the the horizon-size, Index of Most Likely Option \n
       Else: \n
       Output: Predicted Switching Sequence, True Switching Sequence, L2-Distance between Prediction and Ground Truth, Prediction 100% Correct? (Boolean), Indices of Predicted Switches, Indices of True Switches
    '''
    #structure of possibilities: Switching Sequence,Mean Sequence,Count,Obs_List,Switching_Indices,Viability,Termination Status
    #Index:                                0,             1,        2,      3,             4,         5,              6
    possibilities = []
    curr_mean = np.mean(obs[0:6])
    possibilities.append(["000000",[curr_mean]*6,6,obs[0:6].tolist(),[],True,False])
    termination = False
    
    while termination != True:
        to_be_deleted = []
 
        for i in range(len(possibilities)):
            if possibilities[i][-1] == True:
                continue
            if possibilities[i][-2] == False:
                to_be_deleted.append(i)
                continue
            
            var_1 = possibilities[i][1][:]
            var_3 = possibilities[i][3][:]
            var_4 = possibilities[i][4][:]
            staying = simulate_step(horizon_size,possibilities[i][0],var_1,possibilities[i][2],var_3,var_4,p_val_threshold,possibilities[i][5],0,obs)
            if staying:
                possibilities.append(staying)

            var_1 = possibilities[i][1][:]
            var_3 = possibilities[i][3][:]
            var_4 = possibilities[i][4][:]
            switching = simulate_step(horizon_size,possibilities[i][0],var_1,possibilities[i][2],var_3,var_4,p_val_threshold,possibilities[i][5],1,obs)

            if switching:
                possibilities.append(switching)

            to_be_deleted.append(i)

        #delete paths that were propagated by one iteration
        for i in sorted(to_be_deleted, reverse=True):
            del possibilities[i]

        #see if termination criterion is fulfilled
        termination = True
        for poss in possibilities:
            if poss[-1] == False:
                termination = False
                break
    
    if horizon_size == None:
        predicted_string = ""
        mean_seq = []
        predicted_switching_indices = []

        #compute predicted switching seq
        threshold = np.inf
        for i in possibilities:
            dist = np.linalg.norm(true_means-i[1])
            if dist < threshold and i[-2] == True:
                threshold = dist
                predicted_string = i[0]
                mean_seq = i[1]
                predicted_switching_indices = i[4]


        #compute distance of the most likely sequence
        distance = threshold

        #compute correct switching sequence
        true_switching_indices = []
        correct_string = "0"
        for i in range(1,len(true_means)):
            if true_means[i] != true_means[i-1]:
                correct_string += "1"
                true_switching_indices.append(i)
            else:
                correct_string += "0"
            
        return predicted_string, correct_string, distance, predicted_string == correct_string, mean_seq, predicted_switching_indices, true_switching_indices
    
    else:
        index_best_option = np.inf
        threshold = np.inf
        for i in reversed(range(len(possibilities))):
            if possibilities[i][-2] == False:
                del possibilities[i]

        for i in range(len(possibilities)):
            dist = np.linalg.norm(true_means[:horizon_size]-possibilities[i][1])
            if dist < threshold:
                threshold = dist
                index_best_option = i

        return possibilities, index_best_option

def simulate_step(horizon_size, switching_seq, mean_seq, count, obs_list,switching_indices,p_val_threshold,Viability, move, obs):
    '''Helper Function for the Brute-Force Evaluation of the Look-Ahead Algorithm.
       Exectutes one iteration of staying or switching.
    '''
    termination_criterion = False
    bound = 0

    if horizon_size == None:
        bound = len(obs)
    else:
        bound = horizon_size


    #no switch
    if move == 0:
        obs_list.append(obs[count])
        if not switching_indices: 
            mean_seq[:] = [np.mean(obs_list)]*len(obs_list)
        else:
            mean_seq[switching_indices[-1]:] = [np.mean(obs_list)]*len(obs_list)

        count += 1
        switching_seq += "0"
        if count == bound:
            termination_criterion = True
    
    #more than 6 observations left
    elif count+6 <= bound:
        switching_indices.append(count)

        #run two-sided t-test
        _, p_val = sc.ttest_ind(obs_list,obs[count:count+6])
        if p_val > p_val_threshold:
            Viability = False

        obs_list = obs[count:count+6].tolist()
        new_mean = np.mean(obs_list)
        mean_seq += [new_mean]*6
        count += 6
        switching_seq += "100000"
        if count == bound:
            termination_criterion = True

    #less than 6 obs left 
    else:
        if horizon_size == None:
            return []
        else:
            diff = horizon_size-count 

            #run two-sided t-test
            _, p_val = sc.ttest_ind(obs_list,obs[count:count+6])
            if p_val > p_val_threshold:
                Viability = False

            obs_list = obs[count:count+diff].tolist()
            new_mean = np.mean(obs_list)
            mean_seq += [new_mean]*diff
            switching_seq += "1"+"0"*(diff-1)
            switching_indices.append(count)
            termination_criterion = True

    return [switching_seq,mean_seq,count,obs_list,switching_indices,Viability,termination_criterion]

def evaluate_possibilities(possibilities,obs,true_pos,count,horizon,p_val_threshold,switching=True):
    '''Helper Function for the Prediction Horizon Version of the Look-Ahead Algorithm.\n
       Simulates Staying and Switching for every Possibility and Returns the new Set of Possibilitites and the Index of the Most Likely One.
    '''
    new_possibilities = []
    for pos in possibilities:
        pos_stay = [pos[0],pos[1][:],pos[2][:],pos[3][:]]
        pos_stay[0] += "0"
        pos_stay[2].append(obs[count])

        #replace all values from the last switch with the new mean (including obs[count])
        if pos_stay[3] != []:
            pos_stay[1][pos_stay[3][-1]:] = [np.mean(pos_stay[2])]*(len(pos_stay[1])-pos_stay[3][-1])
            pos_stay[1].append(np.mean(pos_stay[2]))
        else:
            pos_stay[1] = [np.mean(pos_stay[2])]*horizon
            
        new_possibilities.append(pos_stay)

        if switching and pos[0][-5:] == "00000":
            Viability = True

            _, p_val = sc.ttest_ind(pos[2],obs[count:count+6])
            if p_val > p_val_threshold:
                Viability = False

            if Viability:
                pos_switch = [pos[0],pos[1][:],pos[2][:],pos[3][:]]
                pos_switch[0] += "1"
                pos_switch[2] = [obs[count]]
                pos_switch[1].append(obs[count])
                pos_switch[3].append(horizon-1)
                new_possibilities.append(pos_switch)

    #infer most likely possibility
    index_best_option = np.inf
    threshold = np.inf
    for i in range(len(new_possibilities)):
        dist = np.linalg.norm(true_pos[count-horizon+1:count+1]-new_possibilities[i][1])
        if dist < threshold:
            threshold = dist
            index_best_option = i

    return new_possibilities, index_best_option


def prediction_horizon(obs, true_pos, p_val_threshold, horizon_size):
    '''Look-Ahead Algorithm - Prediction Horizon Version. \n
       Input: Observations, Ground Truth, P-Value Threshold, Horizon-Size
       Output: Predicted Switching Sequence, True Switching Sequence, L2-Distance between Prediction and Ground Truth, Prediction 100% Correct? (Boolean), Estimates, Indices of Predicted Switches, Indices of True Switches
    '''
    #run Brute-Force Algorithm in the first window of size: horizon_size
    pred_switching_indices = []
    true_switching_indices = []
    switching_sequence = ""
    mean_sequence = []

    print(f"start brute-force")
    possibilities, index_best_option = brute_force_evaluation(obs,true_pos,p_val_threshold, horizon_size+6)
    print(f"finished brute-force")

    #fix the first 6 positions (index:0-5) (initialization) and the 7th (index: 6) because of the window that was evaluated
    switching_sequence = possibilities[index_best_option][0][:7]
    mean_sequence = possibilities[index_best_option][1][:7]
    if switching_sequence[-1] == "1":
        pred_switching_indices.append(6)
        
    #drop all options deviating from the most likely option
    #reduce possibilities structure to: switching_seq, mean_seq, obs_list, switching_indices
    for i in reversed(range(len(possibilities))):
        if possibilities[i][0][6] != switching_sequence[-1]:
            del possibilities[i]
            continue
        else:
            possibilities[i][4] = (np.array(possibilities[i][4])-7).tolist()[1:] if possibilities[i][0][6] == "1" else (np.array(possibilities[i][4])-7).tolist()
            possibilities[i][0] = possibilities[i][0][7:]
            possibilities[i][1] = possibilities[i][1][7:]
            possibilities[i][2] = possibilities[i][3]
            possibilities[i][3] = possibilities[i][4]
            possibilities[i] = possibilities[i][:4]

    count = horizon_size+6
    while count < len(obs):
        print(f"Current Count: {count}")
        #evaluate possibilities after adding one observation at a time
        if count <= len(obs)-6:
            possibilities, index_best_option = evaluate_possibilities(possibilities,obs,true_pos,count,horizon_size,p_val_threshold)
        else:
            possibilities, index_best_option = evaluate_possibilities(possibilities,obs,true_pos,count,horizon_size,p_val_threshold,switching=False)

        #fix results
        switching_sequence += possibilities[index_best_option][0][0]
        mean_sequence += [possibilities[index_best_option][1][0]]
        if switching_sequence[-1] == "1":
            pred_switching_indices.append(count-horizon_size+1)

        #shift the window and adapt the possibilities correspondingly
        if count < len(obs)-1:
            for i in reversed(range(len(possibilities))):
                if possibilities[i][0][0] != switching_sequence[-1]:
                    del possibilities[i]
                    continue
                else:
                    possibilities[i][0] = possibilities[i][0][1:]
                    possibilities[i][1] = possibilities[i][1][1:]
                    possibilities[i][3] = (np.array(possibilities[i][3])-1).tolist()[1:] if switching_sequence[-1] == "1" else (np.array(possibilities[i][3])-1).tolist()
        count +=1

    #fix entire horizon
    switching_sequence += possibilities[index_best_option][0][1:]
    mean_sequence += possibilities[index_best_option][1][1:]
    possibilities[index_best_option][3] = (np.array(possibilities[index_best_option][3])+count-horizon_size).tolist()[1:] if possibilities[index_best_option][3][0] == 0 else (np.array(possibilities[index_best_option][3])+count-horizon_size).tolist()
    pred_switching_indices += possibilities[index_best_option][3]

    #compute correct switching sequence
    true_switching_indices = []
    correct_string = "0"
    for i in range(1,len(true_pos)):
        if true_pos[i] != true_pos[i-1]:
            correct_string += "1"
            true_switching_indices.append(i)
        else:
            correct_string += "0"

    distance = np.linalg.norm(np.array(true_pos)-np.array(mean_sequence))

    return switching_sequence, correct_string, distance, switching_sequence == correct_string, mean_sequence, pred_switching_indices,true_switching_indices
   