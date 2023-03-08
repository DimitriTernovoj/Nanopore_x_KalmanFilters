import numpy as np
from . import KalmanFilter

def LookAhead(filter_set,obs,curr_state,obs_no_noise=None):
    '''Runs the Kalman Filter Look-Ahead Algorithm. \n
       Input: Filter-Set, Observations, Current State, Optional: Ground Truth \n
       If Ground Truth Sequence provided: \n
       Output: True Switching Sequence, Predicted Switching Sequence, Estimates, Prediction 100% Correct? (Boolean), L2-Distance between Prediction and Ground Truth, Indices of Predicted Switches, Indices auf True Switches, Most Likely States \n
       Else: \n
       Output: Predicted Switching Sequence, Estimates, Indices of Predicted Switches, Most Likely States
    '''
    most_likely_states = []
    most_likely_positions = []
    predicted_string = ""
    pred_switching_indices = []

    count = 0
    while count != len(obs):
        print(f"Current Count: {count}")
        probabilities = {}
        curr_filter = filter_set[curr_state]

        # Termination 1
        diff = len(obs) - count
        if diff < 6:
            for i in range(0,diff):
                most_likely_states.append(curr_state)
                most_likely_positions.append(curr_filter.mu)
                predicted_string += "0"

            count += diff
 
        # Termination 2
        elif diff >= 6  and diff < 12:
            probabilities["staying"] = curr_filter.simulateStaying(obs[count:count+diff])**(1/diff)

            #Simulate Transitions
            for i in range(diff-5):
                for filter in filter_set[:curr_state]+filter_set[curr_state+1:]:
                    m,P,l = KalmanFilter.Univariate_KalmanFilter(curr_filter.x,filter.mu,curr_filter.P, filter.Q, obs[count+i], filter.R)
                    probabilities[str(i)+str(filter.name)] = (curr_filter.cf_staying_prob[i] * l * filter.simulateStaying(obs[count+i:count+diff],True))**(1/diff)                
    
            #Infer Most Likely Result
            maximum = max(probabilities, key=probabilities.get)

            #No Switch
            if maximum == "staying":

                for i in range(0,diff):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(curr_filter.mu)
                    predicted_string += "0"

            #Switch
            else:
                for i in range(0,int(maximum[0])):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(curr_filter.mu)
                    predicted_string += "0"

                curr_state = int(maximum[1:])
                for i in range(0,diff-int(maximum[0])):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(filter_set[curr_state].mu)
                    predicted_string += "1" if i == 0 else "0"
                    if i == 0:
                        pred_switching_indices.append(count+int(maximum[0]))

            count += diff

        #Main-Loop
        else:

            probabilities["staying"] = curr_filter.simulateStaying(obs[count:count+6])**(1/6)

            #Simulate Transitions
            for i in range(0,6):
                for filter in filter_set[:curr_state]+filter_set[curr_state+1:]:
                    m,P,l = KalmanFilter.Univariate_KalmanFilter(curr_filter.x,filter.mu,curr_filter.P, filter.Q, obs[count+i], filter.R)
                    probabilities[str(i)+str(filter.name)] = (curr_filter.cf_staying_prob[i] *l * filter.simulateStaying(obs[count+i:count+i+6], True))**(1/(i+6))
    

            #Infer Most Likely Result
            maximum = max(probabilities, key=probabilities.get)

            #No Switch
            if maximum == "staying":
                for i in range(0,6):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(curr_filter.mu)
                    predicted_string += "0"

                count += 6

            #Switch
            else:
                for i in range(0,int(maximum[0])):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(curr_filter.mu)
                    predicted_string += "0" 

                curr_state = int(maximum[1:])
                for i in range(0,6):
                    most_likely_states.append(curr_state)
                    most_likely_positions.append(filter_set[curr_state].mu)
                    predicted_string += "1" if i == 0 else "0"
                    if i == 0:
                        pred_switching_indices.append(count+int(maximum[0]))

                count += int(maximum[0])+6

    #Information about Ground Truth Sequence inferred (if provided)
    if obs_no_noise != None:
        true_string = "0"
        true_switching_indices = []

        for i in range(1,len(obs_no_noise)):
            if obs_no_noise[i] != obs_no_noise[i-1]:
                true_string += "1"
                true_switching_indices.append(i)
            else:
                true_string += "0"

        distance = np.linalg.norm(np.array(obs_no_noise)-np.array(most_likely_positions))

        return true_string, predicted_string, most_likely_positions, true_string == predicted_string, distance, pred_switching_indices, true_switching_indices, most_likely_states
    
    else:
        return predicted_string, most_likely_positions, pred_switching_indices, most_likely_states

def LookAhead_Manager(curr_index,obs,model_mus,model_sigmas,obs_no_noise = None):
    '''Allows Running Kalman Filter Look-Ahead Algorithm on multiple Samples. \n
       Input: Current Sample Index, Observations, Model-Means, Model-Sigmas, Optional: Ground Truth Sequence \n
       Output: Collects and Returns Results from all Samples (used with multiprocessing)
    '''
    M = 1024
    name = np.arange(M)
    filter_set = []
    Q = 0

    start = -1
    threshold = np.inf

    #Synthetic Data
    if obs_no_noise == None:
        mean_start = np.mean(obs[curr_index][:6])
        threshold = np.inf
        for j, mu in enumerate(model_mus):
            dist_f = abs(mean_start-mu)

            if dist_f < threshold:
                start = j
                threshold = dist_f

            filter_set.append(KalmanFilter.KalmanFilter_class(model_mus[j],model_mus[j],model_sigmas[j],Q,model_sigmas[j],name[j]))

        predicted_string_, mlp_, pred_switching_indices_, mls_ = LookAhead(filter_set,obs[curr_index],start)
        return predicted_string_, mlp_, pred_switching_indices_, mls_


    #Real Data
    else:
        for j, mu in enumerate(model_mus):
            if mu == obs_no_noise[curr_index][0]:
                start = j
            
            filter_set.append(KalmanFilter.KalmanFilter_class(model_mus[j],model_mus[j],model_sigmas[j],Q,model_sigmas[j],name[j]))

        true_string_, predicted_string_, mlp_, match_, distance_, pred_switching_indices_, true_switching_indices_, mls_ = LookAhead(filter_set,obs[curr_index],start,obs_no_noise[curr_index])
        return true_string_, predicted_string_, mlp_, match_, distance_, pred_switching_indices_, true_switching_indices_, mls_




