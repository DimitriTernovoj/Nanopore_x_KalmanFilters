import numpy as np
from . import KalmanFilter

def GPB2(filters,transition_matrix,measurements,count):
    '''Runs 1 Iteration of Switching Kalman Filters with GPB2 Collapsing.
       Input: Filter-Set, Transition Matrix, Observations, Index of current Observation
       Output: Most Likely State, Estimate, Likelihood of Result
    '''
    #Compute Transitions
    w_ij_denominator = 0
    for j,filter_curr in enumerate(filters):
        for i,filter_past in enumerate(filters):
            if  transition_matrix[i,j] != 0:
                m,P,l = KalmanFilter.Univariate_KalmanFilter(filter_past.x,filter_curr.mu,filter_past.P, filter_curr.Q, measurements[count], filter_curr.R)
                
                filter_curr.means.append(m)
                filter_curr.vars.append(P)
                filter_curr.likelihoods.append(l)
                filter_curr.w_ij_numerator.append(l * float(transition_matrix[i,j])* filter_past.w)

        w_ij_denominator += sum(filter_curr.w_ij_numerator)

    #Collapsing        
    most_likely_state = ""
    most_likely_position = float(0)
    threshold = -np.inf
    
    for filter in filters:
        if sum(filter.w_ij_numerator) != 0:
            filter.w_ij = np.array(filter.w_ij_numerator)/w_ij_denominator
            filter.w = sum(filter.w_ij)

            filter.m_ij = np.array(filter.w_ij/filter.w)

            # Collapsing all x_ij, P_ij into one mean x_i and variance P_i per model
            filter.x = np.dot(filter.m_ij,np.array(filter.means).T)
            filter.P = np.dot(filter.m_ij,(np.sum([filter.vars,filter.means], axis=0)-float(filter.x)).T)

            if filter.w > threshold:
                most_likely_position = filter.x
                most_likely_state = filter.name
                threshold = filter.w

            filter.reset()
        else:
            filter.reset()


    return most_likely_state, most_likely_position, threshold

def ForwardBackwardGPB2(filter_set_forward,filter_set_backward,transition_matrix_f,transition_matrix_b,obs, obs_no_noise=None):
    '''Forward-Backward Algorithm. \n
       Input: Forward Filter-Set, Backward Filter-Set, Forward Transition Matrix, Backward Transition Matrix, Observations, Optional: Ground Truth Sequence \n
       If Ground Truth Sequence provided: \n
       Output: Predicted Switching Sequence, True Switching Sequence, L2-Distance between Prediction and Ground Truth, Prediction 100% Correct? (Boolean), Indices of Predicted Switches, Indices auf True Switches, Most Likely States, Estimates, Forward-Backward Distribution \n
       Else: \n
       Output: Predicted Switching Sequence, Indices of Predicted Switches, Most Likely States, Estimates, Forward-Backward Distribution \n  
    '''
    #Initialization
    ForwardBackwardDistribution = []
    most_likely_states = np.full((len(obs)),0)
    most_likely_positions = np.full((len(obs)),0.0)

    backward_count = len(obs)-1
    forward_count = 0

    finished = False
    forward = True
    backward = True

    while not finished:

        if forward:
            forward_state, forward_pos, forward_prob = GPB2(filter_set_forward,transition_matrix_f,obs,forward_count)
        if backward:
            backward_state, backward_pos, backward_prob = GPB2(filter_set_backward,transition_matrix_b,obs,backward_count)

        if forward_prob >= backward_prob:
            print(f"forward with count: {forward_count}")
            backward = False
            forward = True
            most_likely_states[int(forward_count)] = forward_state
            most_likely_positions[int(forward_count)] = forward_pos
            forward_count += 1
            ForwardBackwardDistribution.append("F")
        else:
            print(f"backward with count: {backward_count}")
            forward = False
            backward = True
            most_likely_states[backward_count] = backward_state
            most_likely_positions[backward_count] = backward_pos
            backward_count -= 1
            ForwardBackwardDistribution.append("B")
        
        if backward_count == forward_count-1:
            finished = True

    if obs_no_noise != None:
        pred_switching_seq = "0"
        true_switching_seq = "0"
        pred_switching_indices = []
        true_switching_indices = []

        for i in range(1,len(most_likely_states)):
            if most_likely_states[i] != most_likely_states[i-1]:
                pred_switching_seq += "1"
                pred_switching_indices.append(i)
            else: 
                pred_switching_seq += "0"
            
            if obs_no_noise[i] != obs_no_noise[i-1]:
                true_switching_seq += "1"
                true_switching_indices.append(i)
            else:
                true_switching_seq += "0"
            
        distance = np.linalg.norm(obs_no_noise-most_likely_positions)
        match = pred_switching_seq == true_switching_seq

        return pred_switching_seq, true_switching_seq, distance, match, pred_switching_indices,true_switching_indices, most_likely_states.tolist(), most_likely_positions.tolist(), ForwardBackwardDistribution 
    
    else:
        pred_switching_seq = "0"
        pred_switching_indices = []

        for i in range(1,len(most_likely_states)):
            if most_likely_states[i] != most_likely_states[i-1]:
                pred_switching_seq += "1"
                pred_switching_indices.append(i)
            else: 
                pred_switching_seq += "0"

        return pred_switching_seq, pred_switching_indices, most_likely_states.tolist(), most_likely_positions.tolist(), ForwardBackwardDistribution

def Forward_Backward_Loop(curr_index,filter_set_forward,filter_set_backward,model_mus,model_sigmas,Z_f,Z_b,obs,obs_no_noise = None):
    '''Allows Running Forward-Backward Algorithm on multiple Samples. \n
       Input: Current Sample Index, Forward Filter-Set, Backward Filter-Set, Model-Means, Model-Sigmas, Forward Transition Matrix, Backward-Transition Matrix, Observations, Optional: Ground Truth Sequence \n
       Output: Collects and Returns Results from all Samples (used with multiprocessing)
    '''

    M = 1024
    name = np.arange(M)
    Q = 0 #process noise
    filter_set_forward = []
    filter_set_backward = []

    #Infer Start-State and End-State
    start_front = -1
    start_back = -1

    if obs_no_noise == None:
        mean_front = np.mean(obs[curr_index][:6])
        mean_back = np.mean(obs[curr_index][-6:])
        threshold_front = np.inf
        threshold_back = np.inf

        for i in range(len(model_mus)):
            dist_f = abs(mean_front-model_mus[i])
            dist_b = abs(mean_back-model_mus[i])

            if dist_f < threshold_front:
                start_front = i
                threshold_front = dist_f
            
            if dist_b < threshold_back:
                start_back = i
                threshold_back = dist_b

    else:  
        for index, mean in enumerate(model_mus):
            if start_front != -1 and start_back != -1:
                break
            if mean == obs_no_noise[curr_index][0]:
                start_front = index
            if mean == obs_no_noise[curr_index][-1]:
                start_back = index

    for i in range(len(model_mus)):

        if i == start_front:
            filter_set_forward.append(KalmanFilter.KalmanFilter_class(model_mus[i],model_mus[i],model_sigmas[i],Q,model_sigmas[i],name[i],1))
        else:
            filter_set_forward.append(KalmanFilter.KalmanFilter_class(model_mus[i],model_mus[i],model_sigmas[i],Q,model_sigmas[i],name[i],0))

        if i == start_back:
            filter_set_backward.append(KalmanFilter.KalmanFilter_class(model_mus[i],model_mus[i],model_sigmas[i],Q,model_sigmas[i],name[i],1))
        else:
            filter_set_backward.append(KalmanFilter.KalmanFilter_class(model_mus[i],model_mus[i],model_sigmas[i],Q,model_sigmas[i],name[i],0))

    if obs_no_noise != None:
        pred_switching_seq, true_switching_seq, distance, match, pred_switching_indices, true_switching_indices, most_likely_states, most_likely_positions, FB_dist = ForwardBackwardGPB2(filter_set_forward,filter_set_backward,Z_f,Z_b,obs[curr_index], obs_no_noise[curr_index])
        return pred_switching_seq, true_switching_seq, distance, match, pred_switching_indices, true_switching_indices, most_likely_states, most_likely_positions, FB_dist
    else:
        pred_switching_seq, pred_switching_indices, most_likely_states, most_likely_positions, FB_dist = ForwardBackwardGPB2(filter_set_forward,filter_set_backward,Z_f,Z_b,obs[curr_index])
        return pred_switching_seq,pred_switching_indices, most_likely_states, most_likely_positions, FB_dist


def Transition_Matrix(KMER_MODEL, obs, reference_sequence=None):
    '''Input: KMER_MODELs, Observations, Optional: Reference_Sequence.\n
       Use the Ground Truth Sequence as Observations if available, provide a Reference Sequence and the regular Observations otherwise.\n
       Output: Transition Matrix Forward, Transition Matrix Backward
    '''

    M = 1024
    index = KMER_MODEL.index

    if reference_sequence == None:
        expected_switches = 0
        for j in range(1,len(obs)):
            if obs[j] != obs[j-1]:
                expected_switches += 1
    else:
        expected_switches = len([reference_sequence[y-5:y] for y in range(5, len(reference_sequence)+1,1)])

    switching_prob = expected_switches/len(obs)

    #Forward Transition Matrix
    Z_f = np.zeros((M,M))

    #set values for valid switches
    possible_switches = []
    for i in index:
        i = str(i)

        if len(i) != 5:
            list = []
            for j in range(0,4):
                temp = i + str(j)
                list.append(int(temp,4))

            possible_switches.append(list)
        else:
            list = []
            for j in range(0,4):
                temp = i[1:] + str(j)
                list.append(int(temp,4))
            
            possible_switches.append(list)

    for i in range(0, len(index)):
        for j in possible_switches[i]:
            Z_f[i][j] = switching_prob/4
    
    np.fill_diagonal(Z_f,1 - switching_prob)

    #Backward Transition Matrix
    Z_b = np.zeros((M,M))

    #set values for valid switches
    possible_switches_b = []
    for i in index:
        i = str(i)

        if len(i) != 5:
            list = []
            for j in range(0,4):
                temp = str(j) + "".join(map(str,[0]*(5-len(i)))) + i[:-1]
                list.append(int(temp,4))

            possible_switches_b.append(list)
        else:
            list = []
            for j in range(0,4):
                temp = str(j) + i[:-1] 
                list.append(int(temp,4))
            
            possible_switches_b.append(list)

    for i in range(0, len(index)):
        for j in possible_switches_b[i]:
            Z_b[i][j] = switching_prob/4

    np.fill_diagonal(Z_b,1 - switching_prob)

    return Z_f, Z_b











