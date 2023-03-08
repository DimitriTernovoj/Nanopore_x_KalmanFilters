import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def datageneration(KMER_MODELs, n_switches, total_seq_len):
    #randomly generate samples
    count=0
    z = [0]
    z_quaternary = [0]
    last_state = "0"
    #A = 0, C = 1, G = 2, T = 3
    while count < n_switches:
        val = np.random.randint(0,4)

        #for the initial few positions
        if len(last_state) != 5:
            last_state = last_state + str(val)
            z.append(int(last_state,4))
            z_quaternary.append(last_state)
        #for the remaining ones
        else:
            last_state = last_state[1:] + str(val)
            z.append(int(last_state,4))
            z_quaternary.append(last_state)   
        count += 1

    #subset the KMER_MODELs according to the chosen models
    df_subset = KMER_MODELs.iloc[z,0:4]

    #fill the sequence to the desired length
    pos_to_fill = total_seq_len - (n_switches+1) * 6
    
    nums = np.full((len(z),),6)
    for i in range(pos_to_fill):
        index = np.random.randint(0,len(nums))
        nums[index] += 1

    mus = np.repeat(df_subset.level_mean.to_numpy(),nums)
    sigmas = np.repeat(df_subset.level_stdv.to_numpy(),nums)
    kmers = np.repeat(df_subset.kmer.to_numpy(),nums)
    sequence = "".join([x[0] for x in df_subset.kmer])
    z = np.repeat(z, 6)
    z_quaternary = np.repeat(z_quaternary, 6)
    obs = np.random.normal(loc=mus, scale=sigmas)
    

    return obs, mus, sigmas, z, z_quaternary, kmers, sequence

if __name__ == "__main__":
    #parameters
    data_frame = "C:/Users/Dima/Desktop/TestFolder/30len.csv"
    image_path = "C:/Users/Dima/Desktop/TestFolder/"
    image_base_name = "500_Samples_180Dense"
    number_of_samples = 2
    number_of_switches = 3
    desired_seq_len = 30
    seed = 0
    KMER_MODELs = pd.read_csv("C:/Users/Dima/Desktop/TestFolder/template_median69pA.model", sep="\t", index_col="kmer_enc2")

    obs = []
    obs_no_noise = []
    sigmas = [] 
    indices = []
    kmers = []
    sequence = []
    indices_quaternary = []

    for i in range(number_of_samples):
        print(i)
        np.random.seed(i+seed)
        obs_, obs_no_noise_, sigmas_, indices_,indices_quaternary_, kmers_, sequence_ = datageneration(KMER_MODELs,number_of_switches,desired_seq_len)

        obs.append(obs_.tolist())
        obs_no_noise.append(obs_no_noise_.tolist())
        sigmas.append(sigmas_.tolist())
        indices.append(indices_.tolist())
        indices_quaternary.append(indices_quaternary_.tolist())
        kmers.append(kmers_.tolist())
        sequence.append(sequence_)
    
    #safe csv-file
    data_df = pd.DataFrame(list(zip(sequence,kmers,[len(sequence_)]*len(sequence),indices,indices_quaternary,obs,obs_no_noise,sigmas)), columns=["sequence","kmers","sequence length","indices","indices_quaternary","obs","obs_no_noise","sigmas"])
    data_df.to_csv(data_frame, index = False)

    #generate the image path if it doesn't exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)
        
    #generate images
    plt.figure(figsize=(10,4))
    for i in range(len(obs)):
        plt.plot(obs[i], label='observations')
        plt.plot(obs_no_noise[i], c='r', label='no noise')
        plt.ylabel('[measurement unit]')
        plt.xlabel('t')
        plt.legend()
        plt.savefig(image_path+image_base_name+"_"+ str(i) + ".jpg", bbox_inches="tight")
        plt.cla()

    


