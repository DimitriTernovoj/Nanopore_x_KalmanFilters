import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def datageneration(n_switches, total_seq_len):
    #randomly generate samples

    M = 6 # number of models
    mus_ = np.random.permutation(np.arange(20))[:M] * 10. # minimal 10 difference between two mu vals. Get M many different values from 0 to 200 in steps of 10
    sigmas_ = np.abs(np.random.randn(M,)) * 10. ## sample M many values from a standard normal distribution with a variance of 10

    z = np.random.choice(np.arange(M), size=n_switches+1)

    #fill the sequence to the desired length
    pos_to_fill = total_seq_len - (n_switches+1) * 6
    
    nums = np.full((len(z),),6)
    for i in range(pos_to_fill):
        index = np.random.randint(0,len(nums))
        nums[index] += 1

    zs = np.repeat(z,nums)
    obs_no_noise = np.repeat(mus_[z],nums)
    sigmas = np.repeat(sigmas_[z],nums)
    obs = np.random.normal(loc=obs_no_noise, scale=sigmas)
    
    return obs_no_noise, obs, sigmas, zs, mus_, sigmas_

if __name__ == "__main__":
    #parameters
    data_frame = "C:/Users/Dima/Desktop/TestFolder/500_Samples_500Dense.csv"
    image_path = "C:/Users/Dima/Desktop/TestFolder/"
    image_base_name = "500_Samples_500Dense_toy"
    number_of_samples = 2
    number_of_switches = 81
    desired_seq_len = 500
    seed = 0

    obs = []
    obs_no_noise = []
    sigmas = [] 
    zs = []
    model_mus = []
    model_sigmas = []

    for i in range(number_of_samples):
        print(i)
        np.random.seed(i+seed)
        obs_no_noise_, obs_, sigmas_, zs_,model_mus_,model_sigmas_ = datageneration(number_of_switches,desired_seq_len)

        obs_no_noise.append(obs_no_noise_.tolist())
        obs.append(obs_.tolist())
        sigmas.append(sigmas_.tolist())
        zs.append(zs_.tolist())
        model_mus.append(model_mus_.tolist())
        model_sigmas.append(model_sigmas_.tolist())

    #safe csv-file
    data_df = pd.DataFrame(list(zip(obs,obs_no_noise,sigmas,zs,model_mus,model_sigmas)), columns=["obs","obs_no_noise","sigmas","states","model_mus","model_sigmas"])
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

    


