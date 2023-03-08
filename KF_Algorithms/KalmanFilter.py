from scipy.stats import norm

def Univariate_KalmanFilter(x,mu,P,Q,z,R):
    '''Runs 1 Iteration of Kalman Filtering. \n
       Input: Current Estimate, Expected Mean, Uncertainty of Current Estimate, Process Noise, Measurement, Measurement Noise \n
       Output: Estimate for the next Time Point, Uncertainty of the Estimate, Likelihood of the Estimate
    '''
    #prediction
    x = mu
    P = P + Q

    #update
    y = z - x
    K = P/(P+R)

    x = x + K*y
    P = (1-K)*P

    likelihood = norm.pdf(y, 0, P+R)

    return x, P, likelihood 

class KalmanFilter_class:
    def __init__(self,x,mu,P,Q,R,name,w=0):
        #basics
        self.x = x
        self.mu = mu
        self.P = P
        self.Q = Q
        self.R = R
        self.name = name

        #iterations
        self.likelihoods = []
        self.means = []
        self.vars = []
        self.w_ij_numerator = []
        self.w_ij = []
        self.w = w
        self.m_ij = []
        self.cf_staying_prob = []

    def reset(self):
        '''
        Resets Iteration Variable of the Kalman Filter.
        '''
        self.likelihoods = []
        self.means = []
        self.vars = []
        self.w_ij_numerator = []
        self.w_ij = []
        self.m_ij = []
    
    def simulateStaying(self, measurements, reset = False):
        '''Simulates Staying in a Kalman Filter.
           Input: Kalman Filter, Measurements, Reset results or not
           Output: List of Scores representing the Likelihood of Staying in the current Filter after every given Measurement
        '''
        staying_prob = 1
        self.cf_staying_prob = [1]
        for measurement in measurements:
            m,P,l = Univariate_KalmanFilter(self.x,self.mu,self.P, self.Q, measurement, self.R)
            staying_prob *= l
            self.cf_staying_prob.append(staying_prob)
        
        if reset:
            self.cf_staying_prob = [1]

        return staying_prob