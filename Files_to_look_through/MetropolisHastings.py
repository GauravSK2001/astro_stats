import numpy as np

def SimpleMetropolisHastingsASTM21Step(LogPdfFunction, ProposalStepCovarienceMatrix, 
                                       CurrentParameters) :
    '''Takes a step in a Metropolis-Hastings algorithm
    
    Input:
    LogPdfFunction: Function that return log posterior probability if given n-dimensional 
        parameter array
    ProposalStepCovarienceMatrix: n x n covarience matrix (numpy array) which defines the 
        proposal step (step chosen from multivariate Gaussian with this covarience)
    CurrentParameters: n-dimensional numpy array giving current parameters

    Returns:
    NewParameters: n-dimensional numpy array giving parameters for next point in the chain 
        (may be the same as previous)
    '''
    # generate proposal state
    ProposalParameters = np.random.multivariate_normal(CurrentParameters, ProposalStepCovarienceMatrix)
    
    # Note that the below is inefficient - should re-use old value of posterior probability
    OldLogPdfValue = LogPdfFunction(CurrentParameters)
    NewLogPdfValue = LogPdfFunction(ProposalParameters)
    # Make step if better
    if NewLogPdfValue > OldLogPdfValue : 
        return ProposalParameters
    else :
        # Allow for possible step if worse
        tester = np.random.rand() # U(0,1)
        if tester < np.exp(NewLogPdfValue-OldLogPdfValue) : 
            return ProposalParameters
        else :
            return CurrentParameters
        
def SimpleMetropolisHastingsASTM21(LogPdfFunction, ProposalStepCovarienceMatrix, InitialValue, 
                                   NumberofValuesinChain) :
    
    '''Creates a MCMC chain of the required size and shape using the Metropolis-Hastings alogorithm
    
    We assume that there are n paramaters. The user must provide a function which gives the (natural) 
    logarithm of the pdf given an input n-dimensional array, a propsal covarience matrix, and an 
    initial guess at the paramters.
    
    Input:
    LogPdfFunction: Function that return log posterior probability if given n-dimensional 
        parameter array
    ProposalStepCovarienceMatrix: n x n covarience matrix (numpy array) which defines the 
        proposal step (step chosen from multivariate Gaussian with this covarience)
    CurrentParameters: n-dimensional numpy array giving current parameters
    NumberofValuesinChain: Length of the output MCMC chain (positive integer)

    Returns:
    OutputChain: (NumberofValuesinChain,n) dimensional array giving the output MCMC chain
    '''
    
    OutputChain = np.zeros((NumberofValuesinChain,len(InitialValue)))
    OutputChain[0] = InitialValue

    #Run Metropolis Hastings
    for i in range(1,NumberofValuesinChain) :
        OutputChain[i] = SimpleMetropolisHastingsASTM21Step(LogPdfFunction, ProposalStepCovarienceMatrix, 
                                                            OutputChain[i-1])
    return OutputChain