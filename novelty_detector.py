#!/usr/bin/env python
"""
Created : 06-05-2018
Last Modified : Thu 10 May 2018 01:36:25 PM EDT
Created By : Enrique D. Angola
"""
import numpy as np
import signal_library as thm
from multiprocessing import Pool
from functools import partial
import seaborn as sns #remove if you dont have seaborn
import pdb

#######################################################
#########PRE-PROCESSING FUNCTIONS######################
#######################################################
def get_filenames(directory):   #pragma: no cover
    """
    automatically get filenames from a specific directory
    """
    import os, os.path
    datafiles = [name for name in os.listdir(directory)] # scan and see name of datafiles
    datafiles.sort()
    filename = [directory + i for i in datafiles]

    return filename

def read_wav(filename):
    """
    reads a .wav file and creates a thm signal object.

    Parameters
    ----------
    filename: string
        path to data

    Returns
    -------
    sig: signal library object
        data type that allows signal processing methods from thm module.

    """

    from scipy.io.wavfile import read
    sr,signal = read(filename)
    sig = thm.signal(signal)
    return sig

def add_signals(sig,signals):
    """
    appends signals to sig object

    Parameters
    ----------
    sig: signal library object
        original signal
    signals: list
        list of thm signal objects

    Returns
    -------
    sig: signal library object
        signal with appended signals

    """


    for sig2 in signals:
        sig.add_signal(sig2[:])

    return sig

def cut_signal(sig=None,chunkSize=10,sr=44100,totalTime=None):
    """
    Cuts the signal into chunks of different size in seconds

    Parameters
    ----------
    sig: signal library object
        signal to cut
    chunkSize: int
        size of chunks in seconds
    totalTime: int

    Returns
    -------
    signals: List
        list of the chunks of signal library objects

    """
    signals = [sig[int(i):int(i+(chunkSize*sr))] for i in \
            range(0,totalTime*sr,chunkSize*sr)]
    signals = [thm.signal(i) for i in signals]

    return signals

def sample_chunks(signals,chunkTimes=1,totalTime=10,sr=44100):
    """
    Create a new signal by taking chunks of other signals
    taking a sample from each signal creates a better random
    pattern

    Parameters
    ----------
    signals: List
        list of signals to sample from
    chunkTimes: int
        time in seconds of each chunk to sample
    totalTime: int
        total time of end signal
    sr: int
        sampling rate

    Returns
    -------
    signal library signal object
        signal of totalTime seconds with the random
        samples

    """
    import random
    signal = np.zeros(totalTime*chunkTimes*sr)
    i = 0
    for sample in signals:
        rand = random.randint(0,totalTime-chunkTimes)
        tmp = sample[rand*sr:(chunkTimes+rand)*sr]
        signal[i*sr:(i+chunkTimes)*sr] = tmp
        i = i + chunkTimes

    return thm.signal(signal)

def pick_random(signals,howMany = 10):

    import random
    indices = [random.randint(0,len(signals)-1) for i in range(howMany)]
    trainingSet = [signals[i] for i in indices]

    return indices,trainingSet


def fft(chunk,dt):
    """
    Computes fast fourier transform of thm signal object, function is part of spectrogram to compute
    the short time fft in paralell.

    Parameters
    ----------
    chunk: signal library signal object
        chunk of time-domain signal to compute fft on
    dt: float
        total time of signal, used to compute the sampling rate.

    Returns
    -------
    result: numpy array
        fourier transform of chunk

    Examples
    --------
    >>>
    """
    #make sure signal is a signal library type object
    sig = thm.signal(chunk)
    #compute fast fourier transform for 0.01 seconds
    fft_chunk = sig.get_fft(dt)
    #extract fourier energy
    result = np.asarray(fft_chunk['fourier'])
    return result


def spectrogram(data,dt=0.01,sr=44100,totalTime=10):
    """
    Computes short-time FFT of an audio signal.

    Parameters
    ----------
    data: signal library signal object
        signal to compute spectrogram
    dt: float
        delta t size of window to compute short time FFT
    sr: integer
        sampling rate of data.
    Returns
    -------
    fft_chunks: numpy array
        spectrogram of data. spectrum of frequencies of sound as it varies with time.

    """
    #make the signal be exactly totalTime seconds
    data.zero_pad(sr=sr,sg=totalTime)
    #define step for short time ffts
    step = int(dt*sr)
    chunks = [data[i:i+step] for i in range(0,int(len(data[:])/step))]
    pool = Pool(processes = 4)

    #partial creates a partial function for one variable
    partialfft = partial(fft,dt=dt)
    fft_chunks = pool.map(partialfft,chunks, 1)
    pool.close()

    return fft_chunks


#######################################################
#########Kernel-density estimation#####################
#######################################################
from sklearn.neighbors import KernelDensity as kd

def multiprocess_estimator(arg, **kwarg): #pragma: no cover
    """This function is called from the class parzen_network
        to run the novelty detector in parallel processing
        this is a work around for the multiprocessing library,
        if you try to multiprocess from within the class it gives
        a pickle error"""
    return parzen_network.__run_estimator__(*arg,**kwarg)

class parzen_network():

    """
    creates an instance of parzen_network class
    The class implements the novelty detector

    Parameters
    ----------
    nodes: int
        number of nodes in novelty detector
    bw: list
        list of bandwidths, one for each node
    timeSamples: int
        total number of time samples
    n_jobs: int
        number of processors to run in parallel

    Returns
    -------
    instance of class parzen_network

    """

    def __init__(self,nodes = 220,bw = None,timeSamples=1000,n_jobs=18):
        self.__nodes = nodes #global var. (in class)
        if bw:
            self.__bw = bw
        else:
            self.__bw = [0.75 for i in range(nodes)]
        self.__timeSamples = timeSamples
        self.__parzenNetwork = []
        self.n_jobs = n_jobs

    def train_1D(self,data):

        """
        trains the novelty detector

        Parameters
        ----------
        data: numpy array
            training data

        Returns
        -------
        None

        """

        bw = self.__bw
        nodes = self.__nodes
        timeSamples = self.__timeSamples
        if nodes !=1:
            #transpose so that each pattern is one vector in time domain. 
            X = np.transpose(data)
            #reshape elements intro proper [time_samples,1] shape for 1D: refer 
            #to kernel density est. scikit learn documentation
            elements = [np.reshape(e,[len(e),1]) for e in X]
            #create kernels, one for each element, with their individual bandwidth!
            kernels = [kd(bw[i]) for i,e in enumerate(elements)]
            #create a parzen network, containing 1 parzen trained with each element
            parzens = [kernels[i].fit(e) for i,e in enumerate(elements) ]
            self.__parzenNetwork = parzens
        #this is case when there is only ONE pattern useful for estimating PDF
        #it doesnt need to be tested for now, as it is not being used.
        else:   #pragma: no cover 
            X = np.reshape(data,[timeSamples,nodes])
            parzen = kd(bw)     #create the kernel density object
            parzen.fit(X)
            self.__parzenNetwork = [parzen] #this is really not a "network" but only one PDF

    #private method does not need to be included in coverage, its functionallity is technically
    #already tested by other tests, and it is designed so the code can be run in parallel
    def __run_estimator__(self,netAndPattern):  #pragma: no cover
        """
        private method to run the estimator, return the score (log(likelihoods))
        """
        testPattern = np.reshape(netAndPattern[1],[len(netAndPattern[1]),1])
        net = netAndPattern[0]
        a = net.score(testPattern)  #total log-likelihood
        return a

    def estimate_1D(self,testPattern):
        """
        estimates the log(likelihood) of a test pattern for all nodes
        """
	#transpose to properly feed into algorithm
        testPattern = np.transpose(testPattern)
	#zip all different nodes for parallel processing
        netAndPatterns = zip(self.__parzenNetwork,testPattern)
        pool = Pool(processes = self.n_jobs)
	#compute results in parallel
        results = pool.map(multiprocess_estimator,zip([self]*self.__nodes,netAndPatterns))
        pool.close()
        return results

    #estimate a probability density function, used for plotting.
    #does not need to be tested yet
    def estimate_pdf(self,data,spaceLen = 1000,Bin=0): #pragma: no cover
        """
        estimates a probability density function for Bin = Bin
        used for plotting.
        """
        inputSpace = np.linspace(min(data),max(data),spaceLen)
        inputSpace = np.reshape(inputSpace,[len(inputSpace),1])
        a = self.__parzenNetwork
        b = a[Bin].score_samples(inputSpace)
        pdf = [inputSpace,b]
        return pdf

def find_bw(sample,grid=np.logspace(-1,1,100),n_jobs=18):
    """
    Finds optimal bandwidth using grid search cross validation
    and the log(likelihood) estimate as a score. This method
    is called pseudo-log likelihood.

    Parameters
    ----------
    sample: List
        List of short time ffts in subject
    grid: numpy array
        grid of bandwidths to try on cross validation
    n_jobs: int
        how many processors to run in parallel

    Returns
    -------
    bw: List
        List of bandwidths for each node

    """

    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    params = {'bandwidth':grid}
    grid = GridSearchCV(KernelDensity(), params,n_jobs=n_jobs)
    sample = np.asarray(sample)
    shape = np.shape(sample)
    bw = []
    for i in range(0,shape[1]):
        chunk = sample[:,i]
        chunk = chunk.reshape(shape[0],1)
        grid.fit(chunk)
        kde = grid.best_estimator_
        bw.append(kde.bandwidth)

    return bw

#######################################################
#########Post-processing results#######################
#######################################################

def find_treshold(results,sigma = 3):
    """
    Compute treshold for outliers

    Parameters
    ----------
    results: Numpy array
        results from test samples
    sigma: int
        Z score

    Returns
    -------
    t: List
        list of tresholds for each node

    """
    import scipy.stats as st

    t = []

    for node in range(len(results[0])):
        mean = np.mean(results[:,node])
        sd = np.std(results[:,node])
        tresh1 = mean+sigma*sd
        tresh2 = mean-sigma*sd
        t.append([tresh1,tresh2])

    return t

def find_novelties(inputs,t):
    """
    find novelties given inputs and treshold
    """
    novel = np.zeros(len(t))
    for node in range(len(t)):
        t1 = inputs[node] > t[node][0]
        t2 = inputs[node] < t[node][1]
        if t1 or t2:
            novel[node] = 1
    return novel

def get_novelty_metrics(sample,treshold,chunkSize = 10):
    """
    returns measures from the novelty detector, necessary to calculate 
    novelty scores for a particular test sample.

    Parameters
    ----------
    sample: numpy array
        test sample, results obtained by parzen_network.estimate_1D
    treshold: list
        treshold, output from find_treshold.

    Returns
    -------
    noveltyMetrics: Dictionary
        phi: list
            contains log-likelihoods for novel nodes.
        phiTotal: list
            each element in list contains total number of nodes that
            raised novelties from test sample.
        tresholdIndex: list
            contains the tresholding log likelihood values for
            those nodes that raised novelties.
    Examples
    --------
    >>>
    """
    phi = []
    phiTotal = []
    tresholdIndex = []
    treshold = np.asarray(treshold)

    for i in range(0,len(sample),chunkSize):
        chunk = np.asarray(sample[i:i+chunkSize])
        n = find_novelties(chunk,treshold[i:i+chunkSize,:])
        phiIndexes = np.where(n==1)[0]
        tresholdChunk = treshold[i:i+chunkSize,-1]   #only taking the lower boundary, careful.
        tresholdIndex.append(tresholdChunk[phiIndexes])
        phi.append(chunk[phiIndexes])
        phiTotal.append(len(phiIndexes))

    noveltyMetrics = {'phi':phi,'phiTotal':phiTotal,'tresholdIndex':tresholdIndex}

    return noveltyMetrics

def calcTNS(sample):
    """
    calculates INRD and TNS

    Parameters
    ----------
    sample: Dict
        output from get_novelty_metrics

    Returns
    -------
    TN: Dict
        INRD: list
            individual node relative differences for those that raised novelties
        TNS: list
            sums the node relative differences for chunks

    """

    tresholdIndex = sample['tresholdIndex']
    phiTotal = sample['phiTotal']
    phi = sample['phi']
    INRD = [abs((chunk-tresholdIndex[i])/tresholdIndex[i]) for i,chunk in enumerate(phi)]
    TNS = [sum(chunk) for chunk in INRD]
    TN = {'INRD':INRD,'TNS':TNS}

    return TN

#doesn't need to be testes
def write_csv(filename,samples): #pragma: no cover
    """
    Export results to a csv file

    Parameters
    ----------
    filename: str
        path to file to export
    samples: list
        results of TNS or INRD to export

    Returns
    -------
    None

    """
    import csv
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter =' ', quotechar = '|')
        [writer.writerow(i) for i in samples]


if __name__ == '__main__':  #pragma: no cover
   ###########################################
   ######PRE-PROCESSING#######################
   ##########################################
   #define filename for novelty signal
   filename = '/home/enrique/Documents/audioresearch/AudioResearch/5min04262018shaftimbalance.wav'
   data = read_wav(filename)
   #cut the signals in chunks for training
   data = cut_signal(data,chunkSize=10,totalTime=60*5)
   #compute stft for novel data
   fftChunksNovel = [spectrogram(dat) for dat in data]

   directory = '/media/enrique/Backups and data/audiodata/02092018/normal/'
   filename = get_filenames(directory)[0:]
   data = [read_wav(i) for i in filename]

   #calculate stft for healthy data
   fftChunksHealthy = [spectrogram(dat) for dat in data]
   indices,signals = pick_random(data)
   trainingSet = sample_chunks(signals,chunkTimes=1)
   trainingSet = spectrogram(trainingSet,totalTime=10)

   ##################################################
   ####################MACHINE LEARNING##############
   ##################################################
   bw = find_bw(trainingSet)
   p = parzen_network(bw=bw,nodes=220)
   p.train_1D(trainingSet)
   #pop chunks used for training
   fftChunksHealthy = np.delete(fftChunksHealthy,indices,0)

   results = [p.estimate_1D(sample) for sample in fftChunksHealthy]
   resultsNovel = [p.estimate_1D(sample) for sample in fftChunksNovel]
   #find treshold
   #pick data for treshold
   indices,forTreshold = pick_random(results,howMany=30)
   t = find_treshold(np.asarray(forTreshold))
   ##################################################################
   #######################POST-PROCESSING############################
   ##################################################################
   #get metrics
   #choose chunksize=1 since you want all nodes
   metrics = [get_novelty_metrics(i,t,chunkSize=1) for i in results[:]]
   metricsNovel = [get_novelty_metrics(i,t,chunkSize=1) for i in resultsNovel[:]]
   metricsHealthy = [calcTNS(sample) for sample in metrics]
   metricsNovel = [calcTNS(sample) for sample in metricsNovel]
   TNS = [metricsHealthy[i]['TNS'] for i in range(len(metricsHealthy))]
   TNS2 = [metricsNovel[i]['TNS'] for i in range(len(metricsNovel))]
   #SAVE RESULTS
   write_csv('healthyres.csv',TNS)
   write_csv('novelres.csv',TNS2)
