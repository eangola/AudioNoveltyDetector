#!/usr/bin/env python
"""
Created : 07-05-2018
Last Modified : Wed 09 May 2018 01:43:20 PM EDT
Created By : Enrique D. Angola
"""
import novelty_detector as nd
import numpy as np
import signal_library as sl

class test_preprocessing():

    def setup(self):
        t = np.linspace(0,1,44100)
        signal = 5*np.sin(2*np.pi*6*t)
        self.sin = sl.signal(signal)
        self.signal = nd.read_wav('testfile.wav')
        self.signal.zero_pad(sg=1,sr=44100)

    def test_add_signals(self):
        signals = [self.sin for i in range(0,5)]
        sig = nd.add_signals(self.signal,signals)
        assert len(sig[:]) == 6*44100

    def test_cut_signal(self):
        signals = [self.sin for i in range(0,6)]
        sig = nd.add_signals(self.signal,signals)
        chunks = nd.cut_signal(sig,chunkSize=2,totalTime = 6)

        assert len(chunks) == 3 and \
                len(chunks[0][:]) == 2*44100

    def test_sample_chunks(self):
        t = np.linspace(0,10,44100*10)
        signals = [i*np.sin(2*np.pi*6*t) for i in range(1,11)]
        signal = nd.sample_chunks(signals)

        assert len(signal[:]) == 10*44100

    def test_spectrogram(self):
        t = np.linspace(0,10,44100*10)
        signal = 100*np.sin(2*np.pi*1000*t)

        fft_chunks = nd.spectrogram(sl.signal(signal))
        assert len(fft_chunks) == 10/0.01 and\
                np.argmax(fft_chunks[0]) == 10\
                and np.ceil(max(fft_chunks[0])) == 100

class test_novel():

    def setup(self):

        signal = nd.read_wav('testfile.wav')
        self.pattern = nd.spectrogram(signal,totalTime=1)
        self.network = nd.parzen_network()

    def test_bandwidth(self):
        bw = nd.find_bw(self.pattern,grid=np.logspace(-1,1,10))
        assert len(bw) == 220

    def test_detector(self):
        self.network.train_1D(self.pattern)
        r = self.network.estimate_1D(self.pattern)
        assert len(r) == 220

class test_find_treshold():

    def setup(self):
        self.sample = np.asarray([[-3,0,-4.1,-2],[-3,0,-4.1,-2]])

    def test_treshold(self):
        results = nd.find_treshold(self.sample)
        assert results[0] == [-3.0,-3.0] and results[-1] == \
                [-2.0,-2.0]

class TestMetrics():

    def setup(self):
        self.treshold = [[1.5,-2], [2,-3], [5,-4], [1,-1.2]]
        self.sample = np.array([-3,0,-4.1,-2])
        self.noveltyMetrics = nd.get_novelty_metrics(self.sample,self.treshold,chunkSize=2)
        self.INRD = nd.calcTNS(self.noveltyMetrics)['INRD']

    def test_get_novelty_metrics_phi(self):
        assert self.noveltyMetrics['phi'][0][0] == -3 and self.noveltyMetrics['phi'][1][0] == -4.1 and \
        self.noveltyMetrics['phi'][1][1] == -2

    def test_get_novelty_metrics_phiTotal(self):
        assert self.noveltyMetrics['phiTotal'] == [1,2]

    def test_get_novelty_metrics_tresholdIndex(self):
        assert self.noveltyMetrics['tresholdIndex'][1][0] == -4 and \
        self.noveltyMetrics['tresholdIndex'][1][1] == -1.2 and self.noveltyMetrics['tresholdIndex'][0][0] == -2


    def testt_calcTNS(self):
        from nose.tools import assert_almost_equal
        assert_almost_equal(self.INRD[0][0],0.5) and assert_almost_equal(self.INRD[1][0],0.025) and \
        assert_almost_equal(self.INRD[1][1],0)

    def test_find_novelties(self):
        nd.find_novelties(self.sample,self.treshold) == np.asarray([1,0,1,1])

class test_random_pick():

    def setup(self):
        t = np.linspace(0,1,44100)
        signal = 5*np.sin(2*np.pi*6*t)
        self.signals = [signal for i in range(90)]

    def test_function(self):
        indices,training = nd.pick_random(self.signals,howMany=13)

        assert len(training) == 13


