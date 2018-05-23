#!/usr/bin/env python
"""
Created : 06-05-2018
Last Modified : Mon 07 May 2018 07:01:42 PM EDT
Created By : Enrique D. Angola
"""

import numpy as np
import signal_library as sl


class tests():

    def setup(self):
        t = np.linspace(0,10,1000)
        signal = 5*np.sin(2*np.pi*6*t)
        self.signal = sl.signal(signal)

    def test_get_fft(self):

        fft = self.signal.get_fft(total_time = 10)
        fft2 = self.signal.get_fft(sr=100)
        fourier = fft['fourier']
        fourier2 = fft2['fourier']
        freq = fft['freq']
        freq2 = fft2['freq']
        index = np.where(fft['freq'] == 6)[0]

        assert np.ceil(fourier[index]) == 5.0 and \
                np.ceil(fourier2[index]) == 5.0

    def test_add_signal(self):
        t = np.linspace(0,10,1000)
        signal2 = 10*np.sin(2*np.pi*6*t)
        self.signal.add_signal(signal2)
        sig = self.signal[:]
        assert len(sig) == 2000 and \
                np.ceil(max(sig[:])) == 10

    def test_zero_pad(self):
        signalTime = 20
        signalTime2 = 5
        self.signal.zero_pad(sr=100,sg=signalTime)

        print(self.signal[-1])
        assert len(self.signal[:]) == signalTime*100 and \
                self.signal[-1] == 0

        self.signal.zero_pad(sr=100,sg=signalTime2)

        assert len(self.signal[:]) == signalTime2*100

