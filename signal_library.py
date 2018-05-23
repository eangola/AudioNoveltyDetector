#!/usr/bin/env python
"""
Created : 06-05-2018
Last Modified : Thu 10 May 2018 10:55:37 AM EDT
Created By : Enrique D. Angola
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:00:37 2014

@author: eangola
"""

import numpy as np
from scipy import signal as sg




####################################################
##############Class signal##########################
####################################################

class signal():
    """
    Generates an object that represents the selected signal.


    Parameters
    ----------

    signal : list

    Returns
    -------
    signal: Class object that represents the signal

    Examples
    --------
    >>> signal = signal(sig)

    Call the class signal and generate object signal.

    """
    def __init__(self,signal):

        self.__signal = np.asarray(signal) #convert to numerical array

####################################################
##############Functions to manipulate signal########
####################################################
    def __getitem__(self,key):

        return self.__signal[key]



    def add_signal(self,signal):
        """
        concatanates signal with signal object

        Parameters
        ----------
        signal: Numpy Array
            signal to concatanate

        Returns
        -------
        None

        """

        sCombined = list(self.__signal[:]) + list(signal)
        self.__signal = sCombined

    def zero_pad(self,sr,sg):
        """
        Zero pads the signal, adds zeros or cuts
        to achieve a specific total time


        Parameters
        ----------
        sg: int
            signal time desired
        sr: float
            sampling rate of the signal

        Returns
        -------
        None

        """
        # get length of the signal
        L = len(self.__signal[:])
        #if the length is less than desired time, append 0s
        if L < sg*sr:
            sig_temp = np.zeros(sg*sr)
            sig_temp[0:len(self.__signal[:])] = self.__signal[:]
        #if the length is more than designed time, cut it.
        else:
            sig_temp = self.__signal[0:sg*sr]

        self.__signal = sig_temp

    def get_fft(self,total_time=False,sr=False,\
            half_window=True,normalize = True,pad= 0,window=False):

        """
        Performs fast fourier transform on signal.
        choose between total_time, sample_rate, or clockspeed.

        Parameters
        ----------
        total_time : integer
            total time signal was collected. False is default
        sample_rate : integer
            sampling rate for which signal was collected. False is default
        clockspeed: integer
            time between samples. False is default
        half_window: Boolean
            choose False if you want to get both sides of the spectrum; True is default.
        normalizet_fft(: Boolean
            choose True if you want to normalize the amplitudes of the FFT for the size
            of the FFT window. False is default

        Returns
        -------
        FFT: class object
            class object is a dictionary.

        FFT class keys
        --------------
        freq: numpy array
            vector of frequencies for the FFT.
        fourier: numpy array
            vector of amplitudes for the FFT.

        See also:
        ---------

        module thm_FFT.py

        Examples
        --------
        >>> fft = signal.get_fft(sample_rate = 2000000,normalize = True)

        createas fft object. sample rate is 2MHz and the FFT is normalized
        """
        from scipy import fftpack as fft  # Package for FFT
        total_time = float(total_time)
        n = self.__signal.size #total amount of data points

        if total_time:
            timestep = total_time/n #time between each data point
        elif sr:
            timestep = 1.0/sr
        ###################
        try:
            fourier = fft.fft(self.__signal)
            fullfft = fourier
            freq = fft.fftfreq(n, d=timestep)
            fullfreq = freq

        except ZeroDivisionError: #divide by False; 0.

             print('Please enter total_time or sample_rate')

        else: #no error found

            if half_window: #collect only positive side of FFT
                half_n = int(n/2)
                fourier = fourier[:half_n]
                freq = freq[:half_n]

            if normalize: #recover original amplitude from time domain
                fourier = 2.0/n * abs(fourier)


            fft_dict = {'freq':freq,'fourier':fourier,'fullfft':fullfft,\
                    'fullfreq':fullfreq} #dictionary

            return fft_dict


