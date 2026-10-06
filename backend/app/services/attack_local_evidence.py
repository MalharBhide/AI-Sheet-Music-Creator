"""Observed attack-local evidence on a fixed clock, independent of note length."""

import numpy as np

from app.services.temporal_note_model import HOP as CQT_HOP
from app.services.temporal_note_model import OFFSETS, RATE, spectrum

VERSION = 'attack-local-cqt-stft-v1'
HOP = 128
STEPS = 61
CHANNELS = 18
WINDOWS = (1024,4096)
HARMONICS = (1,2,3,4)


def normalized(magnitude):
    peak = float(np.max(magnitude,initial=0.))
    if peak<=1e-12:
        return np.zeros_like(magnitude,dtype=np.float32)
    return np.clip((20*np.log10(np.maximum(magnitude,1e-12)/peak)+80)/80,0.,1.).astype(np.float32)


def observe(samples, rate):
    import librosa

    samples = np.asarray(samples,np.float32)
    if rate!=RATE or samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Attack evidence requires finite mono 22050 Hz audio')
    original_length = len(samples)
    if len(samples)<max(WINDOWS):
        samples = np.pad(samples,(0,max(WINDOWS)-len(samples)))
    result = {'duration':original_length/RATE,'cqt':spectrum(samples,rate)}
    for index,size in enumerate(WINDOWS):
        magnitude = np.abs(librosa.stft(samples,n_fft=size,hop_length=HOP,center=True,pad_mode='constant')).T
        result['stft_'+str(size)] = normalized(magnitude)
        if index==0:
            # Short-window flux captures new attacks without copying note
            # predictions, reference timestamps or precomputed model scores.
            peak = float(np.max(magnitude,initial=0.))
            scaled = magnitude/max(peak,1e-12)
            flux = np.mean(np.maximum(0.,np.diff(scaled,axis=0,prepend=scaled[:1])),axis=1)
            result['flux'] = (flux/max(float(np.max(flux,initial=0.)),1e-12)).astype(np.float32)
    return result


def sequences(observed, events):
    events = np.asarray(events,float)
    if (events.shape!=(len(events),4) or not np.isfinite(events).all()
            or np.any(events[:,0]<0) or np.any(events[:,1]<=events[:,0])
            or np.any(events[:,1]>observed['duration']+1e-6)
            or np.any((events[:,2]<21)|(events[:,2]>108))
            or np.any(events[:,2]!=np.floor(events[:,2]))):
        raise ValueError('Invalid attack-local note intervals')
    arrays = [observed['cqt'],observed['stft_1024'],observed['stft_4096'],observed['flux']]
    if (arrays[0].ndim!=2 or arrays[0].shape[1]!=88 or not len(arrays[0])
            or any(a.ndim!=2 or a.shape[1]!=size//2+1 or not len(a) for size,a in zip(WINDOWS,arrays[1:3],strict=True))
            or arrays[3].shape!=(len(arrays[1]),)
            or any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in arrays)):
        raise ValueError('Invalid attack-local spectral envelopes')
    result = np.zeros((len(events),STEPS,CHANNELS),np.float32)
    cqt_clock = np.arange(len(arrays[0]))*CQT_HOP/RATE
    clocks = [np.arange(len(a))*HOP/RATE for a in arrays[1:3]]
    deltas = np.arange(-(STEPS//2),STEPS//2+1)*HOP/RATE
    for index,(start,_,pitch,_) in enumerate(events):
        times = start+deltas
        for channel,offset in enumerate(OFFSETS):
            column = int(pitch)-21+offset
            if 0<=column<88:
                result[index,:,channel] = np.interp(times,cqt_clock,arrays[0][:,column],left=0.,right=0.)
        hz = 440.*2.**((pitch-69.)/12.)
        for resolution,(size,values,clock) in enumerate(zip(WINDOWS,arrays[1:3],clocks,strict=True)):
            for harmonic_index,harmonic in enumerate(HARMONICS):
                bin_position = hz*harmonic*size/RATE
                if bin_position>size//2:
                    continue
                low = int(np.floor(bin_position))
                high = min(size//2,low+1)
                weight = bin_position-low
                envelope = values[:,low]*(1.-weight)+values[:,high]*weight
                result[index,:,9+resolution*4+harmonic_index] = np.interp(times,clock,envelope,left=0.,right=0.)
        result[index,:,17] = np.interp(times,clocks[0],arrays[3],left=0.,right=0.)
    return result
