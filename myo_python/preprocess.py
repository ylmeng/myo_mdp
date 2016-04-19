'''
Created on Jan 15, 2016

@author: ymeng
'''
import numpy as np
from scipy.signal import savgol_filter # need scipy version 0.16
import os
import tf
from math import pi

# These numbers are from experiments
EMG_MAX = 585.0
EMG_MIN = 32.0
GYRO_MAX = 500.0
GYRO_CUTOFF = 200.0

def get_base(dataPath):
    ORI = np.genfromtxt(os.path.join(dataPath,'orientation.mat'), delimiter=',')
    baseRot = np.mean(ORI[0:5, 0])*pi/180
    return baseRot

def process_emg(EMG, EMG_max, EMG_min):
    EMG = (EMG-EMG_min)/(EMG_max-EMG_min+1)
    return EMG

def process_gyro(GYRO, GYRO_max=500, discrete=False):
    if discrete:
        gyro_pos = np.where(GYRO>GYRO_CUTOFF)
        gyro_neg = np.where(GYRO<-GYRO_CUTOFF)
        GYRO = np.zeros(GYRO.shape)
        GYRO[gyro_pos] = 1
        GYRO[gyro_neg] = -1
    else:
        GYRO = GYRO/GYRO_max
    return GYRO   

def normalize_myo(data, update_extremes=False):
    T = np.reshape(data[:,0], (-1, 1))
    EMG = data[:,1:9]
    QUAT = data[:,-4:]
    ACC = data[:,9:12]
    GYRO = data[:,12:15]
#    EUL = data[:,15:18]
    if update_extremes:
        EMG_max = np.max(EMG)
        EMG_min = np.min(EMG)
        GYRO_max = np.max(GYRO)
    else:
        EMG_max = EMG_MAX
        EMG_min = EMG_MIN
        GYRO_max = GYRO_MAX
        
    EMG = process_emg(EMG, EMG_max, EMG_min)
    # GYRO = GYRO/GYRO_max
    
    # Descritize gyroscope data
    GYRO = process_gyro(GYRO, GYRO_max, False)
   
    
    #baseRot = EUL[0, 0]*pi/180 # used for calibration
    #===========================================================================
    # QUAT = np.zeros((EUL.shape[0], 4))
    # for i, euler in enumerate(EUL):
    #     euler = euler*pi/180
    #     QUAT[i] = tf.transformations.quaternion_from_euler(-euler[2], euler[1], -euler[0] + baseRot)
    #===========================================================================

    Data = np.hstack((T,EMG,ACC,GYRO,QUAT))
#    Data = np.hstack((T, EMG, ORI))
    return Data, EMG_max, EMG_min, GYRO_max 

def preprocess(dataPath, update_extremes=False, identifier=''):
    if identifier:
        identifier = '_' + identifier 
    
    EMG = np.genfromtxt(os.path.join(dataPath,'emg'+identifier+'.mat'), delimiter=',')
    ACC = np.genfromtxt(os.path.join(dataPath,'acceleration'+identifier+'.mat'), delimiter=',')
    GYRO = np.genfromtxt(os.path.join(dataPath,'gyro'+identifier+'.mat'), delimiter=',')
    ORI = np.genfromtxt(os.path.join(dataPath,'orientation'+identifier+'.mat'), delimiter=',')
    IMU = np.hstack((ACC, GYRO, ORI))
    #IMU = ORI

    # Down sample, from 50 Hz to 10 Hz
    # remove end points
    EMG = EMG[6:-6:5, :]
    IMU = IMU[6:-6:5, :]
    
    # There are often slightly more imu data points than emg points
    n = EMG.shape[0]
    print "difference between # imu and # emg:", IMU.shape[0]-n
    IMU = IMU[0:n, :]
    
    Data = np.hstack((EMG, IMU))
#    print Data.shape
#    print EMG.shape
#    print IMU.shape
    # data smoothing polyolder=3, windowsize=41
    Data_smooth = savgol_filter(Data, 31, 3, axis=0)
    
    # add time dimension
    Time = np.arange(n).reshape((n,1))
    Data = np.hstack((Time, Data_smooth))
    
    #baseRot = get_base('../../data/work/0/')
    return normalize_myo(Data, update_extremes);
    

def main():
    data = preprocess('../../data/work/0/') 
    np.savetxt("../data/all_data.csv", data, delimiter=",")    

if __name__ == '__main__':
    main()  