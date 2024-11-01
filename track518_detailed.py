import numpy as np
import glob
from nptdms import TdmsFile
import idas_preprocessing as idas
import matplotlib.pyplot as plt
import argparse
from functools import partial
import os
import multiprocessing
from scipy import signal

def das_read(filename):
    data, _, dt, dx = idas.read_tdms(filename)
    return data, dt, dx

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str)
    opt = parser.parse_args()
    return opt
data_list = np.loadtxt('sc_data.txt')
datapath = './data/'
import utm
loc = np.load('xfj_ch.loc.npy')

fiber_loc = np.load('xfj_ch.loc.npy')
fiber_loc = utm.from_latlon(fiber_loc[:,1],fiber_loc[:,2])
fiber_loc = np.array(fiber_loc[0:2])
fiber_loc[0] = fiber_loc[0]-min(fiber_loc[0])
fiber_loc[1] = fiber_loc[1]-min(fiber_loc[1])
X = fiber_loc[0][900-250:1501-250]
Y = fiber_loc[1][900-250:1501-250]
path = '518'
id = 889
tdms_files = glob.glob(datapath+'*UTC_20220'+path+'_*.tdms')
tdms_files.sort()
data = np.array([])
for interval in range(0,6):
    dataa, dt, dx = das_read(tdms_files[id+interval])
    print(id+interval)
    dataa = dataa[300:2200,:]
    if interval == 3:
        data = dataa    
        continue
    data = np.concatenate((data,dataa),axis=1)
data = idas.das_preprocess(data)
data = idas.bandpass(data, dt, fl=1, fh=140)
data_521 = data[:,:]

# path = '428'
# id = 137
# tdms_files = glob.glob(datapath+'*UTC_20220'+path+'_*.tdms')
# tdms_files.sort()
# data = np.array([])
# for interval in range(-2,4):
#     dataa, dt, dx = das_read(tdms_files[id+interval])
#     print(id+interval)
#     dataa = dataa[300:2200,:]
#     if interval == -2:
#         data = dataa    
#         continue
#     data = np.concatenate((data,dataa),axis=1)
# data = idas.das_preprocess(data)
# data = idas.bandpass(data, dt, fl=1, fh=140)
# data_428 = data[:,:]

import numpy as np

def compute_rotation_angle(vector1, vector2):
    # Convert lists to numpy arrays
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calculate the dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    
    # Determine the direction of the rotation using the cross product
    cross_product = np.cross(v1, v2)
    if cross_product < 0:
        # If the cross product is negative, the angle is in the clockwise direction
        angle = 2 * np.pi - angle
    
    # Return the angle in radians
    return angle

def compute_rotation_angles(vectors):
    n = len(vectors) // 2
    angles = []
    
    for i in range(n):
        v1 = vectors[2 * i]
        v2 = vectors[2 * i + 1]
        try:
            angle = compute_rotation_angle(v1, v2)
            angles.append(angle)
        except ValueError as e:
            angles.append(str(e))
    
    return angles
# Example usage
vector1 = [0,1]
vector2 = [1,0]
angle = compute_rotation_angle(vector1, vector2)
angle/(2*np.pi)

import itertools

def simulate(params):
    x, y, z, a = params
    xs = xs_arr[x]
    ys = ys_arr[y]
    theta = theta_arr[z]
    v = vs_arr[a]
    angle = np.ones(600)
    for i in range(600):
        angle[i] = compute_rotation_angle([xs-X[i],ys-Y[i]],[np.cos(theta),np.sin(theta)])
    data_seg2 = np.copy(data_seg)
    data_seg2[np.array(np.where(angle<np.pi/2)),:] = -1 * data_seg[np.array(np.where(angle<np.pi/2)),:]
    data_seg2[np.array(np.where(angle>3*np.pi/2)),:] = -1 * data_seg[np.array(np.where(angle>3*np.pi/2)),:]
    t = h_dist(X,Y,xs,ys) / v * fs         
    data_sub_roll = np.array([np.roll(data_seg2[i,:], -int(t[i]-t.min())) for i in range(len(data_seg2))])
    return beamforming (data_sub_roll)  

sos = signal.butter(6, [1,2], 'bp', fs=1/dt, output='sos')
sg = np.abs(data_521[600:1200,5000:60000]).sum(axis=0)
sg = signal.detrend(sg)
sg = sg - np.mean(sg)
ev = signal.sosfiltfilt(sos, sg)
ft = signal.argrelextrema(ev, np.less)
ft_time2 = ft[0]
print(ft_time2)
n =  ft_time2.shape[0]
source = np.zeros([n,2])
ft_time = np.zeros([n,2])
v = np.zeros([n,1])
fs = 300
for i in range(n):
    if i == 0:
        x = range(10)
        y = range(10)
        z = range(21)
        a = range(1)
        xs_arr = np.linspace(140,180,len(x))
        ys_arr = np.linspace(100,150,len(y))
        theta_arr = np.linspace(0,np.pi,len(z))
        vs_arr = np.linspace(300,310,len(a))
        data3 = data_521[600:1200,5000:5300]
    else:
        x = range(20)
        y = range(20)
        z = range(21)
        a = range(8)
        xs_arr = np.linspace(source[i-1,0]-0.7,source[i-1,0]+1,len(x))
        ys_arr = np.linspace(source[i-1,1]-1.4,source[i-1,1]+0.7,len(y))
        theta_arr = np.linspace(0,np.pi,len(z))
        vs_arr = np.linspace(v[i-1]-40,v[i-1]+40,len(a))
        data3 = data_521[600:1200,5000+int(ft_time2[i-1]):5000+int(ft_time2[i])]
    if data3.std()<40:
        print(str(i)+'std too small')
        plt.xlabel('Time (s)') 
        plt.ylabel('Channel')
        # plt.xticks(np.arange(0,601,300),np.arange(0,2.1,1))
        # plt.yticks(np.arange(0,601,100),np.arange(900,1501,100))
        plt.xlim(0,600)
        plt.title(str(clim))
        clim = data_seg.std()*2
        #plt.title(str(np.round(ft_time[i,0],0))+','+str(np.round(vs_arr[iv_max],1))+','+str(np.round(xs_arr[ixs_max],1))+','+str(np.round(ys_arr[iys_max],1))+','+str(np.round(data_sub_beamforming[ixs_max, iys_max, ia_max,iv_max]/1e30,2)))
        plt.imshow(data_seg, aspect='auto', cmap='seismic', vmin = -clim, vmax=clim,alpha=0.7)
        plt.savefig('tmp2/518dtsimulation'+str(i)+'.png')
        plt.close()
        source[i,:] = source[i-1,:]
        ft_time[i,:] = [int(ft_time2[i-1]),int(ft_time2[i])]
        v[i] = v[i-1]
        continue
    # 创建参数组合的网格
    param_grid = list(itertools.product(x,y,z,a))
    h_dist = lambda x,y,xs,ys: np.sqrt((x-xs)**2+(y-ys)**2)
    beamforming = lambda arr2d: np.sum((arr2d.sum(axis=0))**6)
    fs = 300
    fl = 2
    fh = 70
    data4 = idas.bandpass(data3,1/300,fl,fh)
    data_seg = data4
    with multiprocessing.Pool(64) as pool:
        # 使用 map 方法并行执行 simulate 函数
        results = pool.map(simulate, param_grid)

    # 创建一个三维数组来存储结果
    result_array = np.zeros((len(xs_arr), len(ys_arr), len(theta_arr) ,len(vs_arr)))
    # 将结果存储在三维数组中
    for (x, y, z, a), result in zip(param_grid, results):
        result_array[x, y, z, a] = result
    data_sub_beamforming = result_array
    ixs_max, iys_max, ia_max,iv_max = np.unravel_index(data_sub_beamforming.argmax(), data_sub_beamforming.shape)
    print(iv_max,  vs_arr[iv_max])
    print(ixs_max, xs_arr[ixs_max])
    print(iys_max, ys_arr[iys_max])
    print(ia_max, theta_arr[ia_max])
    print(data_sub_beamforming[ixs_max,iys_max,ia_max]/1e10)
    print('.................')
    source[i,:] = [xs_arr[ixs_max],ys_arr[iys_max]]
    ft_time[i,:] = [int(ft_time2[i-1]),int(ft_time2[i])]
    v[i] = vs_arr[iv_max]
    theta = theta_arr[ia_max]   
    plt.figure(figsize=(20,15))
    plt.rcParams.update({'font.size': 20})
    center = [xs_arr[ixs_max],ys_arr[iys_max]]
    dist = ((X-center[0])**2+(Y-center[1])**2)**0.5
    plt.plot(dist*300/v[i],np.arange(0,len(dist)),'black',linewidth=4)
    plt.xlabel('Time (s)') 
    plt.ylabel('Channel')
    # plt.xticks(np.arange(0,601,300),np.arange(0,2.1,1))
    # plt.yticks(np.arange(0,601,100),np.arange(900,1501,100))
    plt.xlim(0,600)
    plt.title('sumlation and actual')
    clim = data_seg.std()*2
    plt.title(str(np.round(ft_time[i,0],0))+','+str(np.round(vs_arr[iv_max],1))+','+str(np.round(xs_arr[ixs_max],1))+','+str(np.round(ys_arr[iys_max],1))+','+str(np.round(data_sub_beamforming[ixs_max, iys_max, ia_max,iv_max]/1e30,2)))
    plt.imshow(data_seg, aspect='auto', cmap='seismic', vmin = -clim, vmax=clim,alpha=0.7)
    plt.savefig('tmp2/518dtsimulation'+str(i)+'.png')
    plt.close()
    np.savetxt('source_518_detailed.txt',source)
    np.savetxt('ft_time_518_detailed.txt',ft_time)
    np.savetxt('v_518_detailed.txt',v)
    np.savetxt('theta_428_detailed.txt',theta)

