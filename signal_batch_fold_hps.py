import numpy as np
import os
import kepler_utils
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from subprocess import call

hsps = pd.read_excel("/media/rd3/cchen/cchen/hsp_gpu_fold/hsps_100_200_days.xlsx")
kepler_id = np.array(hsps['kic'].tolist())
tce_period = np.array(hsps['period (d)'].tolist())
tce_duration = np.array(hsps['transit_dur (h)'].tolist())
# hsp_depth = hsps['transit_depth (ppm)'].tolist()
# hsp_smaxis = hsps['sm_axis (au)'].tolist()
tce_t0 = np.array(hsps['t0 (bjd)'].tolist())
# hsp_incl = hsps['incl (deg)'].tolist()
# hsp_eccen = hsps['eccen'].tolist()
KOI_id = np.array(hsps['koi'].tolist())
cc = tce_duration/(tce_period/4096*24)
print(cc.min())




#read the catalog infomation.
# catalog = np.loadtxt('hps_catalog.txt',dtype='str',skiprows=1)

# #100 days ~ 500 days bin_time = 0.5, folding step: 1hr.
# #kepid	kepname	period_days	duration	depth	srad	smass	rprs

# kepler_id=catalog[:,0]
# kepler_name = catalog[:,1]
# tce_period=catalog[:, 2].astype(np.float64)
# tce_duration = catalog[:,3].astype(np.float64)
# print(kepler_id)
# print(tce_period)
# print(tce_period)# *24.0)
# print(tce_duration/(tce_period/4096*24))
# cc = tce_duration/(tce_period/4096*24)
# print(cc.min())
# print(tce_period/(tce_duration/24.0/3.0))

# period_search_array = np.linspace(100,500,38400)
# period_search_array = np.linspace(60,500,44000)
# period_search_array = np.linspace(40, 480, 44000)
period_search_array = np.linspace(100, 200, 44000)
np.savetxt('search_array_hps.txt',period_search_array.T)

# period_search_array = np.linspace(60,400,38400)

'''
plt.plot(tce_period, tce_duration/(tce_period*24.0/256),color='red', marker='.', linestyle='None')
plt.xlabel('Period (Days)')
plt.ylabel('Duration (Pixels)')
plt.show()
'''
def mean_fold(time_array, flux_array, period_id, t0, bin_num, bin_size):


    time_array = kepler_utils.phase_fold_time(time_array, period_id, t0)
    sorted_i = np.argsort(time_array)
    flux = flux_array[sorted_i]
    time = time_array[sorted_i]
    flux_out = np.zeros(bin_num)
    time_out = np.zeros(bin_num)

    for kk in np.arange(bin_num):
        idx_l = int(kk*bin_size)
        idx_u = int((kk+1)*bin_size)
        flux_out[kk] = np.mean(flux[idx_l:idx_u])
        time_out[kk] = np.mean(time[idx_l:idx_u])

    return time_out, flux_out




for id in np.arange(kepler_id.size):
    print(id)
#     id = 4
    kid = int(kepler_id[id])
    print(kid)
    
    #file_name = '/media/etdisk2/Yinan_Zhao/gpu_code/kepler_detrend_new/'+str(kid)+'.npz'
#     file_name = '/media/rd3/surveys/kepler/final_normalized_lcs/KIC'+str(kepler_id[id])+'_withfit_long_v1.npz'
#     file_name = '/media/rd3/cchen/cchen/kepler_ml_v3/norm_lcs/' + str(kepler_id[id]) + '_transitsmasked.npz'

#     file_name = '/media/rd3/cchen/cchen/kepler_ml_v3/norm_lcs/legit/' + str(kepler_id[id]) + '_nofilt.npz'

#9002278_1_filt.npz

    
    file_name = '/media/rd3/cchen/cchen/kepler_ml_v3/norm_lcs/v2/hsps_v2/' + str(kepler_id[id]) + '_filt.npz'
#     file_name = '/media/rd3/cchen/cchen/kepler_ml_v3/norm_lcs/v2/hsps_v2/' + str(kepler_id[id]) + '_rmalltransits.npz'
    
#     file_name = '/media/rd3/surveys/kepler/normalized_lcs/KIC'+str(kepler_id[id])+'_long_v1.npz'

    data_frame = np.load(file_name)
    time = data_frame['time']
    flux = data_frame['flux']
#     spline = data_frame['fit']
    detrend = flux#/spline
    period = tce_period[id]
#     period = 40.79

#     print(period/0.5*24.0, tce_duration[id])

    #plt.plot(time,detrend,marker='.', linestyle='None',color='black')
    #plt.show()
    #exit()


    no_data = flux.size
    no_bin = 8192#4096
    no_sum = np.floor(no_data/no_bin)

    resize = int(no_bin* no_sum)
    time_array = np.ones(resize)
    detrend_array = np.ones(resize)

    time_array = time[:resize]
    detrend_array = detrend[:resize]


    index_min = np.argmin(np.abs(period_search_array-period))

    '''
    print(period*24.0/0.5, period_search_array[index_min]*24.0/0.5)
    pix_out1, out1 = mean_fold(time_array, detrend_array, period, 0.0, no_bin, no_sum)
    pix_out2, out2 = mean_fold(time_array, detrend_array, period_search_array[index_min], 0.0,no_bin, no_sum)
    plt.plot(out1,marker='.', linestyle='None',color='red')
    plt.plot(out2,marker='.', linestyle='None',color='black')
    plt.show()
    '''
    sigma = np.std(detrend_array)/no_sum**0.5


    #exit()
    np.savetxt('kepler_buffer_hps.txt', np.array([time_array, detrend_array]).T)

    gpu_cmd = "CUDA_VISIBLE_DEVICES=0 ./test_hps "+str(resize)+" "+str(kepler_id[id])
    print(gpu_cmd)
    os.system(gpu_cmd)
    saved_file = '/media/rd3/cchen/cchen/hsp_gpu_fold/hsp_100to200/data_'+str(kepler_id[id])+'_hps.bin'
    p_num = 44000 #38400
#     p_num = 32640
    data_cube= np.fromfile(saved_file, dtype='double').reshape(p_num,no_bin)
    #exit()
    print('done')
    #out = mean_fold(time_array, detrend_array, period, 0.0)
    #out = mean_fold(time_array, detrend_array, period_search_array[index_min], 0.0)
    plt.figure()
    plt.plot(data_cube[index_min,:],marker='.', linestyle='None',color = 'black', label = 'GPU')
    #plt.plot(out, color = 'red', label = 'CPU')
    plt.legend(loc='best', fontsize = 9)
    plt.title('kepler ID: '+str(kid)+' period:'+str(period))
    plt.savefig('/media/rd3/cchen/cchen/hsp_gpu_fold/hsp_100to200/foldfigs/' + str(kid)+'_fold.png')
    plt.show()
    plt.close()

#     exit()
