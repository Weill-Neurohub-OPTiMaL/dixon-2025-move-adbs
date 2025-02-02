import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import interpolate, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from rcssim import rcs_sim as rcs
from move_adbs_utils import *

from ipywidgets import *
from IPython.display import clear_output
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt


def compute_specgram(td_df, ch, fs_td, L, interval, hann_win, amp_gain, 
                     bit_shift=3):
    """
    Compute specgrams from time-domain data using RCS-mimetic processing.

    Parameters
    ----------
    td_df : DataFrame
        Pandas dataframe containing time-domain data. Must contain a column 
        named 'timestamp'
    ch : int
        The channel number {0,1,2}
    fs_td : int
        Sampling frequency of the time-domain channel
    
    Returns
    -------

    """
    time_td = td_df.timestamp.values[250:]
    data_td_mv = td_df.values[250:,ch+1]
    data_td_rcs = rcs.transform_mv_to_rcs(data_td_mv, amp_gain[ch])
    data_fft, time_spec = rcs.td_to_fft(data_td_rcs, time_td, fs_td, L,
                                        interval, hann_win)
    spec = rcs.fft_to_pb(data_fft, fs_td, L, bit_shift, band_edges_hz=[])
    center_freqs = np.arange(L/2) * fs_td/L
    
    return spec, center_freqs


def compute_pca(Sxx_left, Sxx_right, log_scale=True, z_score=True):
    """Computes PCA models on spectrogram data"""
    
    # log transform if requested and remove NaN values
    if log_scale:
        Sxx_left = np.log10(Sxx_left[:, ~np.isnan(Sxx_left).any(axis=0)]).T
        Sxx_right = np.log10(Sxx_right[:, ~np.isnan(Sxx_right).any(axis=0)]).T
    else:
        Sxx_left = Sxx_left[:, ~np.isnan(Sxx_left).any(axis=0)].T
        Sxx_right = Sxx_right[:, ~np.isnan(Sxx_right).any(axis=0)].T
    
    # Z-score if requested
    if z_score:
        xform_left = StandardScaler()
        Sxx_left = xform_left.fit_transform(Sxx_left)
        xform_right = StandardScaler()
        Sxx_right = xform_right.fit_transform(Sxx_right)

    # perform pca on the z-scored data
    pca_mdl_left = PCA()
    pca_mdl_left.fit(Sxx_left)
    pca_mdl_right = PCA()
    pca_mdl_right.fit(Sxx_right)
    
    return pca_mdl_left, pca_mdl_right


def convert_pc_to_pb(weights, freq, num_pcs=None, included_idx=[]):
    """Converts principal components of spectrogram data into their closest 
    approximations as power bands. If any frequencies should be left out, remove
    them prior to running this function and specify both the frequencies and the
    original indices of the corresponding weights that have been included so
    that the raw specgram can be appropriately indexed without further 
    modification."""
    
    num_freqs = len(freq)
    if num_pcs == None:
        num_pcs = np.shape(weights)[0]
    if np.size(included_idx) == 0:
        included_idx = np.arange(num_freqs)
    
    power_band_idx = np.zeros([num_pcs,2], dtype='int64')
    power_band_freq = np.zeros([num_pcs,2])
    claimed = np.zeros(num_freqs, dtype='bool')
    for i in range(num_pcs):
        # find the largest available peak weight
        peaks = signal.find_peaks(np.abs(weights[i,:]))[0]
        unclaimed_peaks = peaks[~claimed[peaks]]
        if np.size(unclaimed_peaks) == 0:
            peak_idx = np.argmax(np.abs(weights[i,:]) * ~claimed)
        else:
            peak_idx = unclaimed_peaks[np.argmax(
                                            np.abs(weights[i,unclaimed_peaks]))]
        peak_weight = weights[i,peak_idx]
        dist_from_peak = (peak_weight - weights[i,:]) / peak_weight
        small_weights = dist_from_peak >= 0.5
        
        # find the lower bound
        low_bound = peak_idx
        while low_bound > 0:
            if (dist_from_peak[low_bound]-dist_from_peak[low_bound-1] > 0.02) \
               or (claimed[low_bound-1]) or (small_weights[low_bound-1]) \
               or ((freq[peak_idx]-freq[low_bound-1]) > 25):
                break
            else:
                low_bound-=1
        while freq[low_bound] < 4:
            low_bound+=1
        
        # find the upper bound
        up_bound = peak_idx + (peak_idx < (num_freqs-1))
        while up_bound < (num_freqs-1):
            if (dist_from_peak[up_bound-1]-dist_from_peak[up_bound] > 0.02) \
               or (claimed[up_bound]) or (small_weights[up_bound]) \
               or ((freq[up_bound]-freq[peak_idx]) > 25):
                break
            else:
                up_bound+=1
        
        # add the power band to the output array using the indices of the 
        # original specgram and update the array that indicates which 
        # frequencies have been claimed    
        power_band_idx[i,:] = np.array([included_idx[low_bound], 
                                        included_idx[up_bound]]).T
        claimed[low_bound:up_bound] = True
        # convert the power band indices into their corresponding frequencies
        power_band_freq[i,:] = np.array([freq[low_bound], 
                                         freq[up_bound]]).T
    
    return power_band_idx, power_band_freq


def plot_pca_pb(pca_mdl_left, pca_mdl_right, pb_left, pb_right, freq, 
                num_pcs=5, fig=None, ax=None, abs_val=True, norm_weights='sum'):
    """plots the distribution of component weights across the frequency 
    spectrum from bilateral sepctrogram data. The data is not assumed to arise 
    from any source in particular (may be LFP, acceleration)"""
    
    clr = ['#1f77b4', '#ff7f0e', '#2ca02c', 'tab:red']
    
    # plot the coefficient weights for each component
    if fig==None:
        fig, ax = plt.subplots(1,2, figsize=(10, 3), sharex=True, sharey='row')
    fig.suptitle('Principal component weighting')
    plt.tight_layout()
    for i in range(num_pcs):
        if norm_weights=='sum':
            weight_sum_left = np.sum(np.abs(pca_mdl_left.components_[i,:]))
            weight_sum_right = np.sum(np.abs(pca_mdl_right.components_[i,:]))
        elif norm_weights=='max':
            weight_sum_left = np.max(np.abs(pca_mdl_left.components_[i,:]))
            weight_sum_right = np.max(np.abs(pca_mdl_right.components_[i,:]))
        elif norm_weights==False:
            weight_sum_left = 1
            weight_sum_right = 1
        else:
            raise ValueError('norm_weights parameter not defined correctly.')
            
        if abs_val:
            ax[0].plot(freq, np.abs(pca_mdl_left.components_[i,:]) 
                             / weight_sum_left)
            ax[1].plot(freq, np.abs(pca_mdl_right.components_[i,:]) 
                             / weight_sum_right)
        else:
            ax[0].plot(freq, pca_mdl_left.components_[i,:])
            ax[1].plot(freq, pca_mdl_right.components_[i,:])
            
        # shade background to indicate power bands
        ax[0].axvspan(pb_left[i,0], pb_left[i,1],facecolor=clr[i], alpha=0.5)
        ax[1].axvspan(pb_right[i,0], pb_right[i,1], facecolor=clr[i], alpha=0.5)
        
    # Adjust appearance
    box = ax[0].get_position()
    ax[0].set_position([box.x0+0.02, box.y0+0.02, box.width*0.9, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0-0.03, box.y0+0.02, box.width*0.9, box.height])
    ax[1].legend(['PC' + str(i) for i in range(num_pcs)], 
              loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax[0].grid(True)
    ax[1].grid(True)
    ylabel = 'Model weight' + abs_val * ' (abs value)'
    ax[0].set_ylabel('Model weight (abs value)')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].set_xlabel('Frequency (Hz)')
    if not abs:
        plt.axhline(y=0, color='k', linestyle='-')
    
    return fig, ax

class OptButton:
    
    def __init__(self, output=None, my_folder=None, 
                 left_td=None, left_td_ts=None, 
                 right_td=None, right_td_ts=None,
                 left_accel=None, left_accel_ts=None, 
                 right_accel=None, right_accel_ts=None,
                 canonical_pbs=None, principal_pbs_left=None,
                 principal_pbs_right=None):
        
        self.output = output
        self.my_folder = my_folder
        
        self.left_td = left_td
        self.left_td_ts = left_td_ts
        self.right_td = right_td
        self.right_td_ts = right_td_ts
        self.left_accel = left_accel
        self.left_accel_ts = left_accel_ts
        self.right_accel = right_accel
        self.right_accel_ts = right_accel_ts
        self.canonical_pbs = canonical_pbs
        self.principal_pbs_left = principal_pbs_left
        self.principal_pbs_right = principal_pbs_right
        
        self.left_hem_can_results = None
        self.left_hem_pca_results = None
        self.left_hem_top_results = None
        self.right_hem_can_results = None
        self.right_hem_pca_results = None
        self.right_hem_top_results = None
        
    
    def load_opt_results(self, _):

        with self.output:
            clear_output()
            
            print('Loading prior results...')       

            # Load the results
            self.left_hem_can_results = pd.read_csv(self.my_folder + 
                                        '/opt_results/left_hem_can_results.csv')
            self.right_hem_can_results = pd.read_csv(self.my_folder + 
                                       '/opt_results/right_hem_can_results.csv')

            self.left_hem_pca_results = pd.read_csv(self.my_folder + 
                                        '/opt_results/left_hem_pca_results.csv')
            self.right_hem_pca_results = pd.read_csv(self.my_folder + 
                                       '/opt_results/right_hem_pca_results.csv')

            self.left_hem_top_results = pd.read_csv(self.my_folder + 
                                        '/opt_results/left_hem_top_results.csv')
            self.right_hem_top_results = pd.read_csv(self.my_folder + 
                                       '/opt_results/right_hem_top_results.csv')

            # Fix csv import issues
            self.left_hem_can_results['pb'] = \
                             self.left_hem_can_results['pb'].apply(literal_eval)
            self.right_hem_can_results['pb'] = \
                            self.right_hem_can_results['pb'].apply(literal_eval)
            self.left_hem_pca_results['pb'] = \
                             self.left_hem_pca_results['pb'].apply(literal_eval)
            self.right_hem_pca_results['pb'] = \
                            self.right_hem_pca_results['pb'].apply(literal_eval)
            self.left_hem_top_results['pb'] = \
                             self.left_hem_top_results['pb'].apply(literal_eval)
            self.right_hem_top_results['pb'] = \
                            self.right_hem_top_results['pb'].apply(literal_eval)

            print('Results loaded!')


    def run_optimization(self, _):

        with self.output:
            clear_output()
        
            print('Running optimization routine...')     
            st = time.time()
            
            # Left hem, right hand - SHALLOW OPTIMIZATION
            mdl = RcsOptimizer()
            mdl.set_data(self.left_td, self.left_td_ts, 
                         self.right_accel, self.right_accel_ts)
            
            self.left_hem_can_results = mdl.feature_select(
                                                    pb_list=canonical_pbs, 
                                                    try_combos=True, n_iters=20)
            self.left_hem_can_results = self.left_hem_can_results.sort_values(
                                       by=['f1'], ascending=False).reset_index()
            
            self.left_hem_pca_results = mdl.feature_select(
                                                    pb_list=principal_pbs_left, 
                                                    try_combos=True, n_iters=20)
            self.left_hem_pca_results = self.left_hem_pca_results.sort_values(
                                       by=['f1'], ascending=False).reset_index()
            
            # Left hem, right hand - DEEP OPTIMIZATION
            top_pbs_left = self.left_hem_can_results.iloc[:3,:].pb.tolist()\
                           + self.left_hem_pca_results.iloc[:3,:].pb.tolist()
            
            self.left_hem_top_results = mdl.feature_select(pb_list=top_pbs_left, 
                                                  try_combos=False, n_iters=200)
            
            
            # Right hem, left hand - SHALLOW OPTIMIZATION
            mdl = RcsOptimizer()
            mdl.set_data(self.right_td, self.right_td_ts, 
                         self.left_accel, self.left_accel_ts)
            
            self.right_hem_can_results = mdl.feature_select(
                                                    pb_list=canonical_pbs, 
                                                    try_combos=True, n_iters=20)
            self.right_hem_can_results = self.right_hem_can_results.sort_values(
                                       by=['f1'], ascending=False).reset_index()
            
            self.right_hem_pca_results = mdl.feature_select(
                                                    pb_list=principal_pbs_right, 
                                                    try_combos=True, n_iters=20)
            self.right_hem_pca_results = self.right_hem_pca_results.sort_values(
                                       by=['f1'], ascending=False).reset_index()

            # Right hem, left hand - DEEP OPTIMIZATION
            top_pbs_right = self.right_hem_can_results.iloc[:3,:].pb.tolist()\
                            + self.right_hem_pca_results.iloc[:3,:].pb.tolist()

            self.right_hem_top_results = mdl.feature_select(
                                                  pb_list=top_pbs_right, 
                                                  try_combos=False, n_iters=200)

            et = time.time()
            print('Optimization complete! \n Time elapsed: ',
                  time.strftime('%H:%M:%S', time.gmtime(et-st)))
            
            
def xform_ld_output(ld_output, mdl):
    offset = np.dot(mdl.weights, mdl.pb_scaler.mean_)
    accel_z = (ld_output - offset) + mdl.intercept_z
    accel_pred = mdl.accel_scaler.inverse_transform(accel_z)
    
    return accel_pred


def find_segments(ts, event_ts, segment_names):
    mask = np.zeros(np.shape(ts)).astype(bool)
    for segment in segment_names:
        s = event_ts[segment+'_start'].values
        e = event_ts[segment+'_end'].values
        for block in range(3):
            if ~np.isnan(s[block]):
                mask[(ts>=s[block]) & (ts<=e[block])] = True
                
    return mask


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5 * fs)
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def filter_pose(pose_df, confidence_tol=0, lowpass=10):
    for i in range(138):
        keys = pose_df.iloc[:,(i*3+1):(i*3+4)].keys()
        data = pose_df.iloc[:,(i*3+1):(i*3+4)].values
        data = filter_keypoint(data, 
                               confidence_tol=confidence_tol, lowpass=lowpass)
        pose_df[keys] = data
        
    return pose_df


def filter_keypoint(data, confidence_tol=0, lowpass=10):
    """
    Low-pass filters then spline interpolates position data for a single 
    keypoint.

    Parameters
    ----------
    data : (num_samples, 3) array
        Column 0 is the horizontal pixel distance, column 1 is vertical, and 
        column 2 is the measurement confidence
        
    Returns
    -------
    data : (num_samples, 3) array
        Column 0 is the horizontal pixel distance, column 1 is vertical, and 
        column 2 is the measurement confidence
    """
    
    # Skip empty keypoints
    valid_mask = data[:,2] > confidence_tol
    if np.sum(valid_mask) < 50:
        return data
    
    else:
        # Low-pass filter
        data[valid_mask,0] = butter_lowpass_filter(data[valid_mask,0], 
                                                   lowpass, 30, 6)
        data[valid_mask,1] = butter_lowpass_filter(data[valid_mask,1], 
                                                   lowpass, 30, 6)

        # Spline interpolate over data gaps
        idx = np.arange(np.shape(data)[0])
        tck = interpolate.splrep(idx[valid_mask], data[valid_mask,0])
        data[:,0] = interpolate.splev(idx, tck)
        tck = interpolate.splrep(idx[valid_mask], data[valid_mask,1])
        data[:,1] = interpolate.splev(idx, tck)

        return data