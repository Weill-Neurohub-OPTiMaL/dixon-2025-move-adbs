import numpy as np
from scipy import interpolate, signal
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model
from skopt import gp_minimize
from itertools import combinations
from rcssim import rcs_sim as rcs

class RcsClassifier:
    """
    A class used to perform movement state classification via the rcssim package

    Attributes
    ----------
    fft_size : int
        The number of Time-Domain samples used in the short-time FFT
    interval : int
        The number of milliseconds that the FFT window is shifted for each 
        subsequent Power Band / FFT sample 
    pb_info : list
        A list containing four or fewer nested lists, with each describing the
        Power Bands in the form: [TD_channel, [lower_bound_hz, upper_bound_hz]]
    update_rate : int
        The number of PB samples to average over before producing an LD update.
    weights : array
        The LD weights used for predicting continuous acceleration. Should be 
        the same length as `pb_info`
    lag : float
        The prediction latency. A positive value means that future behavior is 
        being predicted from current neural activity.
    threshold : int
        LD threshold for predicting movement state
    onset : positive int
        The number of LD updates (or outputs) that must be above the threshold
        in order to trigger a state change to the super-threshold state.
    termination : positive int
        The number of LD updates (or outputs) that must be below the threshold
        in order to trigger a state change to the sub-threshold state.
    blank_duration : positive int
        The number of PB samples that LD outputs are frozen after state change.
        
    td, td_ts : lists
        The Time-Domain data for all three channels (STN, PostC, Prec) and the
        corresponding timestamps. Each list entry contains data for a separate
        session - (n,3) array for td; (n,) array for td_ts
    pb, pb_ts : lists
        The Power Band data (as defined by `pb_info`) and the corresponding 
        timestamps. Each list entry contains data for a separate session -
        (n,num_bands) array for pb; (n,) array for pb_ts
    accel, accel_ts : lists
        The accelerometry data and the corresponding timestamps. Each list entry
        contains data for a separate session - 
        (n,) array for accel; (n,) array for accel_ts

    Methods
    -------
    get_params(self)
        Returns a dictionary of all current parameter values.
    set_params(self, **kwargs)
        Sets any selection of the parameters.
    get_data(self, *args)
        Returns a dictionary of the current data streams
    set_data(self, td, td_ts, accel, accel_ts, autoblank=True)
        Sets Time-Domain and accelerometry data
        
    compute_pb(self, autoblank=True)
        Computes Power Band data from the current Time-Domain data given the
        specifications of `pb_info` and stores them in the instance
    fit_reg()
        Fits a regression model predicting continuous acceleration and updates
        the `weights` parameter
    predict(self, return_conf_mat=False)
        Predicts movement state using the current parameters and data
    cross_val(self):
        Performs grouped K-fold cross-validation and returns classification 
        performance metrics. Each session is its own fold.
    """
    
    param_list = ['fft_size', 'interval', 'pb_info', 'update_rate', 'weights', 
                  'lag', 'threshold', 'onset', 'termination', 'blank_duration']
    
    data_list = ['td', 'td_ts', 'pb', 'pb_ts', 'accel', 'accel_ts']
    
    
    def __init__(self, **kwargs):
        for param in self.param_list:
            self.__dict__[param] = None
        for datastream in self.data_list:
            self.__dict__[datastream] = None
        self.pb_scaler = StandardScaler()
        self.accel_scaler = StandardScaler()
        if len(kwargs) > 0:
            self.set_params(**kwargs)
        
        
    def set_params(self, **kwargs):
        for param in kwargs.keys():
            if param not in self.param_list:
                raise TypeError("set_params() got an unexpected keyword "
                                + "argument '" + param + "'")
        self.__dict__.update(kwargs)
        
        # Automatically set state blanking unless manually entered
        if 'blank_duration' not in kwargs.keys():
            self.autoset_blanking()
            
        # If pb_info has been reset, clear the data
        if any(param in ['fft_size', 'interval', 'pb_info'] 
               for param in kwargs.keys()):
            if all(attr is not None for attr in [self.td,
                                                 self.fft_size, 
                                                 self.interval, 
                                                 self.pb_info]):
                self.compute_pb(autoblank=False)
                
                
    def autoset_blanking(self):
        if any(param is None for param in [self.fft_size, 
                                           self.interval, 
                                           self.pb_info]):
            self.blank_duration = None
        elif 0 in [pb[0] for pb in self.pb_info]:
            self.blank_duration = np.ceil((self.fft_size/500 + 250/1000)
                                          /(self.interval/1000)).astype(int)
        else:
            self.blank_duration = 0
        
    
    def get_params(self, *args):
        if len(args) == 0:
            params = {key: self.__dict__[key] for key in self.param_list}
        else:
            params = {key: self.__dict__[key] for key in args}
        
        return params
            
            
    def set_data(self, td, td_ts, accel, accel_ts, autoblank=True):
        """
        Sets Time-Domain and accelerometry data.
        
        Parameters
        ----------
        td : (num_sessions,) list of (n, 3) arrays
            Time domain data. Each array in the list is a separate session, 
            each column in the arrays is a different channel
        td_ts : (num_sessions,) list of (n,) arrays
            Timestamps for each time domain sample. Each array in the list is a
            separate session.
        accel : (num_sessions,) list of (n,) arrays
            Watch acceleration data to be predicted. Each array in the list is a
            separate session.
        accel_ts : (num_sessions,) list of (n,) arrays
            Timestamps for each accel sample.
        """
            
        # store time-domain data
        self.td = td
        self.td_ts = td_ts
        # store accelerometry data
        self.accel = accel
        self.accel_ts = accel_ts
        
        # compute power band signals if pb_info is available
        if all(param is not None for param in [self.fft_size, 
                                               self.interval, 
                                               self.pb_info]):
            self.compute_pb(autoblank=autoblank)
        
        
    def get_data(self, *args):
        if len(args) == 0:
            data = {key: self.__dict__[key] for key in self.data_list}
        else:
            data = {key: self.__dict__[key] for key in args}
        
        return data
    
    
    def compute_pb(self, autoblank=True):
        """
        Computes and stores power band signals from the stored Time-Domain data.
        """
        
        # consolidate pb_info so that all pb's on a single channel are together
        consolidate = [[]]*3
        for ch, band_edges in self.pb_info:
            consolidate[ch] = consolidate[ch] + [band_edges]
        pb_info = consolidate
        
        # store each session's data as an instance attribute
        num_sessions = len(self.td)
        self.pb = [0]*num_sessions
        self.pb_ts = [0]*num_sessions
        hann_win = rcs.create_hann_window(self.fft_size, percent=100)
        for s in range(num_sessions):
            # compute pb and store it
            pb = []
            for ch, band_edges in enumerate(pb_info):
                if len(band_edges) < 1:
                    continue
                fft, pb_ts = rcs.td_to_fft(self.td[s][:,ch], self.td_ts[s], 
                                           fs_td=500, 
                                           L=self.fft_size, 
                                           interval=self.interval,
                                           hann_win=hann_win)
                pb_tmp = rcs.fft_to_pb(fft, fs_td=500, L=self.fft_size, 
                                       bit_shift=3, band_edges_hz=band_edges)
                pb.append(pb_tmp)
            self.pb[s] = np.concatenate(pb, axis=1)
            self.pb_ts[s] = pb_ts
        
        # automatically update state blanking unless requested otherwise
        if autoblank:
            self.autoset_blanking()
            
            
    def preprocess(self, refit_scaler=False):
        """
        Preprocesses the Power Band and Accel data in preparation for decoding. 
        This includes applying lag and update rate to the Power Band data and
        aligning the Accel data to the Power Band data. The result is returned
        as a single numpy array with all sessions concatenated.
        
        Parameters
        ----------
        refit_scaler : boolean
            Indicates whether to refit the Z-scoring scalers (True) or to apply
            existing scalers (False)
        
        Returns
        -------
        pb_full : (n, num_pbs) array
            Power band data to regress on.
        accel_full : (n,) array
            Watch acceleration data to be predicted.
        ts_full : (n,) array
            Timestamps for each accel sample.
        session : (n,) array
            An indexing array for the session ID's
        """
        
        num_sessions = len(self.pb)
        pb_full = []
        accel_full = []
        ts_full = []
        session = []
        for s in range(num_sessions):
            # apply lag
            pb_ts = self.pb_ts[s] + self.lag
            # apply update rate
            num_pb_samples, num_pbs = np.shape(self.pb[s])
            clip_samples = int(num_pb_samples/self.update_rate)*self.update_rate
            pb = self.pb[s][:clip_samples, :]
            pb_reshaped = np.reshape(pb, [-1, self.update_rate, num_pbs])
            pb_updated = np.mean(pb_reshaped, axis=1)
            ts_updated = pb_ts[np.arange(self.update_rate-1,
                                         np.size(pb_ts), 
                                         self.update_rate)]
            # align features and targets
            pb, accel, ts = align_data(pb_updated, ts_updated, 
                                       self.accel[s], self.accel_ts[s])            
            # append session
            pb_full.append(pb)
            accel_full.append(accel)
            ts_full.append(ts)
            session.append(s*np.ones(np.size(ts)))
        pb_full = np.concatenate(pb_full, axis=0)
        accel_full = np.concatenate(accel_full).reshape(-1,1)
        ts_full = np.concatenate(ts_full)
        session = np.concatenate(session)
        
        # Z-score features and targets
        if refit_scaler:
            pb_full = self.pb_scaler.fit_transform(pb_full)
            accel_full = self.accel_scaler.fit_transform(accel_full).flatten()
        else:
            pb_full = self.pb_scaler.transform(pb_full)
            accel_full = self.accel_scaler.transform(accel_full).flatten()
        
        return pb_full, accel_full, ts_full, session
        
            
    def fit_reg(self):
        """
        Fits a regression model predicting accelerometry from Power Band data.

        """
        
        # preprocess the data and concatenate all sessions
        pb, accel, ts, _ = self.preprocess(refit_scaler=True)
        
        # fit model
        mdl = linear_model.LinearRegression()
        mdl.fit(pb, accel)
        self.weights_z = mdl.coef_
        self.weights = self.weights_z / np.sqrt(self.pb_scaler.var_)
        self.intercept_z = mdl.intercept_
        
        
    def predict(self, return_conf_mat=False, clock='neural'):
        """
        Predicts movement state from Power Band data and returns some 
        performance metrics.
        
        Returns
        -------
        state_true_full, state_pred_full, state_full_ts, : (n,) arrays
            The true movement state, predicted movement state, and corresponding
            timestamps, respectively. The state is a boolean array, where 0 
            indicates no movement and 1 indicates movement
        output_full, output_full_ts : (n,) arrays
            The continuous-valued LD output and corresponding timestamps. The 
            units are in SD's (Z-units) but the output is not zero-mean
        conf_mat : dict
            The True Positives (key='TP'), False Positives (key='FP'), 
            True Negatives (key='TN'), and False Negatives (key='FN')
        """
        
        # initialize output lists
        state_true_full = [] 
        state_pred_full = [] 
        state_full_ts = [] 
        output_full = []
        output_full_ts = []
        # compute state predictions for each session of data
        for k in range(len(self.pb)):
            output, output_ts, update_tbl = rcs.pb_to_ld(
                                        self.pb[k], 
                                        self.pb_ts[k], 
                                        update_rate=[self.update_rate,[]], 
                                        weights=[self.weights,[]], 
                                        subtract_vec=[np.zeros(4), np.zeros(4)], 
                                        multiply_vec=[np.ones(4), np.ones(4)])
            state_pred, state_ts, output = rcs.ld_to_state(
                                     output,
                                     update_tbl,
                                     self.pb_ts[k],
                                     update_rate=[self.update_rate,[]], 
                                     dual_threshold=[False,[]], 
                                     threshold=[[self.threshold],[]], 
                                     onset_duration=[self.onset,[]], 
                                     termination_duration=[self.termination,[]], 
                                     blank_duration=[self.blank_duration,[]], 
                                     blank_both=[False, False])
            # align accel to time-lagged RC+S state classification
            state_ts += self.lag
            if clock=='neural':
                state_pred, state_true, state_ts = align_data(state_pred, 
                                                              state_ts, 
                                                              self.accel[k], 
                                                              self.accel_ts[k])
                state_true = state_true > -3
                
            elif type(clock)==int:    
                state_true = self.accel[k] > -3
                output[0], _, output_ts[0] = align_data(np.squeeze(output[0]), 
                                                        output_ts[0], 
                                                        state_true, 
                                                        self.accel_ts[k], 
                                                        interp_method='linear', 
                                                        clock=clock)
                state_pred, state_true, state_ts = align_data(
                                                       state_pred, 
                                                       state_ts, 
                                                       state_true, 
                                                       self.accel_ts[k], 
                                                       interp_method='previous', 
                                                       clock=clock)

            # store outputs
            state_true_full.append(state_true.astype(int))
            state_pred_full.append(state_pred.astype(int))
            state_full_ts.append(state_ts)
            output_full.append(output[0])
            output_full_ts.append(output_ts[0])
            
        state_true_full = np.concatenate(state_true_full, axis=0)
        state_pred_full = np.concatenate(state_pred_full, axis=0)
        state_full_ts = np.concatenate(state_full_ts, axis=0)
        output_full = np.concatenate(output_full, axis=0)
        output_full_ts = np.concatenate(output_full_ts, axis=0)
        
        if return_conf_mat:
            conf_mat = compute_conf_mat(state_true_full, state_pred_full)      
            out = (state_true_full, state_pred_full, state_full_ts, output_full,
                   output_full_ts, conf_mat)
        else:
            out = (state_true_full, state_pred_full, state_full_ts, output_full,
                   output_full_ts)
            
        return out
        
        
    def cross_val(self):
        """
        Performs grouped K-fold cross-validation and returns classification 
        performance metrics.
        
        Returns
        -------
        acc : float
            The overall state prediction accuracy
        conf_mat : dict
            The True Positives (key='TP'), False Positives (key='FP'), 
            True Negatives (key='TN'), and False Negatives (key='FN')
        """
        
        # initialize prediction list
        predictions = []
        
        # preprocess the data and concatenate all sessions
        pb_train, accel_train, _, session = self.preprocess(refit_scaler=True)
        
        # perform K-fold cross-validation
        for k in range(len(self.pb)):
            x_train, y_train = pb_train[session!=k,:], accel_train[session!=k]
            x_test, y_test = self.pb[k], self.accel[k]
            x_test_ts, y_test_ts = self.pb_ts[k], self.accel_ts[k]
            mdl = linear_model.LinearRegression()
            mdl.fit(x_train, y_train)
            weights = mdl.coef_ / np.sqrt(self.pb_scaler.var_)
            output, output_ts, update_tbl = rcs.pb_to_ld(
                                        x_test, 
                                        x_test_ts, 
                                        update_rate=[self.update_rate,[]], 
                                        weights=[weights,[]], 
                                        subtract_vec=[np.zeros(4), np.zeros(4)], 
                                        multiply_vec=[np.ones(4), np.ones(4)])
            state_pred, state_ts, output = rcs.ld_to_state(
                                     output,
                                     update_tbl,
                                     x_test_ts, 
                                     update_rate=[self.update_rate,[]], 
                                     dual_threshold=[False,[]], 
                                     threshold=[[self.threshold],[]], 
                                     onset_duration=[self.onset,[]], 
                                     termination_duration=[self.termination,[]], 
                                     blank_duration=[self.blank_duration,[]], 
                                     blank_both=[False, False])
            # align accel to time-lagged RC+S state classification
            state_ts += self.lag
            y_pred, y_test, session_ts = align_data(state_pred, state_ts, 
                                                    y_test, y_test_ts, 
                                                    interp_method='previous', 
                                                    clock=50)
            # convert true continuous acceleration into binary movement state
            y_test = y_test > -3

            # store predictions
            predictions.append(np.column_stack((y_test.astype(int),
                                                y_pred.astype(int))))
            
        predictions = np.concatenate(predictions, axis=0)
        conf_mat = compute_conf_mat(predictions[:,0], predictions[:,1])
        acc = np.mean(predictions[:,0]==predictions[:,1])
        if conf_mat['TP'] == 0:
            F1 = 0
        else:
            precision = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FP'])
            recall = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FN'])
            F1 = 2*precision*recall / (precision + recall)
        
        return acc, F1, conf_mat
    
    
class RcsOptimizer(RcsClassifier):
    """
    Performs hyperparameter optimization and feature selection for the 
    RcsClassifier.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._output_mean = None  
        
        
    def _obj_fxn(self, params):
        self.set_params(update_rate=round(params[0]),
                        lag=params[1],
                        threshold=params[2] + self._output_mean,
                        onset=round(params[3]),
                        termination=round(params[4]))
        _, F1, _ = self.cross_val()
        score = 1 - F1
        
        return score
        
        
    def optimize(self, n_calls):
        """
        Performs GP-BayesOpt hyperparameter optimization for a given feature set
        
        Returns
        -------
        res : OptimizeResult, scipy object
            The optimization result. The key attributes are:
                x : location (parameters) of the minimum
                fun : value (1 - accuracy) at the minimum
        """
        
        res = gp_minimize(self._obj_fxn,
                          [(1, 10),
                           (-0.3, 0.3),
                           (-1.0, 1.0),
                           (1, 10),
                           (1, 10)],      # the bounds on each parameter
                          acq_func="EI",
                          n_calls=n_calls,
                          n_random_starts=5,
                          n_jobs=4)
        return res
    
    
    def feature_select(self, pb_list, try_combos=False, n_iters=20, 
                       constrain_channels=True):
        """
        Performs hyperparameter optimization for candidate feature inputs so 
        that they be compared for selecting a single best combination
        
        Returns
        -------
        best_mdl : list
            The optimization results for each set of Power Band inputs. In the 
            form: [pb_info, output_mean, parameters, 1 - accuracy]
            parameters is in the form: [update_rate,
                                        lag,
                                        threshold,
                                        onset,
                                        termination]
                Where threshold is given in SD's from the output_mean. 
                Therefore the programmed threshold must be the sum of that
                value and the output_mean
        """
        
        if try_combos:
            pb_idx_combos = list(combinations(np.arange(len(pb_list)), 4)) \
                            + list(combinations(np.arange(len(pb_list)), 3)) \
                            + list(combinations(np.arange(len(pb_list)), 2)) \
                            + [[i] for i in range(len(pb_list))]
            pb_list_new = []
            for pb_idx in pb_idx_combos:
                pb_list_new = pb_list_new + [[pb_list[i] for i in pb_idx]]
            pb_list = pb_list_new
        
        best_mdl = []
        for pb_info in pb_list:
            # can use each channel a maximum of two times
            if constrain_channels:
                channel_usage = [pb_info[i][0] for i in range(len(pb_info))]
                if any(channel_usage.count(channel)>2 for channel in range(3)):
                    continue
            # set initial params pb_info and run hyperparameter optimization
            self.set_params(pb_info=pb_info, fft_size=256, interval=50, 
                            lag=0, update_rate=1)
            self.fit_reg()
            self._output_mean = np.dot(self.weights, self.pb_scaler.mean_)
            # run optimization
            res = self.optimize(n_calls=n_iters)
            # compute final model weights for optimized hyperparameters
            self.set_params(update_rate=round(res.x[0]),
                            lag=res.x[1],
                            threshold=res.x[2] + self._output_mean,
                            onset=round(res.x[3]),
                            termination=round(res.x[4]))
            self.fit_reg()
            # log best model
            best_mdl = best_mdl + [[pb_info,
                                    self.weights,
                                    self._output_mean, 
                                    res.x, 
                                    1-res.fun]]
            
        # organize the outputs into a dataframe
        output_df = pd.DataFrame(best_mdl, columns=['pb', 
                                                    'weights', 
                                                    'output_mean', 
                                                    'params', 
                                                    'f1'])
        params = np.array(output_df.params.tolist())
        output_df['update_rate'] = params[:,0]
        output_df['lag'] = params[:,1]
        output_df['threshold_demeaned'] = params[:,2]
        output_df['onset'] = params[:,3]
        output_df['termination'] = params[:,4]
        output_df = output_df[['pb', 'weights', 'update_rate', 'lag', 
                               'output_mean', 'threshold_demeaned', 
                               'onset', 'termination', 'f1']]
            
        return output_df
            


def compute_accel_power(accel, accel_ts, freq_band, L, interval, demean=True, 
                        data_transform=None):
    """
    Computes movement power from 3D accelerometry.

    Parameters
    ----------
    accel : (num_accel_samples,3) array
        3D accelerometry data
    accel_ts : (num_accel_samples,) array
        Timestamps for the accelerometry samples
    freq_band : (2,) arraylike
        Upper and lower limits of the frequency range
    L : int
        FFT size, in number of samples (watch has 50Hz sampling).
    interval : int 
        The interval, in ms, that the FFT window is shifted for producing each 
        subsequent output sample.
    demean : bool, default=True
        Indicates whether to demean the signal before computing power
    data_transform : function
        A function for transforming the accelerometry power. e.g., np.sqrt
        The function must operate elementwise on the output array
        
    Returns
    -------
    accel_power : (num_power_samples,) array
        Power in the selected frequency range
    power_ts : (num_power_samples,) array
        Timestamps for the power samples
    """
    
    # resample signal evenly at 50Hz
    accel_even_ts = np.arange(accel_ts[0], accel_ts[-1], 0.02)
    accel_even = np.zeros([np.size(accel_even_ts), 3])
    for dim in range(3):
        f = interpolate.interp1d(accel_ts, accel[:,dim], kind='linear')
        accel_even[:,dim] = f(accel_even_ts)
    
    # compute euclidean acceleration
    accel_normed = np.linalg.norm(accel_even, axis=1)
        
    # compute spectrogram and sum over desired frequency range
    if demean:
        accel_normed -= np.nanmean(accel_normed)
    noverlap = int(L - (interval/1000) * 50)
    freq, _, Sxx = signal.spectrogram(accel_normed, fs=50, nperseg=L, 
                                      noverlap=noverlap)
    Sxx = Sxx.T
    power_ts = accel_even_ts[np.arange(L-1, np.size(accel_even_ts), L-noverlap)]
    bin_mask = (freq >= freq_band[0]) & (freq <= freq_band[1])
    accel_power = np.sum(Sxx[:,bin_mask], axis=1)
    if data_transform:
        accel_power = data_transform(accel_power)
    
    return accel_power, power_ts


def align_data(neural, neural_ts, accel, accel_ts, 
               max_gap=1, interp_method='linear', clock='neural'):
    """
    Aligns neural and accelerometry data on a unified clock

    Parameters
    ----------
    neural : (num_neural_samples, d) array
        Neural data, which will be the master clock
    neural_ts : (num_accel_samples,) array
        Timestamps for the neural data samples
    accel : (num_accel_samples,) array
        Accelerometry data
    accel_ts : (num_accel_samples,) array
        Timestamps for the accelerometry samples
    max_gap : float, default=1
        The maximum allowed gap for interpolating accelerometry samples, in sec
    interp_method : string, default='linear'
        Interpolation method, not currently a modifiable feature
    clock : string or int, default='neural'
        The central clock to align to. May be `neural` or `accel` for aligning 
        to the recorded data samples, or an int representing a constant sampling
        rate (in Hz)
        
    Returns
    -------
    neural : (num_unified_samples, d) array
        Neural data samples on the unified clock
    accel : (num_unified_samples,) array
        Accelerometry data, resampled on the unified clock
    ts : (num_unified_samples,) array
        Timestamps for the unified clock
    """
    
    # resample data according to the new clock and interpolation method
    if clock=='neural':
        original_ts = accel_ts
        ts, neural = _find_inrange_samples(accel_ts, neural_ts, 
                                           interp_data=neural)
        f = interpolate.interp1d(accel_ts, accel, kind=interp_method)
        accel = f(ts)
    elif clock=='accel':
        original_ts = neural_ts
        ts, accel = _find_inrange_samples(neural_ts, accel_ts, 
                                          interp_data=accel)
        f = interpolate.interp1d(accel_ts, accel, kind=interp_method)
        neural = f(ts)
    elif type(clock)==int:
        original_ts = [neural_ts, accel_ts]
        ts_start = max([neural_ts[0], accel_ts[0]])
        ts_stop = min([neural_ts[-1], accel_ts[-1]])
        ts = np.arange(ts_start, ts_stop, 1/clock)
        f_neural = interpolate.interp1d(neural_ts, neural, kind=interp_method)
        neural = f_neural(ts)
        f_accel = interpolate.interp1d(accel_ts, accel, kind=interp_method)
        accel = f_accel(ts)
    else:
        raise ValueError("`clock` argument must be 'neural', 'accel', " 
                         + "or an int specifying a constant sampling rate")
    
    # remove large interpolation gaps
    neural, accel, ts = _find_tolerable_gaps(neural, accel, original_ts, ts, 
                                             max_gap)
    
    return neural, accel, ts


def compute_conf_mat(truth, predicted):
    """
    Returns a confusion matrix given true and predicted sample labels. The two 
    arrays must be the same shape.
    """
    P = predicted.astype(bool)
    N = ~(predicted.astype(bool))
    conf_mat = {'TP': np.sum(predicted[P]==truth[P]),
                'FP': np.sum(~(predicted[P]==truth[P])),
                'TN': np.sum(predicted[N]==truth[N]),
                'FN': np.sum(~(predicted[N]==truth[N]))} 
    
    return conf_mat


def compute_holdout_performance(TD_df, accel_df, mdl_df, amp_gains,
                                event_ts, arm_segment_names, clock=50):
    """
    Computes the F1 score and accuracy on specific segments of held-out data
    for a collection of models.
    """
    # Initialize data columns
    f1_test = np.zeros(mdl_df.shape[0])
    accuracy_test = np.zeros(mdl_df.shape[0])
    weights = [[] for i in range(mdl_df.shape[0])]
    
    # Set training (first 5 days) and testing data (6th day)
    td = []
    for i in range(6):
        td_tmp = TD_df.to_numpy()[TD_df['session_id']==i,-3:].astype('float64')
        for ch in range(3):
            td_tmp[:,ch] = rcs.transform_mv_to_rcs(td_tmp[:,ch], amp_gains[ch])
        td.append(td_tmp)

    td_ts = [TD_df.timestamp.values[TD_df['session_id']==i]
             for i in range(6)]
    accel = [np.log10(accel_df.accel.values[accel_df['session_id']==i]) 
             for i in range(6)]
    accel_ts = [accel_df.timestamp.values[accel_df['session_id']==i] 
                for i in range(6)]
    
    # Evaluate 
    for i in range(mdl_df.shape[0]):
        # Fit the weights using the training data and optimized parameters
        mdl = RcsOptimizer()
        mdl.set_data(td[:5], td_ts[:5], accel[:5], accel_ts[:5])
        mdl.set_params(pb_info=mdl_df.pb[i], 
                         fft_size=256, interval=50,
                         update_rate=mdl_df.update_rate[i],
                         lag=mdl_df.lag[i], 
                         threshold=mdl_df.output_mean[i] \
                                   + mdl_df.threshold_demeaned[i],
                         onset=mdl_df.onset[i],
                         termination=mdl_df.termination[i])
        mdl.fit_reg()

        # Final evaluation using held-out 6th day
        mdl.set_data([td[5]], [td_ts[5]], [accel[5]], [accel_ts[5]])
        state_true, state_pred, state_ts, output, output_ts = mdl.predict(
                                                                    clock=clock)
        mask = find_segments(state_ts, event_ts, arm_segment_names)
        conf_mat = compute_conf_mat(state_true[mask], state_pred[mask])

        acc = (conf_mat['TP'] + conf_mat['TN']) / \
             (conf_mat['TP'] + conf_mat['TN'] + conf_mat['FP'] + conf_mat['FN'])
        precision = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FP'])
        recall = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FN'])
        F1 = 2*precision*recall / (precision + recall)
        f1_test[i] = F1
        accuracy_test[i] = acc
        weights[i] = mdl.weights
        
    # Log in output
    mdl_df['f1_test'] = f1_test
    mdl_df['accuracy_test'] = accuracy_test
    mdl_df['weights'] = weights
        
    return mdl_df


def _find_inrange_samples(original_ts, interp_ts, interp_data=None):
    """
    Modifies data and timestamp arrays so that they won't receive out of range
    errors during interpolation.
    """
    inrange_idx = (interp_ts>=original_ts[0]) & (interp_ts<=original_ts[-1])
    interp_ts = interp_ts[inrange_idx]
    
    if interp_data is not None:
        if np.ndim(interp_data) == 1:
            interp_data = interp_data[inrange_idx]
        else:
            interp_data = interp_data[inrange_idx,:]
        return interp_ts, interp_data
    else:
        return interp_ts
    
    
def _find_tolerable_gaps(neural, accel, original_ts, new_ts, max_gap):
    """
    Filters the data to include only samples that have been interpolated with 
    nearby sample points.
    """
    if len(original_ts)==2:
        keep_mask = [0,0]
        for i, ts in enumerate(original_ts):
            upper_bound_idx = np.searchsorted(ts, new_ts)
            gaps = ts[upper_bound_idx] - ts[upper_bound_idx-1]
            keep_mask[i] = gaps <= max_gap
        keep_mask = keep_mask[0] & keep_mask[1]
    else:
        upper_bound_idx = np.searchsorted(original_ts, new_ts)
        gaps = original_ts[upper_bound_idx] - original_ts[upper_bound_idx-1]
        keep_mask = gaps <= max_gap
    
    if np.ndim(neural) == 1:
        neural = neural[keep_mask]
    else:
        neural = neural[keep_mask,:]
    accel = accel[keep_mask]
    new_ts = new_ts[keep_mask]
    
    return neural, accel, new_ts


def compute_online_performance(accel_df, rcs_df, output_time_series=False,
                               clock=50, behavior_segments=None, 
                               config_block=4, lag=0):
    
    # select the behavioral segments of interest
    if behavior_segments:
        rcs_df = rcs_df.loc[rcs_df['segment_id'].isin(behavior_segments)]
    if config_block!=4:
        rcs_df = rcs_df.loc[rcs_df['block_id']==config_block]
    
    # align accel to RC+S state classification
    accel_ts = accel_df.timestamp.values - lag
    accel = np.log10(accel_df.accel.values)
    state_ts = rcs_df.timestamp.values
    state_pred = rcs_df.state.values
    
    if clock=='neural':
        state_pred, state_true, state_ts = align_data(state_pred, state_ts, 
                                                      accel, accel_ts)
        state_true = state_true > -3

    elif type(clock)==int:    
        state_true = accel > -3
        state_pred, state_true, state_ts = align_data(state_pred, state_ts, 
                                               state_true, accel_ts, 
                                               interp_method='previous', 
                                               clock=clock)

    # evaluate and store performance metrics
    conf_mat = compute_conf_mat(state_true, state_pred)
    acc = (conf_mat['TP'] + conf_mat['TN']) / \
         (conf_mat['TP'] + conf_mat['TN'] + conf_mat['FP'] + conf_mat['FN'])
    precision = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FP'])
    recall = conf_mat['TP'] / (conf_mat['TP'] + conf_mat['FN'])
    f1 = 2*precision*recall / (precision + recall)
    
    if output_time_series:
        return acc, f1, conf_mat, state_pred, state_true, state_ts
    else:
        return acc, f1, conf_mat
    
    
def compute_stim_distribution(accel_df, rcs_df, stim_lvls=None,
                              behavior_segments=None, return_ppn=False, lag=0):
    """Returns a list of np arrays containing the number of samples at each stim
    level within each movement state. 
    Indexed as: stim_distribution[block][stim_lvl_idx, move_state]
    The bottom row contains the total number of samples in each move_state"""
    
    # select the behavioral segments of interest
    if behavior_segments:
        rcs_df = rcs_df.loc[rcs_df['segment_id'].isin(behavior_segments)]
    num_lvls = len(stim_lvls)
    
    # align accel to RC+S data
    accel_ts = accel_df.timestamp.values - lag
    accel = np.log10(accel_df.accel.values)
    stim_ts = rcs_df.timestamp.values
    stim = rcs_df.stim.values
    block_id = rcs_df.block_id.values
    
    stim, state_true, ts = align_data(stim, stim_ts, accel, accel_ts)
    state_true = state_true > -3

    # compute proportion of time spent at each stimulation amplitude within
    # each block and state
    stim_distribution = []
    for b in range(3):
        stim_distribution.append(np.zeros([num_lvls+1,2]))
        block_stim = stim[block_id==(b+1)]
        block_state_true = state_true[block_id==(b+1)]
        for s, stim_lvl in enumerate(stim_lvls):
            stim_distribution[b][s,0] = np.sum((block_stim==stim_lvl) 
                                               & (block_state_true==0))
            stim_distribution[b][s,1] = np.sum((block_stim==stim_lvl) 
                                               & (block_state_true==1))
        stim_distribution[b][-1,0] = np.sum(block_state_true==0)
        stim_distribution[b][-1,1] = np.sum(block_state_true==1)
        if return_ppn:
            stim_distribution[b][:,0] = stim_distribution[b][:,0] \
                                        / stim_distribution[b][-1,0]
            stim_distribution[b][:,1] = stim_distribution[b][:,1] \
                                        / stim_distribution[b][-1,1]
    
    return stim_distribution