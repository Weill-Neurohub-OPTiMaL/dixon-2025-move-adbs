import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

KEYPOINT_PAIRS = [[['Body','RLeg','RSmallToe', ['x','y','c']], 
                   ['Body','RLeg','RBigToe', ['x','y','c']]],
                  [['Body','RLeg','RHeel', ['x','y','c']], 
                   ['Body','RLeg','RBigToe', ['x','y','c']]],
                  [['Body','RLeg','RHeel', ['x','y','c']], 
                   ['Body','RLeg','RHeel', ['x','y','c']]],
                  [['Body','RLeg','RHeel', ['x','y','c']], 
                   ['Body','RLeg','RKnee', ['x','y','c']]],
                  
                  [['Body','LLeg','LSmallToe', ['x','y','c']], 
                   ['Body','LLeg','LBigToe', ['x','y','c']]],
                  [['Body','LLeg','LHeel', ['x','y','c']], 
                   ['Body','LLeg','LBigToe', ['x','y','c']]],
                  [['Body','LLeg','LHeel', ['x','y','c']], 
                   ['Body','LLeg','LHeel', ['x','y','c']]],
                  [['Body','LLeg','LHeel', ['x','y','c']], 
                   ['Body','LLeg','LKnee', ['x','y','c']]],
                  
                  [['Body','RLeg','RKnee', ['x','y','c']], 
                   ['Body','Torso','RHip', ['x','y','c']]],
                  [['Body','LLeg','LKnee', ['x','y','c']], 
                   ['Body','Torso','LHip', ['x','y','c']]],
                  [['Body','Torso','RHip', ['x','y','c']], 
                   ['Body','Torso','MidHip', ['x','y','c']]],
                  [['Body','Torso','LHip', ['x','y','c']], 
                   ['Body','Torso','MidHip', ['x','y','c']]],
                  [['Body','Torso','MidHip', ['x','y','c']], 
                   ['Body','Torso','Neck', ['x','y','c']]],
                  [['Body','Torso','Neck', ['x','y','c']], 
                   ['Body','Torso','RShoulder', ['x','y','c']]],
                  [['Body','Torso','Neck', ['x','y','c']], 
                   ['Body','Torso','LShoulder', ['x','y','c']]],
                  
                  [['Body','RArm','RElbow', ['x','y','c']], 
                   ['Body','Torso','RShoulder', ['x','y','c']]],
                  [['Body','RArm','RWrist', ['x','y','c']], 
                   ['Body','RArm','RElbow', ['x','y','c']]],
                  
                  [['Body','LArm','LElbow', ['x','y','c']], 
                   ['Body','Torso','LShoulder', ['x','y','c']]],
                  [['Body','LArm','LWrist', ['x','y','c']], 
                   ['Body','LArm','LElbow', ['x','y','c']]],
                  
                  [['Body','Torso','Neck', ['x','y','c']], 
                   ['Body','Head','Nose', ['x','y','c']]],
                  [['Body','Head','Nose', ['x','y','c']], 
                   ['Body','Head','REye', ['x','y','c']]],
                  [['Body','Head','Nose', ['x','y','c']], 
                   ['Body','Head','LEye', ['x','y','c']]],
                  [['Body','Head','REar', ['x','y','c']], 
                   ['Body','Head','REye', ['x','y','c']]],
                  [['Body','Head','LEar', ['x','y','c']], 
                   ['Body','Head','LEye', ['x','y','c']]],
                  
                  
                  [['Body','RArm','RWrist', ['x','y','c']], 
                   ['R Hand','Palm','Wrist', ['x','y','c']]],
                  [['Body','LArm','LWrist', ['x','y','c']], 
                   ['L Hand','Palm','Wrist', ['x','y','c']]],
                  
                  [['R Hand','Palm','Wrist', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x','y','c']]],
                  [['R Hand','Palm','Wrist', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.1','y.1','c.1']]],
                  [['R Hand','Palm','Wrist', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.2','y.2','c.2']]],
                  [['R Hand','Palm','Wrist', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.3','y.3','c.3']]],
                  [['R Hand','Palm','Wrist', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.4','y.4','c.4']]],
                  
                  [['R Hand','Thumb','Proximal', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x','y','c']]],
                  [['R Hand','Pointer','Proximal', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.1','y.1','c.1']]],
                  [['R Hand','Middle','Proximal', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.2','y.2','c.2']]],
                  [['R Hand','Ring','Proximal', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.3','y.3','c.3']]],
                  [['R Hand','Pinky','Proximal', ['x','y','c']], 
                   ['R Hand','Palm','Base', ['x.4','y.4','c.4']]],
                  
                  [['R Hand','Thumb','Proximal', ['x','y','c']], 
                   ['R Hand','Thumb','Distal', ['x','y','c']]],
                  [['R Hand','Pointer','Proximal', ['x','y','c']], 
                   ['R Hand','Pointer','Distal', ['x','y','c']]],
                  [['R Hand','Middle','Proximal', ['x','y','c']], 
                   ['R Hand','Middle','Distal', ['x','y','c']]],
                  [['R Hand','Ring','Proximal', ['x','y','c']], 
                   ['R Hand','Ring','Distal', ['x','y','c']]],
                  [['R Hand','Pinky','Proximal', ['x','y','c']], 
                   ['R Hand','Pinky','Distal', ['x','y','c']]],
                  
                  [['R Hand','Thumb','Tip', ['x','y','c']], 
                   ['R Hand','Thumb','Distal', ['x','y','c']]],
                  [['R Hand','Pointer','Tip', ['x','y','c']], 
                   ['R Hand','Pointer','Distal', ['x','y','c']]],
                  [['R Hand','Middle','Tip', ['x','y','c']], 
                   ['R Hand','Middle','Distal', ['x','y','c']]],
                  [['R Hand','Ring','Tip', ['x','y','c']], 
                   ['R Hand','Ring','Distal', ['x','y','c']]],
                  [['R Hand','Pinky','Tip', ['x','y','c']], 
                   ['R Hand','Pinky','Distal', ['x','y','c']]],
                  
                  [['L Hand','Palm','Wrist', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x','y','c']]],
                  [['L Hand','Palm','Wrist', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.1','y.1','c.1']]],
                  [['L Hand','Palm','Wrist', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.2','y.2','c.2']]],
                  [['L Hand','Palm','Wrist', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.3','y.3','c.3']]],
                  [['L Hand','Palm','Wrist', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.4','y.4','c.4']]],
                  
                  [['L Hand','Thumb','Proximal', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x','y','c']]],
                  [['L Hand','Pointer','Proximal', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.1','y.1','c.1']]],
                  [['L Hand','Middle','Proximal', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.2','y.2','c.2']]],
                  [['L Hand','Ring','Proximal', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.3','y.3','c.3']]],
                  [['L Hand','Pinky','Proximal', ['x','y','c']], 
                   ['L Hand','Palm','Base', ['x.4','y.4','c.4']]],
                  
                  [['L Hand','Thumb','Proximal', ['x','y','c']], 
                   ['L Hand','Thumb','Distal', ['x','y','c']]],
                  [['L Hand','Pointer','Proximal', ['x','y','c']], 
                   ['L Hand','Pointer','Distal', ['x','y','c']]],
                  [['L Hand','Middle','Proximal', ['x','y','c']], 
                   ['L Hand','Middle','Distal', ['x','y','c']]],
                  [['L Hand','Ring','Proximal', ['x','y','c']], 
                   ['L Hand','Ring','Distal', ['x','y','c']]],
                  [['L Hand','Pinky','Proximal', ['x','y','c']], 
                   ['L Hand','Pinky','Distal', ['x','y','c']]],
                  
                  [['L Hand','Thumb','Tip', ['x','y','c']], 
                   ['L Hand','Thumb','Distal', ['x','y','c']]],
                  [['L Hand','Pointer','Tip', ['x','y','c']], 
                   ['L Hand','Pointer','Distal', ['x','y','c']]],
                  [['L Hand','Middle','Tip', ['x','y','c']], 
                   ['L Hand','Middle','Distal', ['x','y','c']]],
                  [['L Hand','Ring','Tip', ['x','y','c']], 
                   ['L Hand','Ring','Distal', ['x','y','c']]],
                  [['L Hand','Pinky','Tip', ['x','y','c']], 
                   ['L Hand','Pinky','Distal', ['x','y','c']]]
                 ]


def plot_session(ld_data, accel_data, threshold, 
                 accel_viz='continuous', scaler=None):
    
    # scale the LD outputs if requested
    if scaler is None:
        output = [ld_data[i].output for i in [0,1]]
    else:
        output = [scaler.transform(ld_data[i].output) for i in [0,1]]
    
    # convert the accelerometry to binary movement state if requested
    if accel_viz == 'continuous':
        accel = [np.log10(accel_data[i].accel_power) for i in [0,1]]
    elif accel_viz == 'state':
        accel = [np.log10(accel_data[i].accel_power)>-3 for i in [0,1]]
    
    # plot the data
    fig, ax = plt.subplots(3, 2, figsize=(9, 4.5), sharex='all')
    for col in [0,1]:
        ax[0,col].plot(ld_data[col].timestamp, output[col])
        ax[0,col].axhline(y=threshold, color='r', linestyle='-')
        ax_r = ax[0,col].twinx()
        if accel_viz == 'continuous':
            ax_r.plot(accel_data[col].timestamp, accel[col])
        elif accel_viz == 'state':
            t = accel_data[col].timestamp
            idx_range = np.arange(len(move_state))
            start_idx = 0
            while start_idx <len(accel[col]):
                if move_state[start_idx]:
                    end_idx = np.argmax((idx_range>start_idx) & (~accel[col]))
                    ax_r.axvspan(t[start_idx], t[end_idx],
                                 ymax=0.88, color=[0.7,0.7,0.7], zorder=0)
                    start_idx = end_idx
                else:
                    start_idx += 1
        
        ax[1,col].plot(ld_data[col].timestamp, ld_data[col].state)
        
        ax[3,col].plot(ld_data[col].timestamp, ld_data[col].stim)
        
    return fig, ax
        
        
def construct_scaler(mu_y, sigma_y, mu_x, weights, intercept):
    scaler = StandardScaler()
    scaler.mean_ = mu_y/sigma_y - np.dot(weights, mu_x) + intercept
    scaler.scale_ = 1/sigma_y
    
    
def find_participating_idx(performance_df, pb_list):
    
    participating_idx = [np.array([], dtype=int) for i in range(len(pb_list))]
    for pb_idx, pb in enumerate(pb_list):
        for mdl_idx, mdl in enumerate(performance_df.pb.tolist()):
            if pb in mdl:
                participating_idx[pb_idx] = np.append(participating_idx[pb_idx],
                                                      mdl_idx)
    
    return participating_idx


def find_participating_proportion(performance_df, pb_list, num_pbs=4):
    performance_df = performance_df.sort_values(by=['accuracy_test'], 
                                                ascending=False).reset_index()
    part_idx = find_participating_idx(performance_df[:200], pb_list)
    pb_proportions = [np.size(part_idx[i])/200 for i in range(12)]
    top_idx = np.flip(np.argsort(pb_proportions))[:num_pbs]
    top_proportions = [pb_proportions[i] for i in top_idx]
    top_pbs = [pb_list[i] for i in top_idx]
    
    return top_proportions, top_pbs


def plot_participating_performance(performance_df, pb_list, ax,
                                   metric='accuracy_test', num_pbs=4,
                                   use_median=False, clip=False):
    # Setup
    if clip:
        l = 0.5
    else:
        l = 0
    
    # Find the indices of all models containing each PB (poarticpating models)
    participating_idx = find_participating_idx(performance_df, pb_list)
    
    # Construct an array containing the performance metrics for participating 
    # models
    participating_perf = [[] for i in range(len(pb_list))]
    for pb_idx, part_idx in enumerate(participating_idx):
        participating_perf[pb_idx] = np.clip(
                                performance_df[metric].values[part_idx], l, 1)
    if use_median:
        center_perf = np.array([np.median(perf) for perf in participating_perf])
    else:
        center_perf = np.array([np.mean(perf) for perf in participating_perf])
    rank_order = np.flip(np.argsort(center_perf))
    
    participating_perf = [participating_perf[rank_order[i]] 
                          for i in range(num_pbs)]
    center_perf = center_perf[rank_order[:num_pbs]]
    pb_list = [pb_list[rank_order[i]] for i in range(num_pbs)]
    
    # plot
    vp = ax.violinplot(participating_perf, np.arange(num_pbs), widths=0.75,
                       showmeans=(not use_median), showmedians=use_median, 
                       showextrema=False)
    
    # styling
    for body in vp['bodies']:
        body.set_alpha(0.9)
    ax.set(xlim=(-1, num_pbs), xticks=np.arange(num_pbs))
    ax.grid(True, alpha=0.4)
    
    ch_labels = ['STN', 'PostC', 'PreC']
    xticklbls = [[] for i in range(num_pbs)]
    for pb_idx, pb in enumerate(pb_list):
        xticklbls[pb_idx] = ch_labels[pb[0]] + ' \n ' \
                            + str(pb[1][0]) + '-' + str(pb[1][1]) + 'Hz'
    ax.set_xticklabels(xticklbls, 
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Participating model \n performance (' + metric + ')')

    plt.tight_layout()
    
    
def find_reduced_idx(performance_df, pb_list):
    
    participating_idx = find_participating_idx(performance_df, pb_list)
    
    reduced_idx = [np.array([], dtype=int) for i in range(len(pb_list))]
    for pb_idx, pb in enumerate(pb_list): # for each PB
        for part_idx in participating_idx[pb_idx]: # for each participant model
            
            # if the current PB is the only one in the model, flag with -1
            full_pbs = performance_df.pb[part_idx]
            reduced_pbs = list(filter((pb).__ne__, full_pbs))
            if len(reduced_pbs) == 0:
                reduced_idx[pb_idx] = np.append(reduced_idx[pb_idx], -1)
                break
            
            # otherwise, log the index of the reduced model
            for mdl_idx, pb_combo in enumerate(performance_df.pb.tolist()):
                if pb_combo == reduced_pbs:
                    reduced_idx[pb_idx] = np.append(reduced_idx[pb_idx], 
                                                    mdl_idx)
                    break
        
    return participating_idx, reduced_idx


def plot_marginal_performance(performance_df, pb_list, ax, 
                              metric='accuracy_test', num_pbs=4,
                              use_median=False, clip=False, color='tab:blue'):
    
    # Setup
    if clip:
        l = 0.5
    else:
        l = 0
        
    # Find the indices of all models containing each PB (particpating models)
    # and the corresponding models where only that PB has been removed (reduced)
    participating_idx, reduced_idx = find_reduced_idx(performance_df, pb_list)
    
    # Construct an array containing the difference in performance metric between
    # the participating models and their reduced versions
    marginal_perf = [[] for i in range(len(pb_list))]
    for pb_idx in range(len(pb_list)):
        p = participating_idx[pb_idx]
        r = reduced_idx[pb_idx]
        single_pb_mask = r == -1
        p_perf = np.clip(performance_df[metric].values[p], l, 1)
        r_perf = np.clip(performance_df[metric].values[r], l, 1)
        r_perf[single_pb_mask] = 0.5
        marginal_perf[pb_idx] = p_perf - r_perf
    
    if use_median:
        center_perf = np.array([np.median(perf) for perf in marginal_perf])
    else:
        center_perf = np.array([np.mean(perf) for perf in marginal_perf])
    rank_order = np.flip(np.argsort(center_perf))
    
    marginal_perf = [marginal_perf[rank_order[i]] for i in range(num_pbs)]
    center_perf = center_perf[rank_order[:num_pbs]]
    pb_list = [pb_list[rank_order[i]] for i in range(num_pbs)]
    
    # plot
    vp = ax.violinplot(marginal_perf, np.arange(num_pbs), widths=0.75,
                       showmeans=(not use_median), showmedians=use_median, 
                       showextrema=False)
    ax.axhline(0, color='k')
    
    # styling
    for body in vp['bodies']:
        body.set_alpha(0.75)
        body.set_facecolor(color)
    if use_median:
        vp['cmedians'].set_color(color)
    else:
        vp['cmeans'].set_color(color)
    ax.set(xlim=(-1, num_pbs), xticks=np.arange(num_pbs))
    ax.grid(True, alpha=0.4)
    
    ch_labels = ['STN', 'PostC', 'PreC']
    xticklbls = [[] for i in range(num_pbs)]
    for pb_idx, pb in enumerate(pb_list):
        xticklbls[pb_idx] = ch_labels[pb[0]] + ' \n ' \
                            + str(pb[1][0]) + '-' + str(pb[1][1]) + 'Hz'
    ax.set_xticklabels(xticklbls, 
                       rotation=45, ha='right', rotation_mode='anchor')
    ax.set_ylabel('Marginal performance \n contribution (' + metric + ')')

    plt.tight_layout()
    
    return marginal_perf, pb_list
    
    
def animate_session():
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 2))
    fig.patch.set_facecolor('w')

    gs = GridSpec(2, 3, figure=fig)
    ax_Lstim = fig.add_subplot(gs[0, 0])
    ax_Raccel = fig.add_subplot(gs[1, 0], sharex=ax_Lstim)
    ax_Loutput = ax_Raccel.twinx()

    ax_Rstim = fig.add_subplot(gs[0, 2], sharex=ax_Lstim)
    ax_Laccel = fig.add_subplot(gs[1, 2], sharex=ax_Lstim)
    ax_Routput = ax_Laccel.twinx()
    ax_Routput.set_zorder(0)

    ax_pose = fig.add_subplot(gs[:, 1])

    format_anim_axes(fig)
    ax_Loutput.set_xlabel('Grid = 1sec')

    Lstim_line, = ax_Lstim.plot(Lstim[i][0,:], Lstim[i][1,:], color='tab:blue')
    Lstim_dot, = ax_Lstim.plot([], [], 'o', color='tab:blue', zorder=10)
    Raccel_line, = ax_Raccel.plot(Raccel[i][0,:], Raccel[i][1,:], color='tab:blue')
    Raccel_dot, = ax_Raccel.plot(Raccel[i][0,-1], Raccel[i][1,-1], 'o', color='tab:blue', zorder=200)
    Loutput_line, = ax_Loutput.plot(Loutput[i][0,:], Loutput[i][1,:], color='tab:orange')
    Loutput_dot, = ax_Loutput.plot([], [], 'o', color='tab:orange', zorder=10)

    Rstim_line, = ax_Rstim.plot(Rstim[i][0,:], Rstim[i][1,:], color='tab:blue')
    Rstim_dot, = ax_Rstim.plot([], [], 'o', color='tab:blue', zorder=10)
    Laccel_line, = ax_Laccel.plot(Laccel[i][0,:], Laccel[i][1,:], color='tab:blue')
    Laccel_dot, = ax_Laccel.plot([], [], 'o', color='tab:blue', zorder=10)
    Routput_line, = ax_Routput.plot(Routput[i][0,:], Routput[i][1,:], color='tab:orange')
    Routput_dot, = ax_Routput.plot([], [], 'o', color='tab:orange', zorder=10)

    ax_Lstim.set_xlim([t0, t0+5]);
    ax_Lstim.set_xticks(np.arange(t0 + i/10, t0+window+0.1 + i/10, 1))
    # ax_Loutput.tick_params(labelbottom=True)
    # ax_Loutput.set_xticklabels(['']*6)
    ax_Loutput.set_xlabel('Grid = 1sec')

    ax_Lstim.set_ylim([1.4, 2.6])
    ax_Raccel.set_ylim([-10, 0])
    ax_Raccel.set_yticks(np.linspace(ax_Raccel.get_ylim()[0],
                                     ax_Raccel.get_ylim()[1], 4))
    ax_Loutput.set_ylim(np.quantile(rcs_left.output.values, [0.001, 0.999]))
    ax_Loutput.set_yticks(np.linspace(ax_Loutput.get_ylim()[0],
                                      ax_Loutput.get_ylim()[1], 4))

    ax_Rstim.set_ylim([1.4, 2.6])
    ax_Laccel.set_ylim([-10, 0])
    ax_Laccel.set_yticks(np.linspace(ax_Laccel.get_ylim()[0],
                                     ax_Laccel.get_ylim()[1], 4))
    ax_Routput.set_ylim(np.quantile(rcs_right.output.values, [0.001, 0.999]))
    ax_Routput.set_yticks(np.linspace(ax_Routput.get_ylim()[0],
                                      ax_Routput.get_ylim()[1], 4))

    animate(0)

    ax_Loutput.get_xaxis().set_visible(True)
    ax_Loutput.get_xaxis().set_ticks([])
    plt.close()
    
    anim = animation.FuncAnimation(fig, animate, 
                               frames=int(duration/interval/4), 
                               interval=int(interval*1000), blit=False)
    
    
def animate(i):
    Lstim_line.set_data(Lstim[i][0,:], Lstim[i][1,:])
    Lstim_dot.set_data(Lstim[i][0,-1], Lstim[i][1,-1])
    Raccel_line.set_data(Raccel[i][0,:], Raccel[i][1,:])
    Raccel_dot.set_data(Raccel[i][0,-1], Raccel[i][1,-1])
    Loutput_line.set_data(Loutput[i][0,:], Loutput[i][1,:])
    Loutput_dot.set_data(Loutput[i][0,-1], Loutput[i][1,-1])
    
    Rstim_line.set_data(Rstim[i][0,:], Rstim[i][1,:])
    Rstim_dot.set_data(Rstim[i][0,-1], Rstim[i][1,-1])
    Laccel_line.set_data(Laccel[i][0,:], Laccel[i][1,:])
    Laccel_dot.set_data(Laccel[i][0,-1], Laccel[i][1,-1])
    Routput_line.set_data(Routput[i][0,:], Routput[i][1,:])
    Routput_dot.set_data(Routput[i][0,-1], Routput[i][1,-1])
    
    ax_Lstim.set_xlim([t0 + interval*i, t0 + interval*i + window])
    ax_Lstim.set_xticks(np.arange(t0 + interval*i, t0 + interval*i + window, 1))
        
        
def organize_anim_data(rcs_left, rcs_right, watch_left, watch_right, 
                       interval=0.1, window=5, ts0=0, duration=None):
    Lstim = []
    Raccel = []
    Loutput = []
    
    Rstim = []
    Laccel = []
    Routput = []
    
    if duration == None:
        duration = np.min(np.array([rcs_left.timestamp.values[-1],
                                    rcs_right.timestamp.values[-1],
                                    watch_left.timestamp.values[-1],
                                    watch_right.timestamp.values[-1]])) - ts0
    start_ts = np.arange(ts0, ts0+duration, interval)
    end_ts = start_ts + 0.975*window
    
    for i, s in enumerate(start_ts):
        e = end_ts[i]
        
        Lstim.append(_get_data_window(rcs_left.timestamp.values, 
                                      rcs_left.stim.values, 
                                      s, e))
        Raccel.append(_get_data_window(
                          watch_right.timestamp.values, 
                          np.clip(np.log10(watch_right.accel.values),-5.8,-0.2), 
                          s, e))
        Loutput.append(_get_data_window(rcs_left.timestamp.values, 
                                        rcs_left.output.values, 
                                        s, e))
        
        Rstim.append(_get_data_window(rcs_right.timestamp.values, 
                                      rcs_right.stim.values, 
                                      s, e))
        Laccel.append(_get_data_window(
                           watch_left.timestamp.values, 
                           np.clip(np.log10(watch_left.accel.values),-5.8,-0.2), 
                           s, e))
        Routput.append(_get_data_window(rcs_right.timestamp.values, 
                                        rcs_right.output.values, 
                                        s, e))
        
    return Lstim, Raccel, Loutput, Rstim, Laccel, Routput


def _get_data_window(ts, data, s, e):
    
    first_point = np.interp(s, ts, data)
    last_point = np.interp(e, ts, data)
    
    mask = (ts>s) & (ts<e)
    ts = ts[mask]
    data = data[mask]
    
    ts = np.concatenate(([s], ts, [e]))
    data = np.concatenate(([first_point], data, [last_point]))
    
    data_window = np.vstack([ts, data])
    
    return data_window

def format_anim_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.tick_params(labelleft=False, labelright=False)
        ax.xaxis.set_ticklabels([])
        
        
def plot_pose_frame(ax, pose_df, confidence_tol=0.1):
    for keypoint in KEYPOINT_PAIRS:
        if any([keypoint[0][0][0] == 'L', keypoint[0][1][0] == 'L',
                keypoint[1][0][0] == 'L', keypoint[1][1][0] == 'L']):
            color = 'tab:blue'
        elif any([keypoint[0][0][0] == 'R', keypoint[0][1][0] == 'R',
                  keypoint[1][0][0] == 'R', keypoint[1][1][0] == 'R']):
            color = 'tab:red'
        else:
            color = 'tab:purple'
        if (keypoint[0][1][1:]!='Leg') & (keypoint[1][1][1:]!='Leg'):
            plot_body_segment(ax, pose_df, keypoint[0], keypoint[1], 
                              confidence_tol=confidence_tol, color=color)       
    
    
def plot_body_segment(ax, pose_df, keypoint0, keypoint1, 
                      confidence_tol=0.1, color='tab:blue'):
    """
    Enter keypoints as [Pose Set, Body Part, Point, [Three Coordinates]]
    """
    
    p0 = pose_df[keypoint0[0], keypoint0[1], keypoint0[2]][keypoint0[3]].values
    p1 = pose_df[keypoint1[0], keypoint1[1], keypoint1[2]][keypoint1[3]].values
    
    if (p0[2]>confidence_tol) & (p1[2]>confidence_tol) \
        & all(p0[:2]>0) & all(p1[:2]>0):
            ax.plot([p0[0]+500, p1[0]+500], 
                    [2160-p0[1], 2160-p1[1]], 
                    color, linewidth=3, marker='o', markersize=5)