import os, webbrowser
import cocoex, cocopp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import math
import datetime
import seaborn as sns
matplotlib.use('TkAgg')

# print(matplotlib.get_backend())
chunkyfy = lambda arr,window_size: np.array_split(arr, math.ceil(len(arr)/window_size))

# computes relative order instead of absolute measurents => model performance can now be compared across different functions and instances
def compute_ranks(df):
        def to_ranks(arr):
            arr = arr.to_list()
            b = np.apply_along_axis(lambda a: 100-(a.argsort().argsort())*100/(len(a)-1),0,arr) # rank within each column and turn to percentils
            # b = np.apply_along_axis(lambda a: (a-a.min())/(a.max()-a.min()),0,a) # normalised to 0-1
            return list(b)
        
        def normalise_val_length(df:pd.DataFrame):
            only_evals = df['evals']
            all_steps = [e[2] - e[1] for e in only_evals] 
            master_step = min(all_steps)
            max_eval = min([e[-1] for e in only_evals])
            
            master_evals = np.array(range(max(all_steps),max_eval+1,master_step))
            
            def vals_to_correct_sampling(run):
                (evals,vals) = run['evals'], run['vals']
                cur_step = evals[2] - evals[1]
                begin_i = np.nonzero(evals>=master_evals[0])[0][0] #all evals should start the same
                evals,vals = evals[begin_i:],vals[begin_i:]
                assert(cur_step >= master_step)
                k = int(cur_step/master_step) # how many times is cur_step bigger than master_step
                
                if master_step*k == cur_step: # if the ratio is exactly an int 
                    res = [a for a in vals for _ in range(k)] #duplicate each item k-times
                else:
                    res = []
                    cur_i = 0
                    for master_eval in master_evals:
                        while cur_i < len(evals) and master_eval > evals[cur_i]:
                            cur_i+=1
                        if cur_i == len(evals): break #end of arr
                        res.append(vals[cur_i])
                return np.array(res)

            transformed = df.apply(vals_to_correct_sampling,axis=1)
            # ahe = transformed.value_counts().to_numpy()
            # print(ahe)
            lens = [len(a) for a in transformed]
            # u,inx, cs = np.unique(lens,return_index=True,return_counts=True )
            # aaaa = df.loc[inx[0]].to_list()
            end_i = min(lens)
            transformed = [a[:end_i] for a in transformed] #all evals should end the same
            master_evals = master_evals[:end_i]
            df['normalised_len_vals'] = transformed

            return df, master_evals
        
        df, common_eval = normalise_val_length(df)
        df['ranks'] = df.groupby(['function', 'instance', 'dim','train_num', 'sort_train','scale_train'])['normalised_len_vals'].transform(to_ranks)
        df = df.drop(['normalised_len_vals'], axis=1)
        return df, common_eval 
    
def plot(df,window_size = 5):
    df,common_eval = compute_ranks(df)
    plot_ranks(df,common_eval,window_size)

def get_param_desc_title(df):
    title_stringer = lambda beginning, name: beginning + ' ' + str(df[name].min()) + (('-'+str(df[name].max())) if df[name].min() != df[name].max() else '')
    title = title_stringer('fun','function') + title_stringer(', dim','dim') + title_stringer(', inst','instance')
    return title

def plot_ranks(df,common_eval, window_size = 5):
    fig, ax = plt.subplots()
    ax.set(
        xlabel='evals', 
        ylabel='average rank',
        title= 'ranks\n'+get_param_desc_title(df),
        xscale = 'linear',
        yscale='linear'
    )
    # ax.invert_yaxis()  # better ranks to be upper instead of lower on the graph
    ax.grid()

    avg_rank_series = df.groupby(['full_desc'])['ranks'].apply(lambda a:np.average(a.to_list(),axis=0)) # rank achieved by each setting in time, avg across fun&dim
    eval_checkpoints = list(map(lambda a: a[-1],chunkyfy(common_eval,window_size)))
    for (desc, ranks) in avg_rank_series.items():
        # ax.plot(common_eval, ranks,label=desc, linestyle='-', marker='|')
        r = list(map(np.average,chunkyfy(ranks,window_size)))
        ax.plot(eval_checkpoints,r,label=desc, linestyle='-', marker='.')

    #sort legend  
    def sort_legend(ax, values):
        order = np.argsort(values)[::-1]
        handles, labels = plt.gca().get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(values[idx],2)) for idx in order])
    
    avg_rank = avg_rank_series.apply(np.average) # avg across time from avg across fun&dim, 'final score' of a setting
    sort_legend(ax, avg_rank)
    
    n = datetime.datetime.now().strftime("%m_%d___%H_%M_%S")
    fig.savefig(f"graphs/ranks_{n}.png")
    plt.pause(0.01)
    plt.show()
    
# def boxplot():
#   fig, ax = plt.subplots()
#     ax.set(xlabel='log evals', ylabel='func value',
#     title=f'{func_names[problem_num-1]}\n{problem_info}',xscale = 'linear',yscale='linear')
        
    
#     fun_mins = []
#     for config,config_res in zip(configs,results):
    
#         config_desc = config[-1]
#         evals, vals_ = list(map(np.array,zip(*config_res)))
#         # ii = np.argmax(evals[0]>100)
#         # evals = evals[:,ii:]
#         # vals_ = vals_[:,ii:]
#         for i,med in enumerate(vals_):
#             # med = np.median(vals,0)
#             ax.plot(evals[i], med,label=config_desc)
#             fun_mins.append(med[-1])
#             # print([a[-1] for a in med])
#     order = np.argsort(fun_mins)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(fun_mins[idx],2)) for idx in order])
#     ax.grid()
#     graphs = os.listdir('graphs')
#     ord = int(graphs[-1].split('.')[0]) + 1 if any(graphs) else 0
#     fig.savefig(f"graphs/{ord}.png")
#     # plt.show()

def coco_plot(df):
    out_folders = df['coco_directory'].unique()
    ### coco plotting 
    cocopp.genericsettings.isConv = True
    cocopp.main(' '.join(out_folders))
    # cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

