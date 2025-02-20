import os, webbrowser
import cocoex, cocopp
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import pandas as pd
import math
import datetime
datetime
chunkyfy = lambda arr,window_size: np.array_split(arr, math.ceil(len(arr)/window_size))

def plot_ranks(df,window_size = 5):
    def compute_ranks(df):
        df, common_eval = normalise_val_length(df)
        def to_ranks(arr):
            a = arr.to_list()
            b = np.apply_along_axis(lambda a: a.argsort().argsort()+1,0,a) # rank within each column 
            return list(b)
        df['ranks'] = df.groupby(['function', 'instance', 'dim'])['normalised_len_vals'].transform(to_ranks)
        df = df.drop(['normalised_len_vals'], axis=1)
        return df, common_eval 
    df,common_eval = compute_ranks(df)
    avg_rank_series = df.groupby(['full_desc'])['ranks'].apply(lambda a:np.average(a.to_list(),axis=0))
    avg_rank = avg_rank_series.apply(np.average)

    fig, ax = plt.subplots()
    title_stringer = lambda name, mn, mx: name + ' ' + str(mn) + (('-'+str(mx)) if mn != mx else '')
    title = title_stringer('fun',  df['function'].min(), df['function'].max()) + title_stringer(', dim', df['dim'].min(),df['dim'].max()) + title_stringer(', inst', df['instance'].min(),df['instance'].max())
    
    ax.set(
        xlabel='evals', 
        ylabel='average rank',
        title=title,
        xscale = 'linear',
        yscale='linear'
    )
    ax.invert_yaxis()  # better ranks to be higher instead of lower on the graph

    
    eval_checkpoints = list(map(lambda a: a[-1],chunkyfy(common_eval,window_size)))
    for (desc, ranks) in avg_rank_series.items():
        # ax.plot(common_eval, ranks,label=desc, linestyle='-', marker='|')
        r = list(map(np.average,chunkyfy(ranks,window_size)))
        ax.plot(eval_checkpoints,r,label=desc, linestyle='-', marker='.')

    #sort 
    order = np.argsort(avg_rank)
    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx]+'-->'+str(round(avg_rank[idx],2)) for idx in order])
    ax.grid()
    n = datetime.datetime.now().strftime("%m_%d___%H_%M_%S")
    fig.savefig(f"graphs/{n}.png")
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

# replaces absolute measurents by relative order => model performance can now be compared across different functions and instances
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
        k = int(cur_step/master_step)
        exact = master_step*k == cur_step #bez zaokrouhlovani
        if exact and k == 1:
            res = vals
        elif exact and k <1:
            res = vals[::k] #take each k-th item 
        elif exact and k > 1:
            res = [a for a in vals for _ in range(k)] #duplicate each item k-times
        else:
            res = []
            cur_i = 0
            for master_eval in master_evals:
                while len(evals) < cur_i and master_eval > evals[cur_i]:
                    cur_i+=1
                if master_eval > evals[cur_i]: break #end of arr
                res.append(vals[cur_i])
        return np.array(res)

    transformed = df.apply(vals_to_correct_sampling,axis=1) 
    end_i = min([len(a) for a in transformed])
    transformed = [a[:end_i] for a in transformed] #all evals should end the same
    master_evals = master_evals[:end_i]
    df['normalised_len_vals'] = transformed

    return df, master_evals

if __name__ == '__main__':
    import main
    main.main()