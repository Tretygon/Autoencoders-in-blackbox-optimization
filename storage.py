import numpy as np
import pandas as pd
from evo import Alternate_full_generations,Best_k,Pure
import os
import glob
import numpy.ma as ma


# arr2str = lambda arr: ' '.join(str(x) for x in arr)
def arr2str(arr): 
    import numpy as np
    return np.array2string(arr,max_line_width=np.inf, separator=' ')[1:-1]
str2arr = lambda dtype: lambda string: np.fromstring(string, dtype=dtype, sep=' ')
listmap = lambda func, collection: list(map(func, collection))


def str2evomode(s):
        if s.startswith("Pure"): return Pure()
        elif s.startswith("BestK"):
            s1 = s[len("BestK"):]
            return Best_k(float(s1) if '.' in s1 else int(s1), 1, 1) 
        else: raise Exception
def str2trueRatio(s):
            if s.startswith("Pure"): return 1
            elif s.startswith("BestK"):
                s1 = s[len("BestK"):]
                return float(s1) if '.' in s1 else int(s1)
            else: raise Exception
        

unzip = lambda a:list(map(list,list(zip(*a))))

def store_data(df, name=None,  datastore='data/data.h5'):
    if df.empty: return
    vals = df['vals'].to_list()
    evals = df['evals'].to_list()

    max_len = max(map(len, vals))
    lenghts = np.stack((len(a) for a in vals),axis=0)
    vals = np.stack([np.pad(a, (0,max_len - len(a)), mode='empty') for a in vals],axis=0)
    evals = np.stack([np.pad(a, (0,max_len - len(a)), mode='empty') for a in evals],axis=0)
    
    df = df[['pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'full_desc', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget','train_num', 'sort_train','scale_train']]
    # df = df.drop(columns=['vals','evals','true_ratio'], errors = 'ignore')
    # df['vals'] = df['vals'].parallel_map(arr2str)
    # df['evals'] = df['evals'].parallel_map(arr2str)
    df['evo_mode'] = df['evo_mode'].map(str)

    with pd.HDFStore(datastore,'a') as data_storage:
        if name== None:
            names = listmap(lambda a: a[1:], data_storage.keys())
            names = [int(a) for a in names if a.isnumeric()]
            name = max(names) if len(names)>0 else 0
            name = str(name+1)
        data_storage.put(name,df)
    np.savez_compressed(f'data/{name}_numpy', vals=vals, evals=evals, lenghts=lenghts)


# def remove(names, datastore):
#     with pd.HDFStore(datastore,'a') as data_storage:
#         for name in names:
#             os.remove('data/'+name)
#             del data_storage[name]

    
def store_as_num(data, datastore='data/data.h5'):
    with pd.HDFStore(datastore,'a') as data_storage:
        names = map(lambda a: a[1:], data_storage.keys())
        max_name = max([a for a in names if a.isnumeric()])
    store_data(data,max_name,datastore)

def load_data(target_names=None, datastore='data/data.h5'):
    def transform(df, name):
        loaded = np.load(f'data/{name}_numpy.npz')
        lenghts = list(loaded['lenghts'])
        for c in ['vals', 'evals']:
            unstacked = list(loaded[c])
            df[c] =  [a[:l] for a,l in zip(unstacked, lenghts)]
        df['true_ratio'] = df['evo_mode'].map(str2trueRatio)
        df['evo_mode'] = df['evo_mode'].map(str2evomode)
        return df
    try:
        with pd.HDFStore(datastore,'r') as data_storage:
            if isinstance(target_names, (str, bytes, bytearray)): # target_name is a string
                return transform(data_storage[str(target_names)], str(target_names))
            else: # array of names or None == accept all available
                slices = []
                keys = data_storage.keys()

                if target_names == None:
                    check_if_needed = lambda a: True 
                else: 
                    target_names = set(target_names)
                    check_if_needed = lambda a: a in target_names

                for name in keys:
                    name = name[1:]
                    if check_if_needed(name):
                        df = transform(data_storage[name], name)
                        slices.append(df)
                return pd.concat(slices, ignore_index=True) if len(slices)>0 else None
    except FileNotFoundError:
        return None
    
# def load_old(target_name=None, datastore='data/data.h5'):
#     from pandarallel import pandarallel
#     pandarallel.initialize(progress_bar=True)
#     def transform(df, name):
#         df['vals'] = df['vals'].parallel_map(str2arr(float))
#         df['evals'] = df['evals'].parallel_map(str2arr(int))
#         df['true_ratio'] = df['evo_mode'].map(str2trueRatio)
#         df['evo_mode'] = df['evo_mode'].map(str2evomode)
#         return df
#     try:
        
        
#         with pd.HDFStore(datastore,'r') as data_storage:
#             if target_name != None: 
#                 return transform(data_storage[str(target_name)], str(target_name))
#             else:
#                 slices = []
#                 keys = data_storage.keys()
#                 for name in keys:
#                     name = name[1:]
#                     df = transform(data_storage[name], name)
#                     slices.append(df)
#                 return pd.concat(slices, ignore_index=True) if len(slices)>0 else None
#     except FileNotFoundError:
#         return None         
    

def map_all(f,datastore='data/data.h5'):
    df = load_data(datastore=datastore)
    df = f(df)
    store_data(df, 'df', datastore=datastore)

def merge_and_load(datastore='data/data.h5', data_glob='data/*'):
    df = load_data(datastore=datastore)
    files = glob.glob(data_glob)
    if len(files)>2:
        for f in files:
            os.remove(f)
        store_data(df, 'df', datastore)
    return df

def overwrite(df,datastore='data/data.h5', data_glob='data/*'):
    files = glob.glob(data_glob)
    for f in files:
        os.remove(f)
    store_data(df, 'df', datastore)
    return df
