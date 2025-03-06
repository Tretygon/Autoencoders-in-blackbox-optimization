import numpy as np
import pandas as pd
from evo import Alternate_full_generations,Best_k,Pure



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
    vals = df['vals'].to_list()
    evals = df['evals'].to_list()

    max_len = max(map(len, vals))
    lenghts = np.stack([len(a) for a in vals],axis=0)
    vals = np.stack([np.pad(a, (0,max_len - len(a)), mode='empty') for a in vals],axis=0)
    evals = np.stack([np.pad(a, (0,max_len - len(a)), mode='empty') for a in evals],axis=0)
    
    if 'true_ratio' in df: 
        df = df.drop(columns=['true_ratio'])
    df = df.drop(columns=['vals'])
    df = df.drop(columns=['evals'])
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
    
def merge_datastore_numbers(into:str, datastore='data/data.h5'):
    with pd.HDFStore(datastore,'a') as data_storage:
        names = map(lambda a: a[1:], data_storage.keys())
        names = [a for a in names if a.isnumeric()]
        # slices = [load_data(name) for name in names]
        # df = pd.concat(slices, ignore_index=True)
        # data_storage.put(into,df)
        # data_storage.

def load_data(target_name=None, datastore='data/data.h5'):
    def transform(df, name):
        loaded = np.load(f'data/{name}_numpy.npz')
        lenghts = list(loaded[lenghts])
        for c in ['vals', 'evals']:
            unstacked = list(loaded[c])
            df[c] =  [a[:l] for a,l in zip(unstacked, lenghts)]
        df['true_ratio'] = df['evo_mode'].map(str2trueRatio)
        df['evo_mode'] = df['evo_mode'].map(str2evomode)
        return df
    try:
        with pd.HDFStore(datastore,'r') as data_storage:
            if target_name != None: 
                return transform(data_storage[str(target_name)], str(target_name))
            else:
                slices = []
                keys = data_storage.keys()
                for name in keys:
                    name = name[1:]
                    df = transform(data_storage[name], name)
                    slices.append(df)
                return pd.concat(slices, ignore_index=True) if len(slices)>0 else None
    except FileNotFoundError:
        return None
    
def load_old(target_name=None, datastore='data/data.h5'):
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    def transform(df, name):
        df['vals'] = df['vals'].parallel_map(str2arr(float))
        df['evals'] = df['evals'].parallel_map(str2arr(int))
        df['true_ratio'] = df['evo_mode'].map(str2trueRatio)
        df['evo_mode'] = df['evo_mode'].map(str2evomode)
        return df
    try:
        
        
        with pd.HDFStore(datastore,'r') as data_storage:
            if target_name != None: 
                return transform(data_storage[str(target_name)], str(target_name))
            else:
                slices = []
                keys = data_storage.keys()
                for name in keys:
                    name = name[1:]
                    df = transform(data_storage[name], name)
                    slices.append(df)
                return pd.concat(slices, ignore_index=True) if len(slices)>0 else None
    except FileNotFoundError:
        return None         
def map_all_data(f):
    with pd.HDFStore('data/data.h5','a') as data_storage:
        names = listmap(lambda a: a[1:], data_storage.keys())
        for name in names:
            df = load_data()
            df = f(df)
            data_storage.put(name,df)

def purge_numerics():
    with pd.HDFStore('data/data.h5','a') as data_storage:
        names = listmap(lambda a: a[1:], data_storage.keys())
        names = [int(a) for a in names if a.isnumeric()]
        max_name = max(names) if len(names)>0 else 0
        for name in names: 
            if name == max_name: continue
            else: 
                del data_storage[str(name)]
    df = load_data(str(max_name))
    store_data(df, 'df', 'data/dat.h5')