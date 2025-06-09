all_cols = ['vals', 'evals', 'pop_size', 'evo_mode', 'model', 'dim_red', 'instance','function', 'dim', 'elapsed_time', 'coco_directory', 'timestamp', 'true_eval_budget','train_num', 'sort_train','scale_train', 'cma_sees_appoximations', 'note'] # and 'ranks', '
determining_cols = ['pop_size','evo_mode', 'model', 'dim_red','true_eval_budget','train_num', 'note'] # the 'run settings' that we measure the performance of  
pure_desc_cols = ['pop_size','evo_mode','true_eval_budget', 'note'] # the 'run settings' that we measure the performance of  

def get_full_desc(item):
    # nonempty_append = lambda name: ('_'+str(item[name]) if len(str(item[name]))>0 else '')
    if str(item['evo_mode']) == 'Pure':
        desc = '_'.join([s for name in pure_desc_cols if len((s:=str(item[name])))>0])
    else:
        desc = '_'.join([s for name in determining_cols if len((s:=str(item[name])))>0])
    return desc