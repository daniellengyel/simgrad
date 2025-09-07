from copy import deepcopy


# ==== Generate Configs ====
# We treat the config set-up as a tree. 
# Each node combines all the configs which are passed up from the children. 
# If the node is a "list" note, the configs received from the children are put into a list and passed to the parent node
# A "dictionary" node, recombines all the configs. This can only happen when each of the children have different kind of configs paramters. 
# Recombining means that for a list of configs from two nodes, we create for each config pair from both lists another config being the union of the two configs.
# A "leaf" node will store the value and passes on the value in a list to the parent.  

# given the config tree structure, creates a list of configs
def generate_configs(root): 
    if isinstance(root, dict):
        res = [{}]
        for k in root:
            child_res = generate_configs(root[k])
            # recombination step
            new_res = []
            for r in res:
                for c in child_res: 
                    r_copy = deepcopy(r)
                    r_copy[k] = c
                    new_res.append(r_copy)
            res = new_res
        return res
    elif isinstance(root, list):
        if isinstance(root[0], list):
            return root

        res = []
        for v in root:
            res += generate_configs(v)
        return res
    else:
        return [root]
    
# flattens a config so that we can use it in a dataframe. the flattened keys store the path in the original config. 
def flatten_config(conf):
    assert isinstance(conf, dict)

    res = {}
    for k in conf:
        if isinstance(conf[k], dict):
            for c_k, c_v in flatten_config(conf[k]).items():
                res[f"{k}/{c_k}"] = c_v
        else:
            res[k] = conf[k]
    return res

def flatten_configs(configs):
    return [flatten_config(c) for c in configs]

# given the keys. restores the config into a nested dictionary        
def unflatten_config(config):
    res = {}
    
    def add_to_tree(root, k, v):
        if len(k) == 1:
            root[k[0]] = v
            return 
        
        if k[0] not in root:
            root[k[0]] = {}
        add_to_tree(root[k[0]], k[1:], v)
    
    for k, v in config.items():
        add_to_tree(res, k.split("/"), v)
    return res
        
def unflatten_configs(configs):
    return [unflatten_config(c) for c in configs]
    