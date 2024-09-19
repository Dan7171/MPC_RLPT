import yaml

# Default settings dictionary
default_settings = {
    'cost_sniffer':
        {
        'include': False, # include in run or not. If False, all other settings irrelevant. If True - sniffer will be used (collecting information about costs during run and optionally show them in guit)
        'gui': False, # TODO: to show costs in gui or not (interactive graphs in browser)
        'save_costs': False, # to save all episode costs (real and mpc) to a pickle file or not
        'buffer_n': 1000  # how many costs to aggregate into each real or mpc buffer, before flushing them to the pickle file (relevant only if saving costs to pickle) 
        },
    'profile_memory':
        {
        'include': False, # True will generate a pickle file that can be parsed later using pytorch memory_viz. Links: https://pytorch.org/docs/stable/torch_cuda_memory.html , https://pytorch.org/memory_viz
        'pickle_path': 'mem_profile.pickle'     
        },
    'gui': 
        {
        'render_ee_icons': True,
        'render_trajectory_lines': True
        }
}

# Load YAML file
def load_config_with_defaults(yaml_path):
    
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file) or {}  # Load YAML and fallback to empty dict if file is empty

    # Recursively update defaults with the values from YAML
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Merge defaults and loaded YAML config
    merged_config = recursive_update(default_settings.copy(), config)
    return merged_config


