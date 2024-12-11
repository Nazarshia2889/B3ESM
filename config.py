import os

import yaml
from yacs.config import CfgNode as CN

from datetime import datetime

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    with open(args.cfg, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    output_dir = yaml_cfg['OUTPUT']
    # os.makedirs(output_dir, exist_ok=True)

    dataset = yaml_cfg['DATA']['DATASET']
    current_time = datetime.now().strftime('%H%M%S')
    current_date = datetime.now().strftime('%Y%m%d')
    subfolder_name = f"{dataset}_{current_time}_{current_date}"
    subfolder_path = os.path.join(output_dir, subfolder_name)
    # os.makedirs(subfolder_path, exist_ok=True)

    yaml_cfg['OUTPUT'] = subfolder_path

    # config_path = os.path.join(subfolder_path, 'config.yaml')

    # # Save the config dictionary to config.yaml
    # with open(config_path, 'w') as file:
    #     yaml.dump(yaml_cfg, file, default_flow_style=False)

    return CN(yaml_cfg)