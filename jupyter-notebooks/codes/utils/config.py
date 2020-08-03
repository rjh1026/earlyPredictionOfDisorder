import yaml
from easydict import EasyDict
from utils.osutils import isfile, join_path


def get_config(filepath=''):
    if not isfile(filepath):
        assert False
    
    with open(filepath) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        cfg = EasyDict(data)
    return cfg
    