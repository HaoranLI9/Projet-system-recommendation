# import pickle5 as pickle
import pickle
from functools import partial
import json


def save(f, *arg):
    with open(f, 'wb') as f:
        for object in arg:
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


tempsave = partial(save, '../runs/temp.pkl')


def load(f, objLen):
    with open(f, 'rb') as f:
        return [pickle.load(f) for _ in range(objLen)]


filename = 'confs/conf_din.json'
with open(filename, 'r') as f:
    conf_din = json.load(f)

filename = 'confs/conf_global.json'
with open(filename, 'r') as f:
    conf_global = json.load(f)

filename = 'confs/conf_imn.json'
with open(filename, 'r') as f:
    conf_imn = json.load(f)

filename = 'confs/conf_base.json'
with open(filename, 'r') as f:
    conf_base = json.load(f)

local_mode = conf_global["local_mode"]
dataset_name = conf_global['dataset_name']

