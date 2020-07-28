import os
import importlib
from pathlib import Path


def isfile(fname):
    return os.path.isfile(fname)


def isdir(dirname):
    return os.path.isdir(dirname)


def makedirs(dirname):
    if not isdir(dirname):
        os.makedirs(dirname)


def join_path(path, *paths):
    return os.path.join(path, *paths)


def get_filelist(dirname):
    return os.listdir(dirname)


def import_module(path):
    return importlib.import_module(path)

# return correct path (the seperator is different between window and linux)
def get_correct_path(path): 
    return Path(path)