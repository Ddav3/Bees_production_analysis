from .load_datasets import*
from .df_analysis import*
from .plots import*
from .ml import*
import importlib

def reload_all():
    '''
    Used to reload all python util files when applied changes to them.
    If other utils are to be added, add them also to this function for reloading.
    '''

    importlib.reload(load_datasets) 
    importlib.reload(df_analysis) 
    importlib.reload(plots) 
    importlib.reload(ml) 