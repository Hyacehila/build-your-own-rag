"""Five schemes sharing storage, retrieval, source readers and evaluation."""
from importlib import import_module

MODULES = {"A":"a", "B":"b", "C":"c", "pre-D":"pre_d", "D":"d"}

def scheme(arm):
    return import_module("."+MODULES[arm], __name__)
