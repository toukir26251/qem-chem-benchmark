import yaml

_CFG = None

def load_config(path):
    """Loads a YAML config and make it global"""
    global _CFG
    with open(path) as f:
        _CFG = yaml.safe_load(f)
    return _CFG

def get_config():
    """Returns the global config. Need to call load_config('configs/<molecule>.yaml') first."""
    if _CFG is None:
        raise RuntimeError("No config loaded. Call load_config('configs/<molecule>.yaml') first.")

    return _CFG