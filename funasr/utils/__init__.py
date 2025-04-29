
from functools import lru_cache
import os, inspect, threading

@lru_cache(maxsize=1)
def use_dlc() -> bool:
    return os.environ.get("ACCELERATE_TORCH_DEVICE", None) == 'dlc'

@lru_cache()
def debug_enabled(level: int) -> bool:
    try:
        return int(os.environ.get("PY_DEBUG", "0")) >= level
    except:
        return False

def debug_print(level: int = 1, *args):
    if not debug_enabled(level) or not isinstance(level, int):
        return
    f = inspect.currentframe().f_back
    frameinfo = inspect.getframeinfo(f)
    filename = frameinfo.filename
    lineno = frameinfo.lineno
    # get the class name
    try:
        class_name = f.f_locals['self'].__class__.__name__
    except:
        class_name = None
    # get the function name
    func = frameinfo.function
    msg = f"\n\033[36m[{threading.get_ident()}] {filename}:{lineno} \033[0m"
    if class_name:
        msg += f"\033[1m\033[33m{class_name} [{func}]\033[0m "
    else:
        msg += f"\033[1m\033[33m[{func}]\033[0m "
    msg += " ".join(str(arg) for arg in args)
    print(msg, "\n", flush=True)
