import os

def path(path:str) -> str:
    MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
    f_path = os.path.join(MAIN_DIR, path)
    return f_path