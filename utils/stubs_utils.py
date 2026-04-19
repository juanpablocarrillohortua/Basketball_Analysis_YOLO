import os
import pickle

def save_stubs(stub_path: str, object):
    if not os.path.exists(os.path.dirname(stub_path)):
        os.mkdir(os.path.dirname(stub_path))

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)

def read_stubs(read_from_stub: bool, stub_path: str):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None
