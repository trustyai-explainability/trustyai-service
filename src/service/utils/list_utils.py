import numpy as np
import pickle


def get_list_shape(l: list):
    """Get the shape of a nested list, assuming the sublists are not jagged"""
    return [len(l)] + get_list_shape(l[0]) if isinstance(l, list) else []


def contains_non_numeric(l: list) -> bool:
    """Check if an arbitrarily deep nested list contains any non-numeric elements"""
    if isinstance(l, (list, np.ndarray)):
        return any(contains_non_numeric(item) for item in l)
    else:
        return isinstance(l, (bool, str))


def serialize_rows(l: list):
    """Convert a nested list to a 1D numpy array, where the nth element contains a bytes serialization of the nth row"""
    serialized = [np.void(pickle.dumps(row)) for row in l]
    return np.array(serialized)


def deserialize_rows(serialized: np.ndarray):
    """Convert a 1D numpy array from `serialize_rows` to a numpy object array"""
    deserialized = [pickle.loads(row) for row in serialized]
    return np.array(deserialized, dtype="O")