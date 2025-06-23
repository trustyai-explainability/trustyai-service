import numpy as np
import pickle


def get_list_shape(lst: list):
    """Get the shape of a nested list, assuming the sublists are not jagged"""
    return [len(lst)] + get_list_shape(lst[0]) if isinstance(lst, list) else []


def contains_non_numeric(lst: list) -> bool:
    """Check if an arbitrarily deep nested list contains any non-numeric elements"""
    if isinstance(lst, (list, np.ndarray)):
        return any(contains_non_numeric(item) for item in lst)
    else:
        return isinstance(lst, (bool, str))


def serialize_rows(lst: list, max_void_type_length):
    """Convert a nested list to a 1D numpy array, where the nth element contains a bytes serialization of the nth row"""
    serialized = [np.void(pickle.dumps(row)) for row in lst]
    return np.array(serialized, dtype=f"V{max_void_type_length}")


def deserialize_rows(serialized: np.ndarray):
    """Convert a 1D numpy array from `serialize_rows` to a numpy object array"""
    deserialized = [pickle.loads(row) for row in serialized]
    return np.array(deserialized, dtype="O")
