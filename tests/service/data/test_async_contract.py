import inspect
import importlib
import pytest


CLASSES = [
    ("src.service.data.storage.maria.maria", "MariaDBStorage"),
    ("src.service.data.storage.pvc", "PVCStorage"),
]

SYNC_ALLOWED = {
    "PVCStorage": {"allocate_valid_dataset_name", "get_lock"}, # no resource usage, non blocking
}

def public_methods(cls):
    for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in {"__init__", "__new__", "__class_getitem__"}:
            continue
        yield name, fn

@pytest.mark.parametrize("module_path,class_name", CLASSES)
def test_storage_methods_are_async(module_path, class_name):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    allowed = SYNC_ALLOWED.get(class_name, set())

    offenders = []
    for name, fn in public_methods(cls):
        if name in allowed:
            continue
        if not inspect.iscoroutinefunction(fn):
            offenders.append(name)

    assert not offenders, f"{class_name} has non-async methods: {offenders}"