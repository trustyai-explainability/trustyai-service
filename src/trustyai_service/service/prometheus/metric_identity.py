"""Canonical scheduler keys and compatible external metric identifiers."""

_CANONICAL_NAMES = {
    "IDENTITY": "BATCHMEAN",
    "BATCHMEAN": "BATCHMEAN",
    "MEANSHIFT": "COMPAREMEANS",
    "COMPAREMEANS": "COMPAREMEANS",
    "APPROXKSTEST": "KSTESTSTREAMING",
    "KSTESTSTREAMING": "KSTESTSTREAMING",
    "FOURIERMMD": "MMD",
    "MMD": "MMD",
    "KSTEST": "KSTEST",
    "JENSENSHANNON": "JENSENSHANNON",
}

_PYTHON_NAMES = {
    "IDENTITY": "BatchMean",
    "BATCHMEAN": "BatchMean",
    "MEANSHIFT": "Meanshift",
    "COMPAREMEANS": "CompareMeans",
    "KSTEST": "KSTest",
    "JENSENSHANNON": "JensenShannon",
}


def canonical_metric_name(name: str) -> str:
    """Resolve known aliases to one scheduler key; preserve extension names."""
    return _CANONICAL_NAMES.get(name.upper(), name)


def external_metric_name(name: str, *, legacy: bool) -> str:
    """Choose the exposition identifier without modifying the REST request."""
    if legacy:
        return _PYTHON_NAMES.get(name.upper(), name)
    return name.upper() if name.upper() in _CANONICAL_NAMES else name
