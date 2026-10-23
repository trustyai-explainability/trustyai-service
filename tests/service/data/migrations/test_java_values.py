"""Negative object-stream validation shared by CSV and legacy SQL migration."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from .test_pvc_csv_source import _TENSOR


@pytest.mark.parametrize(
    "mutation",
    [
        "class",
        "flags",
        "fields",
        "superclass",
        "annotations",
        "shape",
        "count",
        "element",
    ],
)
def test_tensor_rejects_unexpected_structure(mutation):
    """Valid streams with unsafe or inconsistent objects still fail closed."""
    import javaobj

    from trustyai_service.service.data.migrations.errors import MigrationValidationError
    from trustyai_service.service.data.migrations.java_values import tensor_value

    tensor = javaobj.loads(base64.b64decode(_TENSOR))
    if mutation == "class":
        tensor.classdesc.name = "example.Unexpected"
    elif mutation == "flags":
        tensor.classdesc.flags = 3
    elif mutation == "fields":
        tensor.classdesc.fields_names.append("unexpected")
    elif mutation == "superclass":
        tensor.classdesc.superclass = SimpleNamespace()
    elif mutation == "annotations":
        tensor.annotations.append("unexpected")
    elif mutation == "shape":
        tensor.dimensions[0] = 3
    elif mutation == "count":
        tensor.nEntries = 3
    else:
        tensor.data[0].classdesc.name = "example.Unexpected"
    with pytest.raises(MigrationValidationError):
        tensor_value(tensor)


def serialized_legacy_string(value):
    """Build the exact legacy SerializableObject wrapper for string data."""
    import javaobj
    from javaobj.v1.beans import JavaClass, JavaObject, JavaString

    descriptor = JavaClass()
    descriptor.name = "org.kie.trustyai.explainability.model.SerializableObject"
    descriptor.serialVersionUID = 1
    descriptor.flags = 2
    descriptor.fields_names = ["object"]
    descriptor.fields_types = [JavaString("Ljava/lang/Object;")]
    wrapper = JavaObject()
    wrapper.classdesc = descriptor
    wrapper.object = JavaString(value)
    return javaobj.dumps(wrapper)


def test_legacy_wrapper_and_value_are_allowlisted():
    """Only the service wrapper containing supported values is accepted."""
    import javaobj

    from trustyai_service.service.data.migrations.errors import MigrationValidationError
    from trustyai_service.service.data.migrations.java_values import legacy_value

    wrapper = javaobj.loads(serialized_legacy_string("hello"))
    assert legacy_value(wrapper) == "hello"
    wrapper.object = SimpleNamespace(
        classdesc=SimpleNamespace(name="example.Unexpected")
    )
    with pytest.raises(MigrationValidationError, match="unexpected element type"):
        legacy_value(wrapper)
    wrapper.classdesc.name = "example.Unexpected"
    with pytest.raises(MigrationValidationError, match="unexpected class"):
        legacy_value(wrapper)
