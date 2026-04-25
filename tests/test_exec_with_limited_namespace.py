import pytest
from arid_badger.exec_with_limited_namespace import (
    ExecWithLimitedNamespace,
    SecurityException,
)


class TestExecWithLimitedNamespace:
    @pytest.mark.parametrize(
        "import_statement",
        [
            "import os",
            "from os import path",
            "import numpy as np",
            "from numpy import *",
        ],
    )
    def test_imports_are_restricted(self, import_statement: str):
        with pytest.raises(SecurityException):
            ExecWithLimitedNamespace()(import_statement)

    def test_scope_names_are_usable(self):
        class ImagePatch:
            pass

        with pytest.raises(NameError):
            ExecWithLimitedNamespace()("ImagePatch")

        executor = ExecWithLimitedNamespace(scope={"ImagePatch": ImagePatch})
        executor("x = ImagePatch()")
        assert isinstance(executor.namespace["x"], ImagePatch)

    def test_cannot_open_files(self):
        with pytest.raises(SecurityException):
            ExecWithLimitedNamespace()("open('file.txt', 'w')")

    def test_cannot_access_locals_or_globals(self):
        executor = ExecWithLimitedNamespace()
        with pytest.raises(SecurityException):
            executor("locals()")
        with pytest.raises(SecurityException):
            executor("globals()")

    def test_cannot_access_subprocess(self):
        with pytest.raises(SecurityException):
            ExecWithLimitedNamespace()("import subprocess")

    def test_scope_name_can_be_called(self):
        def get_ipython():
            return "ok"

        executor = ExecWithLimitedNamespace(scope={"get_ipython": get_ipython})
        executor("result = get_ipython()")
        assert executor.namespace["result"] == "ok"

    def test_forbidden_names_are_rejected(self):
        def get_ipython():
            return "ok"

        executor = ExecWithLimitedNamespace(
            scope={"get_ipython": get_ipython},
            forbidden_names={"get_ipython"},
        )
        with pytest.raises(SecurityException):
            executor("get_ipython()")

    def test_serialize_excludes_builtins(self):
        executor = ExecWithLimitedNamespace()
        executor("x = 42")
        executor("y = 'hello'")
        import json

        serialized = json.loads(executor.serialize())
        assert serialized == {"x": "42", "y": "'hello'"}
