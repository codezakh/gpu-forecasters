import pytest
from arid_badger.local_code_execution import (
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
    def test_imports_are_restricted(self, import_statement):
        with pytest.raises(SecurityException):
            executor = ExecWithLimitedNamespace()
            executor(import_statement)

    def test_allowed_names_are_usable(self):
        class ImagePatch:
            pass

        executor = ExecWithLimitedNamespace()
        with pytest.raises(NameError):
            executor("ImagePatch")
        executor = ExecWithLimitedNamespace(allowed_names={"ImagePatch"})

    def test_cannot_open_files(self):
        with pytest.raises(SecurityException):
            executor = ExecWithLimitedNamespace()
            executor("open('file.txt', 'w')")

    def test_cannot_access_locals_or_globals(self):
        executor = ExecWithLimitedNamespace()
        with pytest.raises(SecurityException):
            executor("locals()")
        with pytest.raises(SecurityException):
            executor("globals()")

    def test_cannot_access_subprocess(self):
        with pytest.raises(SecurityException):
            executor = ExecWithLimitedNamespace()
            executor("import subprocess")

    def test_cannot_access_restricted_names(self):
        def get_ipython():
            pass

        executor = ExecWithLimitedNamespace(
            inherited_scope=locals(), allowed_names={"get_ipython"}
        )
        executor("get_ipython()")
        with pytest.raises(SecurityException):
            executor = ExecWithLimitedNamespace(
                inherited_scope=locals(), restricted_names={"get_ipython"}
            )
            executor("get_ipython()")
