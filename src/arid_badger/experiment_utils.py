import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from loguru import logger


class _LoguruInterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru.

    Lifted verbatim from the copy that was being duplicated across
    e0019 and e0020. Frame-walking depth of 6 matches the stdlib
    ``logging`` module's internal call stack so loguru reports the
    caller's frame rather than this handler's.
    """

    def emit(self, record: logging.LogRecord) -> None:  # pyright: ignore[reportImplicitOverride]
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6  # pyright: ignore[reportPrivateUsage]
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def install_loguru_intercept(level: int = logging.WARNING) -> None:
    """Install a stdlib-logging → loguru bridge at the root logger.

    Idempotent per-process: calling it multiple times replaces the root
    handler list (via ``force=True``) rather than stacking duplicates.
    """
    logging.basicConfig(
        handlers=[_LoguruInterceptHandler()], level=level, force=True
    )


@dataclass(frozen=True)
class ExperimentIdentifier:
    number: int
    description: str

    @classmethod
    def parse_from_script_name(cls, script_name: str) -> Self:
        """
        Parse an experiment identifier from a script name.

        Example:
        ```python
        print(__file__) # Should be something like experiments/e0001_example_experiment.py
        identifier = ExperimentIdentifier.parse_from_script_name(__file__)
        # Also works with just the script name
        identifier = ExperimentIdentifier.parse_from_script_name("e0001_example_experiment")
        ```


        Args:
            script_name: The name of the script to parse the identifier from.
                Should be of the form e<number>_<description>.py

        Returns:
            An experiment identifier.
        """
        name_as_path = Path(script_name)
        stem = name_as_path.stem
        # We expect the script name to be of the form e<number>_<description>.py
        # where <number> is a unique identifier for the experiment and <name> is the name of the experiment.
        # Ex: e201_train_model.py
        pattern = r"^e(\d+)_(.+)$"
        match = re.match(pattern, stem)
        if not match:
            raise ValueError(f"Invalid experiment name format: {stem}")
        number = int(match.group(1))
        description = match.group(2)
        return cls(number=number, description=description)


class ExperimentWorkspaceManager:
    """Manages the on-disk output directory for a single experiment.

    Each experiment gets a deterministic output path under ``workspace/``
    derived from its number and description::

        workspace/experiments__0042_my_experiment/

    Note the **double underscore** and **no ``e`` prefix** — this is
    intentional and differs from the source directory name
    (``experiments/e0042_my_experiment/``).

    **Canonical usage** — declare once in the experiment's ``__init__.py``::

        from arid_badger.experiment_utils import ExperimentWorkspaceManager

        WORKSPACE = ExperimentWorkspaceManager.from_module_name(__name__)

    Then access from other modules in the same experiment::

        from . import WORKSPACE

        WORKSPACE.setup()          # creates the directory (idempotent)
        out = WORKSPACE.output_dir # Path to workspace/experiments__0042_.../
    """

    def __init__(
        self,
        identifier: ExperimentIdentifier,
        workspace_path: Path = Path("./workspace"),
    ):
        self.identifier = identifier
        self.workspace_path = workspace_path

    @property
    def output_dir_name(self) -> str:
        """Directory basename, e.g. ``experiments__0042_my_experiment``."""
        return (
            f"experiments__{self.identifier.number:04d}_{self.identifier.description}"
        )

    @property
    def output_dir(self) -> Path:
        """Full path to the experiment's output directory (may not exist yet)."""
        return self.workspace_path / self.output_dir_name

    def setup(self) -> None:
        """Create the output directory if it doesn't already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Delete the output directory and all its contents."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    @classmethod
    def from_file_path(cls, file_path: Path) -> Self:
        """
        Initialize a workspace manager from a file path (e.g. the __file__ of an experiment script)

        Args:
            file_path: The path of the file to initialize the workspace manager from.

        Returns:
            A workspace manager for the experiment.
        """
        identifier = ExperimentIdentifier.parse_from_script_name(file_path.stem)
        return cls(identifier)

    @classmethod
    def from_module_name(cls, module_name: str) -> Self:
        """
        Initialize a workspace manager from a module name.

        Example:
        ```python
        print(__name__) # Should be something like experiments.e0001_example_experiment
        # If you are in the __init__.py of an experiment folder, this will just work
        workspace = ExperimentWorkspaceManager.from_module_name(__name__)
        ```

        Args:
            module_name: The name of the module to initialize the workspace manager from.
                Should be of the form experiments.<experiment_name>.

        Returns:
            A workspace manager for the experiment.
        """

        path_parts = module_name.split(".")

        if len(path_parts) != 2:
            error_msg = (
                "Expected to be able to split module name into two parts by the dot, "
                f"got {path_parts}"
            )
            raise ValueError(error_msg)

        experiment_root, experiment_name = module_name.rsplit(".", 1)

        if experiment_root != "experiments":
            error_msg = (
                "Expected experiment root to be 'experiments', got "
                f"{experiment_root} by splitting {module_name}"
            )
            raise ValueError(error_msg)

        identifier = ExperimentIdentifier.parse_from_script_name(experiment_name)
        return cls(identifier)
