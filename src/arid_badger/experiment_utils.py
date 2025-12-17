import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Self


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
    def __init__(
        self,
        identifier: ExperimentIdentifier,
        workspace_path: Path = Path("./workspace"),
    ):
        self.identifier = identifier
        self.workspace_path = workspace_path

    @property
    def output_dir_name(self) -> str:
        return (
            f"experiments__{self.identifier.number:04d}_{self.identifier.description}"
        )

    @property
    def output_dir(self) -> Path:
        return self.workspace_path / self.output_dir_name

    def setup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
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
