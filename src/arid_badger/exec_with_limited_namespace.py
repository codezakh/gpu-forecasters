import ast
import builtins
import json
from typing import Any, final

DEFAULT_FORBIDDEN_NAMES = {
    "compile",
    "exec",
    "eval",
    "globals",
    "locals",
    "open",
    "input",
    "execfile",
    "__import__",
    "exit",
    "quit",
    "importlib",
}


def find_violations(
    code: str, forbidden_names: set[str]
) -> tuple[list[str], list[str]]:
    """
    Walk the AST once, collecting both imports and forbidden function calls.

    Returns:
        (imports, forbidden_calls)
    """
    imports: list[str] = []
    forbidden_calls: list[str] = []

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.append(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                name = f"{node.func.value.id}.{node.func.attr}"
            if name and name in forbidden_names:
                forbidden_calls.append(name)

    return imports, forbidden_calls


class SecurityException(Exception):
    pass


@final
class ExecWithLimitedNamespace:
    def __init__(
        self,
        scope: dict[str, Any] | None = None,
        forbidden_names: set[str] = DEFAULT_FORBIDDEN_NAMES,
    ):
        """
        This is a very janky way to get some security for the code we're running
        from the LLM. You can easily break out of this jail by doing Python tricks,
        but this is what I could whip up in a short time.

        Parameters
        -----------
        scope: dict[str, Any]
            Names exposed to the executed code on top of the (filtered) builtins.
            For the visual programming environment, this is where you put `image`,
            `ImagePatch`, `bool_to_yesno`, and so on. Filter at the call site —
            don't pass `locals()` blindly.
        forbidden_names: set[str]
            Names that are stripped from builtins AND rejected by an AST scan
            (so the LLM gets a SecurityException with a useful message instead
            of a NameError). The default covers the obvious filesystem / eval /
            import escape hatches. This is still "unsafe" — getattr tricks can
            still reach exec — but the LLM isn't an adversary here.
        """
        self.forbidden_names = forbidden_names
        self.builtins = {
            k: v for k, v in builtins.__dict__.items() if k not in forbidden_names
        }
        self.namespace: dict[str, Any] = {}
        self.namespace.update(self.builtins)
        if scope is not None:
            self.namespace.update(scope)

    def __call__(self, code: str):
        imports, forbidden_calls = find_violations(code, self.forbidden_names)
        if forbidden_calls:
            raise SecurityException(
                f"""Your code used the following not allowed functions: {forbidden_calls}.
Do not attempt to access the filesystem or network."""
            )
        if imports:
            raise SecurityException(
                "You are not allowed to use imports. Please use only the provided modules and functions."
            )
        bytecode = compile(code, filename="<string>", mode="exec")
        exec(bytecode, self.namespace, self.namespace)

    def serialize(self) -> str:
        namespace_to_repr = {
            k: repr(v)
            for k, v in self.namespace.items()
            if k not in self.builtins and k != "builtins" and not k.startswith("__")
        }
        return json.dumps(namespace_to_repr)
