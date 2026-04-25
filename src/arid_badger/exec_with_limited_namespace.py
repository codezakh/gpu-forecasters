import ast
import builtins
import json
from typing import Optional

DEFAULT_RESTRICTED_BUILTINS = {
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


def find_imports(code: str) -> list:
    """
    Identify import statements in the given Python code.

    Args:
        code (str): The Python code to analyze.

    Returns:
        list: A list of import statements found in the code.
    """
    import_statements = []

    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Traverse the AST to find import statements
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_statements.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name:
                for alias in node.names:
                    import_statements.append(f"{module_name}.{alias.name}")

    return import_statements


def find_not_allowed_functions(code: str, restricted_functions: set) -> list:
    """
    Identify not allowed function calls in the given Python code.

    Args:
        code (str): The Python code to analyze.
        restricted_functions (list): A list of functions not allowed to be called.

    Returns:
        list: A list of not allowed function calls found in the code.
    """
    not_allowed_functions = []

    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Traverse the AST to find function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = None
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                function_name = f"{node.func.value.id}.{node.func.attr}"
            if function_name and function_name in restricted_functions:
                not_allowed_functions.append(function_name)

    return not_allowed_functions


class SecurityException(Exception):
    pass


class ExecWithLimitedNamespace:
    def __init__(
        self,
        allowed_names: Optional[set[str]] = None,
        restricted_names: Optional[set[str]] = None,
        restricted_builtins: set[str] = DEFAULT_RESTRICTED_BUILTINS,
        inherited_scope: Optional[dict] = None,
    ):
        """
        This is a very janky way to get some security for the code we're running
        from the LLM. You can easily break out of this jail by doing Python tricks,
        but this is what I could whip up in a short time.

        Parameters
        -----------
        allowed_names: set[str]
            These are names that will explicitly be allowed in the namespace. For the
            visual programming environment, you want to give the agent access to `image`,
            `ImagePatch`, `bool_to_yesno`, and so on.
        restricted_names: set[str]
            These are function calls that are not allowed. We already have a mechanism to prevent using
            anything but allowed builtins and the whitelisted names in allowed_names, but this
            is an extra layer of security. We will check the ast to make sure none of these functions
            are called. The one I specifically want to disable is stuff like `get_ipython`, because it
            allows you to run shell commands.
        restricted_builtins: set[str]
            I've already given a reasonable set in DEFAULT_RESTRICTED_BUILTINS. This set is still "unsafe"
            because you can do stuff with getattr that will let you exec stuff. But I don't think the LLM
            will be doing anything like this.
        inherited_scope: Optional[dict]
            These are the variables from the enclosing scope. Anything from `allowed_names` will be inherited from
            the enclosing scope, while everything else will be inaccessible.
        """

        if restricted_names is None:
            self.restricted_names: set[str] = set()
        else:
            self.restricted_names = restricted_names

        self.restricted_builtins = restricted_builtins

        self.inherited_scope = inherited_scope or {}
        self.builtins = {
            k: v
            for k, v in builtins.__dict__.items()
            if k not in self.restricted_builtins
        }
        self.namespace = {}
        self.namespace.update(self.builtins)

        if allowed_names is not None:
            self.namespace.update(
                {k: v for k, v in self.inherited_scope.items() if k in allowed_names}
            )

    def __call__(self, code: str):
        imports = find_imports(code)
        not_allowed_functions = find_not_allowed_functions(
            code, self.restricted_builtins | self.restricted_names
        )
        if not_allowed_functions:
            raise SecurityException(
                f"""Your code used the following not allowed functions: {not_allowed_functions}.
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
