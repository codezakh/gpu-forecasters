from typing import Optional
from dataclasses import dataclass

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code
from litellm import completion
from ulid import ULID


@dataclass
class MutationContext:
    """Context for mutating a kernel."""

    previous_kernel_code: Optional[str] = None
    previous_kernel_ulid: Optional[str] = None
    prompt: str = ""
    ref_arch_src: str = ""
    backend: str = "cuda"
    precision: str = "fp32"


@dataclass
class MutatedKernel:
    """Result of a kernel mutation."""

    kernel_code: str
    ulid: str
    ancestor_ulid: Optional[str] = None


class MutationFunction:
    """Generates mutated kernels using an LLM."""

    def __init__(
        self,
        model: str = "gemini/gemini-3-flash-preview",
        backend: str = "cuda",
        option: str = "one_shot",
        precision: str = "fp32",
    ):
        """
        Initialize the mutation function.

        Args:
            model: LiteLLM model identifier
            backend: Kernel backend ("cuda", "triton", etc.)
            option: Prompt option ("zero_shot", "one_shot", "few_shot")
            precision: Precision string ("fp32", "fp16", "bf16")
        """
        self.model = model
        self.backend = backend
        self.option = option
        self.precision = precision

    def __call__(self, context: MutationContext) -> MutatedKernel:
        """
        Generate a mutated kernel from the context.

        Args:
            context: Mutation context containing previous kernel and prompt

        Returns:
            MutatedKernel with code, ulid, and ancestor_ulid
        """
        # Build prompt using KernelBench's default prompt constructor
        prompt = get_prompt_for_backend(
            ref_arch_src=context.ref_arch_src,
            backend=self.backend,
            option=self.option,
            precision=self.precision,
        )

        # Print prompt for debugging (as specified in requirements)
        print("=" * 80)
        print("MUTATION PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        # Call LLM using LiteLLM
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=20.0,
        )

        # Extract content from response
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        if content is None:
            raise ValueError("LLM returned empty content")

        # Parse code from response using KernelBench's extract_first_code utility
        # The prompt asks for Python code, so we look for python code blocks
        kernel_code = extract_first_code(content, code_language_types=["python"])

        if not kernel_code:
            raise ValueError(
                f"Could not extract code from LLM response. Response: {content[:500]}"
            )

        # Generate ULID for this mutation
        ulid = str(ULID())

        # Create MutatedKernel with ancestor reference
        return MutatedKernel(
            kernel_code=kernel_code,
            ulid=ulid,
            ancestor_ulid=context.previous_kernel_ulid,
        )
