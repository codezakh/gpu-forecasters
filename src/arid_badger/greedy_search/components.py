from typing import Optional
from dataclasses import dataclass

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code
from litellm import completion
from ulid import ULID
from pydantic import BaseModel, Field, computed_field
from typing import Literal
import attrs


class MutationContext(BaseModel):
    """Context for mutating a kernel."""

    previous_kernel_code: Optional[str] = Field(default=None)
    previous_kernel_ulid: Optional[ULID] = Field(default=None)
    model_slug: str = Field(default="gemini/gemini-3-flash-preview")
    ref_arch_src: str = Field(default="")
    backend: Literal["cuda", "triton"] = Field(default="cuda")
    prompt_option: Literal["zero_shot", "one_shot", "few_shot"] = Field(
        default="one_shot"
    )
    precision: Literal["fp32", "fp16", "bf16"] = Field(default="fp32")

    @computed_field
    @property
    def prompt(self) -> str:
        return get_prompt_for_backend(
            ref_arch_src=self.ref_arch_src,
            backend=self.backend,
            option=self.prompt_option,
            precision=self.precision,
        )


class MutatedKernel(BaseModel):
    """Result of a kernel mutation."""

    kernel_code: str
    ulid: ULID = Field(default_factory=ULID)
    ancestor_ulid: Optional[ULID] = Field(default=None)


class MutationFunction:
    """Generates mutated kernels using an LLM."""

    def __call__(self, context: MutationContext) -> MutatedKernel:
        """
        Generate a mutated kernel from the context.

        Args:
            context: Mutation context containing previous kernel and prompt

        Returns:
            MutatedKernel with code, ulid, and ancestor_ulid
        """
        # Call LLM using LiteLLM
        response = completion(
            model=context.model_slug,
            messages=[{"role": "user", "content": context.prompt}],
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

        # Create MutatedKernel with ancestor reference
        return MutatedKernel(
            kernel_code=kernel_code,
            ancestor_ulid=context.previous_kernel_ulid,
        )
