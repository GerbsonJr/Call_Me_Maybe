from pydantic import BaseModel, Field, field_validator
from typing import Any


ALLOWED_JSON_TYPES = {
    "string", "number", "boolean", "integer", "object", "array", "null",
}


class ParameterDefinition(BaseModel):
    """Definition of a single function parameter."""
    type: str = Field(..., description="JSON schema type of the parameter.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that parameter type is supported."""
        if value not in ALLOWED_JSON_TYPES:
            raise ValueError(f"Unsupported parameter type: {value}")
        return value


class ReturnDefinition(BaseModel):
    """Definition of a function return type."""

    type: str = Field(..., description="JSON schema type of the return value.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Validate that return type is supported."""
        if value not in ALLOWED_JSON_TYPES:
            raise ValueError(f"Unsupported return type: {value}")
        return value


class FunctionDefinition(BaseModel):
    """Definition of an available callable function."""

    name: str = Field(..., description="Function name.")
    description: str = Field(..., description="Function description.")
    parameters: dict[str, ParameterDefinition] = Field(
        default_factory=dict,
        description="Map of parameter names to parameter definitions.",
    )
    returns: ReturnDefinition = Field(
        ..., description="Return schema definition.")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, value: str) -> str:
        """Ensure function name is not empty."""
        if not value.strip():
            raise ValueError("Function name cannot be empty.")
        return value


class InputItem(BaseModel):
    """Input prompt item from function_calling_tests.json."""

    prompt: str = Field(..., description="Natural-language prompt.")

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, value: str) -> str:
        """Ensure prompt is not empty."""
        if not value.strip():
            raise ValueError("Prompt cannot be empty.")
        return value


class OutputItem(BaseModel):
    """Output function call item to be written in results JSON."""

    prompt: str = Field(..., description="Original prompt.")
    name: str = Field(..., description="Chosen function name.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the chosen function.",
    )

    @field_validator("prompt", "name")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure required string fields are not empty."""
        if not value.strip():
            raise ValueError("Field cannot be empty.")
        return value
