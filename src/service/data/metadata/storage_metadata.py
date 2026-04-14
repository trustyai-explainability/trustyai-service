"""Storage metadata management for model data schemas and observation counts."""

import warnings
from typing import TypedDict, Unpack

from pydantic import BaseModel, ConfigDict, Field

from src.service.data.exceptions import InvalidSchemaException
from src.service.payloads.service.schema import Schema


class StorageMetadataKwargs(TypedDict, total=False):
    """Type definition for legacy keyword arguments.

    Used for backward compatibility with parameter-based initialization.
    """

    model_id: str
    input_schema: Schema
    output_schema: Schema
    input_tensor_name: str
    output_tensor_name: str
    observations: int
    recorded_inferences: bool


class StorageMetadataConfig(BaseModel):
    """Configuration for StorageMetadata initialization.

    Groups related parameters for cleaner function signatures and better maintainability.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow Schema type

    model_id: str = Field(..., description="Unique identifier for the model")
    input_schema: Schema = Field(..., description="Schema for model inputs")
    output_schema: Schema = Field(..., description="Schema for model outputs")
    input_tensor_name: str = Field(
        default="input", description="Name of the input tensor"
    )
    output_tensor_name: str = Field(
        default="output", description="Name of the output tensor"
    )
    observations: int = Field(
        default=0, ge=0, description="Number of observations recorded"
    )
    recorded_inferences: bool = Field(
        default=False,
        description="Whether inferences have been recorded",
    )


class StorageMetadata:
    """Metadata for model data storage including schemas and observation.

    counts.
    """

    def __init__(
        self,
        config: StorageMetadataConfig | None = None,
        **kwargs: Unpack[StorageMetadataKwargs],
    ) -> None:
        """Initialize storage metadata.

        Args:
            config: Configuration object containing all metadata parameters.
                Preferred method for initialization.
            **kwargs: Legacy keyword arguments for backward compatibility:
                - model_id (str): Unique identifier for the model
                - input_schema (Schema): Schema for model inputs
                - output_schema (Schema): Schema for model outputs
                - input_tensor_name (str): Name of the input tensor (default: "input")
                - output_tensor_name (str): Name of the output tensor (default: "output")
                - observations (int): Number of observations recorded (default: 0)
                - recorded_inferences (bool): Whether inferences have been recorded (default: False)

        Note:
            Prefer using the config parameter. Direct parameter passing is maintained
            for backward compatibility but is deprecated and will be removed in a future version.

        Raises:
            ValueError: If neither config nor required legacy parameters are provided.

        """
        # Support both new config-based and legacy parameter-based initialization
        if config is not None:
            self.model_id = config.model_id
            self.input_schema = config.input_schema
            self.output_schema = config.output_schema
            self.input_tensor_name = config.input_tensor_name
            self.output_tensor_name = config.output_tensor_name
            self.observations = config.observations
            self.recorded_inferences = config.recorded_inferences
        elif kwargs:
            # Legacy initialization path - issue deprecation warning
            warnings.warn(
                "Direct parameter passing to StorageMetadata is deprecated. "
                "Use StorageMetadataConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Require model_id and schemas
            model_id = kwargs.get("model_id")
            input_schema = kwargs.get("input_schema")
            output_schema = kwargs.get("output_schema")

            if model_id is None or input_schema is None or output_schema is None:
                msg = "Either config or (model_id, input_schema, output_schema) must be provided"
                raise ValueError(msg)

            self.model_id = model_id
            self.input_schema = input_schema
            self.output_schema = output_schema
            self.input_tensor_name = kwargs.get("input_tensor_name", "input")
            self.output_tensor_name = kwargs.get("output_tensor_name", "output")
            self.observations = kwargs.get("observations", 0)
            self.recorded_inferences = kwargs.get("recorded_inferences", False)
        else:
            msg = "Either config or keyword arguments must be provided"
            raise ValueError(msg)

    def get_model_id(self) -> str:
        """Get model ID."""
        return self.model_id

    def set_model_id(self, model_id: str) -> None:
        """Set model ID."""
        self.model_id = model_id

    def get_input_schema(self) -> Schema:
        """Get input schema."""
        return self.input_schema

    def set_input_schema(self, input_schema: Schema) -> None:
        """Set input schema."""
        self.input_schema = input_schema

    def get_output_schema(self) -> Schema:
        """Get output schema."""
        return self.output_schema

    def set_output_schema(self, output_schema: Schema) -> None:
        """Set output schema."""
        self.output_schema = output_schema

    def get_input_tensor_name(self) -> str:
        """Get input tensor name."""
        return self.input_tensor_name

    def set_input_tensor_name(self, input_tensor_name: str) -> None:
        """Set input tensor name."""
        self.input_tensor_name = input_tensor_name

    def get_output_tensor_name(self) -> str:
        """Get output tensor name."""
        return self.output_tensor_name

    def set_output_tensor_name(self, output_tensor_name: str) -> None:
        """Set output tensor name."""
        self.output_tensor_name = output_tensor_name

    def get_observations(self) -> int:
        """Get observation count."""
        return self.observations

    def set_observations(self, observations: int) -> None:
        """Set observation count."""
        self.observations = observations

    def increment_observations(self, observations: int) -> None:
        """Increment the observation count."""
        self.observations += observations

    def is_recorded_inferences(self) -> bool:
        """Check if inferences are recorded."""
        return self.recorded_inferences

    def set_recorded_inferences(self, *, recorded_inferences: bool) -> None:
        """Set whether inferences are recorded."""
        self.recorded_inferences = recorded_inferences

    def merge_input_schema(self, other_schema: Schema) -> None:
        """Merge another schema with the input schema."""
        if other_schema != self.input_schema:
            msg = "Original schema and schema-to-merge are not compatible"
            raise InvalidSchemaException(
                msg,
            )

    def merge_output_schema(self, other_schema: Schema) -> None:
        """Merge another schema with the output schema."""
        if other_schema != self.output_schema:
            msg = "Original schema and schema-to-merge are not compatible"
            raise InvalidSchemaException(
                msg,
            )

    def get_joint_name_aliases(self) -> dict[str, str]:
        """Get combined name mappings from input and output schemas."""
        joint_mapping = {}
        joint_mapping.update(self.input_schema.name_mapping)
        joint_mapping.update(self.output_schema.name_mapping)
        return joint_mapping
