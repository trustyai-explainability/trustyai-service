from typing import Dict

from src.service.data.exceptions import InvalidSchemaException
from src.service.payloads.service.schema import Schema


class StorageMetadata:
    def __init__(
        self,
        model_id: str,
        input_schema: Schema,
        output_schema: Schema,
        input_tensor_name: str = "input",
        output_tensor_name: str = "output",
        observations: int = 0,
        recorded_inferences: bool = False,
    ) -> None:
        self.model_id = model_id
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name
        self.observations = observations
        self.recorded_inferences = recorded_inferences

    def get_model_id(self) -> str:
        return self.model_id

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def get_input_schema(self) -> Schema:
        return self.input_schema

    def set_input_schema(self, input_schema: Schema) -> None:
        self.input_schema = input_schema

    def get_output_schema(self) -> Schema:
        return self.output_schema

    def set_output_schema(self, output_schema: Schema) -> None:
        self.output_schema = output_schema

    def get_input_tensor_name(self) -> str:
        return self.input_tensor_name

    def set_input_tensor_name(self, input_tensor_name: str) -> None:
        self.input_tensor_name = input_tensor_name

    def get_output_tensor_name(self) -> str:
        return self.output_tensor_name

    def set_output_tensor_name(self, output_tensor_name: str) -> None:
        self.output_tensor_name = output_tensor_name

    def get_observations(self) -> int:
        return self.observations

    def set_observations(self, observations: int) -> None:
        self.observations = observations

    def increment_observations(self, observations: int) -> None:
        """Increment the observation count."""
        self.observations += observations

    def is_recorded_inferences(self) -> bool:
        return self.recorded_inferences

    def set_recorded_inferences(self, recorded_inferences: bool) -> None:
        self.recorded_inferences = recorded_inferences

    def merge_input_schema(self, other_schema: Schema) -> None:
        """Merge another schema with the input schema."""
        if other_schema != self.input_schema:
            raise InvalidSchemaException("Original schema and schema-to-merge are not compatible")

    def merge_output_schema(self, other_schema: Schema) -> None:
        """Merge another schema with the output schema."""
        if other_schema != self.output_schema:
            raise InvalidSchemaException("Original schema and schema-to-merge are not compatible")

    def get_joint_name_aliases(self) -> Dict[str, str]:
        """Get combined name mappings from input and output schemas."""
        joint_mapping = {}
        joint_mapping.update(self.input_schema.name_mapping)
        joint_mapping.update(self.output_schema.name_mapping)
        return joint_mapping
