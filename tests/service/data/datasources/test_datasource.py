from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.service.constants import UNLABELED_TAG
from src.service.data.datasources.data_source import DataSource
from src.service.data.exceptions import DataframeCreateException, StorageReadException
from src.service.data.metadata.storage_metadata import StorageMetadata
from src.service.data.model_data import ModelData
from src.service.payloads.service.schema import Schema
from src.service.payloads.service.schema_item import SchemaItem
from src.service.payloads.values.data_type import DataType


class TestDataSource:
    """Test DataSource functionality."""

    @pytest.fixture
    def data_source(self) -> DataSource:
        return DataSource()

    @pytest.fixture
    def mock_model_data(self) -> Mock:
        mock = Mock(spec=ModelData)

        # Mock async methods
        mock.row_counts = AsyncMock(return_value=(100, 100, 100))
        mock.column_names = AsyncMock(
            return_value=(["feature1", "feature2"], ["target"], ["metadata1"])
        )
        mock.data = AsyncMock(
            return_value=([[1, 2], [3, 4]], [[0], [1]], [["meta1"], ["meta2"]])
        )

        return mock

    @pytest.fixture
    def sample_metadata(self) -> StorageMetadata:
        input_items = {
            "feature1": SchemaItem(DataType.DOUBLE, "feature1", 0),
            "feature2": SchemaItem(DataType.DOUBLE, "feature2", 1),
        }
        input_schema = Schema(input_items)

        output_items = {"target": SchemaItem(DataType.DOUBLE, "target", 0)}
        output_schema = Schema(output_items)

        return StorageMetadata(
            model_id="test_model",
            input_schema=input_schema,
            output_schema=output_schema,
            observations=100,
            recorded_inferences=True,
        )

    def test_initialization(self, data_source: DataSource) -> None:
        """Test DataSource initialization."""
        assert isinstance(data_source.known_models, set)
        assert len(data_source.known_models) == 0
        assert data_source.storage_interface is not None
        assert isinstance(data_source.metadata_cache, dict)
        assert len(data_source.metadata_cache) == 0
        assert data_source.executor is not None

    @pytest.mark.asyncio
    async def test_add_model_to_known(self, data_source: DataSource) -> None:
        """Test adding model to known models."""

        model_id = "test_model"
        await data_source.add_model_to_known(model_id)

        assert model_id in data_source.known_models
        assert len(data_source.known_models) == 1

    @pytest.mark.asyncio
    async def test_get_known_models(self, data_source: DataSource) -> None:
        """Test getting known models."""

        model_ids = ["model1", "model2", "model3"]

        for model_id in model_ids:
            await data_source.add_model_to_known(model_id)

        known = await data_source.get_known_models()
        assert known == set(model_ids)

        # Ensure it's a copy (not the original set)
        known.add("new_model")
        assert "new_model" not in data_source.known_models

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_dataframe_with_batch_size_success(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test successful dataframe retrieval with batch size."""

        mock_model_data_class.return_value = mock_model_data

        mock_model_data.data.return_value = (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.0], [1.0]]),
            np.array([["meta1"], ["meta2"]]),
        )

        df = await data_source.get_dataframe_with_batch_size("test_model", 50)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert "target" in df.columns
        assert "metadata1" in df.columns

        # Verify ModelData was called correctly
        mock_model_data.row_counts.assert_called_once()
        mock_model_data.column_names.assert_called_once()
        mock_model_data.data.assert_called_once()

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_dataframe_default_batch_size(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test dataframe retrieval with default batch size."""

        mock_model_data_class.return_value = mock_model_data

        mock_model_data.data.return_value = (
            np.array([[1.0, 2.0]]),
            np.array([[0.0]]),
            np.array([["meta1"]]),
        )

        with patch.dict("os.environ", {"SERVICE_BATCH_SIZE": "200"}):
            df = await data_source.get_dataframe("test_model")

        assert isinstance(df, pd.DataFrame)
        mock_model_data.data.assert_called_once()

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_dataframe_handles_exceptions(
        self, mock_model_data_class: Mock, data_source: DataSource
    ) -> None:
        """Test that dataframe creation exceptions are handled properly."""

        mock_model_data = Mock()
        mock_model_data.row_counts.side_effect = Exception("Test error")
        mock_model_data_class.return_value = mock_model_data

        with pytest.raises(
            DataframeCreateException,
            match="Error creating dataframe for model=test_model",
        ):
            await data_source.get_dataframe("test_model")

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_organic_dataframe_filters_unlabeled(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test that organic dataframe filters out unlabeled (synthetic) data."""

        mock_model_data_class.return_value = mock_model_data

        # Add unlabeled column to mock data
        mock_model_data.column_names.return_value = (
            ["feature1", "feature2"],
            ["target"],
            [UNLABELED_TAG],
        )
        mock_model_data.data.return_value = (
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            np.array([[0.0], [1.0], [0.0]]),
            np.array([[False], [True], [False]]),  # Second row is synthetic
        )

        df = await data_source.get_organic_dataframe("test_model", 100)

        # Should filter out synthetic rows
        assert len(df) == 2  # Should exclude the synthetic row
        assert not df[UNLABELED_TAG].any()  # No True values should remain

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_metadata_creates_and_caches(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test metadata creation and caching."""

        mock_model_data_class.return_value = mock_model_data

        metadata = await data_source.get_metadata("test_model")

        assert isinstance(metadata, StorageMetadata)
        assert metadata.get_model_id() == "test_model"
        assert "test_model" in data_source.metadata_cache

        # Second call should use cache
        metadata2 = await data_source.get_metadata("test_model")
        assert metadata2 is metadata  # Same object from cache

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_metadata_handles_exceptions(
        self, mock_model_data_class: Mock, data_source: DataSource
    ) -> None:
        """Test metadata retrieval exception handling."""

        mock_model_data = Mock()
        mock_model_data.row_counts.side_effect = Exception("Test error")
        mock_model_data_class.return_value = mock_model_data

        with pytest.raises(
            StorageReadException, match="Error getting metadata for model=test_model"
        ):
            await data_source.get_metadata("test_model")

    @pytest.mark.asyncio
    async def test_has_metadata_true(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test has_metadata returns True when metadata exists."""

        data_source.metadata_cache["test_model"] = sample_metadata

        assert await data_source.has_metadata("test_model") is True

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_has_metadata_false(
        self, mock_model_data_class: Mock, data_source: DataSource
    ) -> None:
        """Test has_metadata returns False when metadata doesn't exist."""

        mock_model_data = Mock()
        mock_model_data.row_counts.side_effect = Exception("Not found")
        mock_model_data_class.return_value = mock_model_data

        assert await data_source.has_metadata("nonexistent_model") is False

    @pytest.mark.asyncio
    async def test_get_num_observations(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test getting number of observations."""

        data_source.metadata_cache["test_model"] = sample_metadata

        count = await data_source.get_num_observations("test_model")
        assert count == 100

    @pytest.mark.asyncio
    async def test_has_recorded_inferences(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test checking for recorded inferences."""

        data_source.metadata_cache["test_model"] = sample_metadata

        has_inferences = await data_source.has_recorded_inferences("test_model")
        assert has_inferences is True

    @pytest.mark.asyncio
    async def test_get_verified_models_from_known(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test getting verified models from known models."""

        await data_source.add_model_to_known("test_model")
        await data_source.add_model_to_known("invalid_model")
        data_source.metadata_cache["test_model"] = sample_metadata

        verified = await data_source.get_verified_models()
        assert "test_model" in verified
        assert "invalid_model" not in verified
        assert len(verified) == 1

    @patch.dict("os.environ", {"TEST_MODEL_ID": "discovered_model"})
    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_verified_models_discovers_from_storage(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test discovering models from storage when no known models exist."""

        mock_model_data_class.return_value = mock_model_data

        verified = await data_source.get_verified_models()

        assert "discovered_model" in verified
        assert "discovered_model" in data_source.known_models

    def test_ground_truth_name_generation(self) -> None:
        """Test ground truth name generation."""
        name = DataSource.get_ground_truth_name("test_model")
        assert name == "test_model-ground-truths"

    @pytest.mark.asyncio
    async def test_has_ground_truths(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test checking for ground truths."""

        gt_name = DataSource.get_ground_truth_name("test_model")
        data_source.metadata_cache[gt_name] = sample_metadata

        has_gt = await data_source.has_ground_truths("test_model")
        assert has_gt is True

        has_gt_missing = await data_source.has_ground_truths("missing_model")
        assert has_gt_missing is False

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_get_ground_truths(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test getting ground truths dataframe."""
        mock_model_data_class.return_value = mock_model_data

        mock_model_data.data.return_value = (
            np.array([[1.0, 2.0]]),
            np.array([[1.0]]),
            np.array([["ground_truth"]]),
        )

        gt_df = await data_source.get_ground_truths("test_model")
        assert isinstance(gt_df, pd.DataFrame)

        # Verify correct model name was used
        expected_gt_name = DataSource.get_ground_truth_name("test_model")
        mock_model_data_class.assert_called_with(expected_gt_name)

    @pytest.mark.asyncio
    async def test_save_dataframe(self, data_source: DataSource) -> None:
        """Test saving dataframe."""

        df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

        await data_source.save_dataframe(df, "test_model", overwrite=True)

        # Should add model to known models
        assert "test_model" in data_source.known_models

    def test_save_metadata(
        self, data_source: DataSource, sample_metadata: StorageMetadata
    ) -> None:
        """Test saving metadata."""

        data_source.save_metadata(sample_metadata, "test_model")

        # Should cache the metadata
        assert "test_model" in data_source.metadata_cache
        assert data_source.metadata_cache["test_model"] == sample_metadata

    @patch("src.service.data.datasources.data_source.ModelData")
    @pytest.mark.asyncio
    async def test_batch_size_calculation_with_limited_data(
        self,
        mock_model_data_class: Mock,
        data_source: DataSource,
        mock_model_data: Mock,
    ) -> None:
        """Test batch size calculation when limited data is available."""
        # Mock limited data availability
        mock_model_data.row_counts.return_value = (10, 10, 10)  # Only 10 rows available
        mock_model_data_class.return_value = mock_model_data

        mock_model_data.data.return_value = (
            np.array([[1.0, 2.0]] * 10),
            np.array([[0.0]] * 10),
            np.array([["meta"]] * 10),
        )

        # Request more data than available
        df = await data_source.get_dataframe_with_batch_size("test_model", 100)

        # Should get all available data
        assert len(df) == 10

        # Verify correct parameters were passed to ModelData.data()
        call_args = mock_model_data.data.call_args[1]
        assert call_args["start_row"] == 0  # Should start from beginning
        assert call_args["n_rows"] == 10  # Should get all 10 rows
