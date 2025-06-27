import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Dict, Optional, Set, List, Tuple
from pandas import DataFrame
from pandas.errors import DataError
from src.endpoints.metrics.metrics_directory import MetricsDirectory
from src.service.data.datasources.data_source import DataSource
from src.service.data.exceptions import StorageReadException, DataframeCreateException
from src.service.data.storage.pvc import MissingH5PYDataException
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest
from src.service.payloads.metrics.request_reconciler import RequestReconciler
from src.service.prometheus.prometheus_publisher import PrometheusPublisher
from src.service.prometheus.metric_value_carrier import MetricValueCarrier

logger: logging.Logger = logging.getLogger(__name__)


class PrometheusScheduler:

    @staticmethod
    def get_service_config() -> Dict:
        """Get service configuration from environment variables."""

        metrics_schedule = os.getenv("SERVICE_METRICS_SCHEDULE", "30s")
        batch_size = int(os.getenv("SERVICE_BATCH_SIZE", "100"))

        if metrics_schedule.endswith("s"):
            interval = int(metrics_schedule[:-1])
        elif metrics_schedule.endswith("m"):
            interval = int(metrics_schedule[:-1]) * 60
        elif metrics_schedule.endswith("h"):
            interval = int(metrics_schedule[:-1]) * 60 * 60
        else:
            interval = 30  # Default to 30 seconds

        return {"batch_size": batch_size, "metrics_schedule": interval}

    def __init__(
        self,
        publisher: Optional[PrometheusPublisher] = None,
        data_source: Optional[DataSource] = None,
    ) -> None:
        self.requests: Dict[str, Dict[uuid.UUID, BaseMetricRequest]] = defaultdict(dict)
        self.has_logged_skipped_request_message: Set[str] = set()
        self.metrics_directory: MetricsDirectory = MetricsDirectory()
        self.publisher: PrometheusPublisher = publisher or PrometheusPublisher()
        self.data_source: DataSource = data_source or DataSource()
        self.service_config: Dict = self.get_service_config()
        self._logged_skipped_request_message_lock: threading.Lock = threading.Lock()
        self._requests_lock: threading.Lock = threading.Lock()

    def get_requests(self, metric_name: str) -> Dict[uuid.UUID, BaseMetricRequest]:
        """Get all requests for a specific metric."""
        with self._requests_lock:
            return dict(self.requests.get(metric_name, {}))

    def get_all_requests_flat(self) -> Dict[uuid.UUID, BaseMetricRequest]:
        """Get all requests across all metrics as a flat dictionary."""
        result = {}
        with self._requests_lock:
            for metric_dict in self.requests.values():
                result.update(metric_dict)
        return result

    async def calculate(self) -> None:
        """Calculate scheduled metrics."""
        await self.calculate_manual(False)

    async def calculate_manual(self, throw_errors: bool = True) -> None:
        """
        Calculate scheduled metrics.

        Args:
            throw_errors: If True, errors will be thrown. If False, they will just be logged.
        """
        try:
            verified_models = await self._get_verified_models(throw_errors)

            # Global service statistic
            self._publish_global_statistics(verified_models, throw_errors)

            # Process each model
            for model_id in verified_models:
                await self._process_model(model_id, throw_errors)
                
        except Exception as e:
            self._handle_error(e, "Unexpected error during metric calculation", throw_errors)
    

    async def register(
        self, metric_name: str, id: uuid.UUID, request: BaseMetricRequest
    ) -> None:
        """Register a metric request."""
        await RequestReconciler.reconcile(request, self.data_source)
        with self._requests_lock:
            if metric_name not in self.requests:
                self.requests[metric_name] = {}
            self.requests[metric_name][id] = request

    async def delete(self, metric_name: str, id: uuid.UUID) -> None:
        """Delete a metric request."""
        with self._requests_lock:
            if metric_name in self.requests and id in self.requests[metric_name]:
                del self.requests[metric_name][id]

        self.publisher.remove_gauge(metric_name, id)

    def has_requests(self) -> bool:
        """Check if there are any requests."""
        with self._requests_lock:
            return any(bool(requests) for requests in self.requests.values())

    def get_model_ids(self) -> Set[str]:
        """Get unique model IDs with registered Prometheus metrics."""
        model_ids = set()
        for request in self.get_all_requests_flat().values():
            model_ids.add(request.model_id)
        return model_ids

    def _handle_error(self, error: Exception, context: str, throw_errors: bool) -> None:
        """
        Helper function to handle an error.
        
        Args:
            error: The exception that occurred.
            context: Description of where the error occurred.
            throw_errors: Whether to raise the exception or just log it.
            
        Raises:
            The original exception if throw_errors is True.
        """
        error_msg = f"{context}: {error}"
        logger.error(error_msg)
        if throw_errors:
            raise error

    async def _get_verified_models(self, throw_errors: bool) -> List[str]:
        """
        Retrieve list of verified models from data source.
        
        Args:
            throw_errors: Whether to raise exceptions or just log them.
            
        Returns:
            List of verified model IDs, or empty list if error occurred.
        """
        try:
            return await self.data_source.get_verified_models()
        except (StorageReadException, MissingH5PYDataException) as e:
            self._handle_error(e, "Failed to retrieve verified models from data source", throw_errors)
            return []
        except (OSError, IOError) as e:
            self._handle_error(e, "File system error while retrieving verified models", throw_errors)
            return []
    
    def _publish_global_statistics(self, verified_models: List[str], throw_errors: bool) -> None:
        """
        Publish global service statistics.
        
        Args:
            verified_models: List of verified model IDs.
            throw_errors: Whether to raise exceptions or just log them.
        """
        try:
            self.publisher.gauge(
                model_name="",
                id=PrometheusPublisher.generate_uuid("model_count"),
                metric_name="MODEL_COUNT_TOTAL",
                value=len(verified_models),
            )
        except ValueError as e:
            self._handle_error(
                e, 
                "Failed to publish model count metric (MODEL_COUNT_TOTAL)", 
                throw_errors
            )

    async def _process_model(self, model_id: str, throw_errors: bool) -> None:
        """
        Process metrics for a single model.
        
        Args:
            model_id: The ID of the model to process.
            throw_errors: Whether to raise exceptions or just log them.
        """
        try:
            await self._publish_model_statistics(model_id, throw_errors)

            if not await self._should_process_model(model_id, throw_errors):
                return
            
            if self.has_requests() and model_id in self.get_model_ids():
                await self._process_model_requests(model_id, throw_errors)
                
        except Exception as e:
            self._handle_error(e, f"Unexpected error processing model={model_id}", throw_errors)
    
    async def _should_process_model(self, model_id: str, throw_errors: bool) -> bool:
        """
        Check if a model has recorded inferences and should be processed.
        
        Args:
            model_id: The ID of the model to check.
            throw_errors: Whether to raise exceptions or just log them.
            
        Returns:
            True if model should be processed, False otherwise.
        """
        try:
            has_recorded_inferences = await self.data_source.has_recorded_inferences(model_id)
            
            if not has_recorded_inferences:
                self._log_skipped_model(model_id)
                return False
                
            return True
            
        except (StorageReadException, MissingH5PYDataException) as e:
            self._handle_error(e, f"Failed to check recorded inferences for model={model_id}", throw_errors)
            return False
        except (OSError, IOError) as e:
            self._handle_error(e, f"File system error checking inferences for model={model_id}", throw_errors)
            return False
    
    def _log_skipped_model(self, model_id: str) -> None:
        """Log when a model is skipped due to no inference data."""
        with self._logged_skipped_request_message_lock:
            if model_id not in self.has_logged_skipped_request_message:
                logger.info(
                    f"Skipping metric calculation for model={model_id}, "
                    "as no inference data has yet been recorded. "
                    "Once inference data arrives, metric calculation will resume."
                )
                self.has_logged_skipped_request_message.add(model_id)
    
    async def _publish_model_statistics(self, model_id: str, throw_errors: bool) -> None:
        """
        Publish basic statistics for a model.
        
        Args:
            model_id: The ID of the model.
            throw_errors: Whether to raise exceptions or just log them.
        """
        total_observations = await self._get_observation_count(model_id, throw_errors)

        if total_observations is None:
            return
        
        try:
            self.publisher.gauge(
                model_name=model_id,
                id=PrometheusPublisher.generate_uuid(model_id),
                metric_name="MODEL_OBSERVATIONS_TOTAL",
                value=total_observations,
            )
        except ValueError as e:
            self._handle_error(
                e, 
                f"Failed to publish observation count metric (MODEL_OBSERVATIONS_TOTAL) for model={model_id}", 
                throw_errors
            )
    
    async def _get_observation_count(self, model_id: str, throw_errors: bool) -> Optional[int]:
        """
        Get the number of observations for a model.
        
        Args:
            model_id: The ID of the model.
            throw_errors: Whether to raise exceptions or just log them.
            
        Returns:
            Number of observations, or None if error occurred.
        """
        try:
            return await self.data_source.get_num_observations(model_id)
        except (StorageReadException, MissingH5PYDataException) as e:
            self._handle_error(e, f"Failed to get observation count for model={model_id}", throw_errors)
            return None
        except KeyError as e:
            self._handle_error(e, f"Missing metadata/schema for model={model_id}", throw_errors)
            return None
        except (OSError, IOError) as e:
            self._handle_error(e, f"File system error for model={model_id}", throw_errors)
            return None
    
    async def _process_model_requests(self, model_id: str, throw_errors: bool) -> None:
        """
        Process all requests for a specific model.
        
        Args:
            model_id: The ID of the model.
            throw_errors: Whether to raise exceptions or just log them.
        """
        try:
            requests_for_model = self._get_requests_for_model(model_id)
            if not requests_for_model:
                logger.warning(f"No requests found for model={model_id}, skipping calculations")
                return
            
            max_batch_size = self._calculate_max_batch_size(requests_for_model)
            df = await self._get_model_dataframe(model_id, max_batch_size, throw_errors)
            if df is None or df.empty:
                return
            
            for req_id, request in requests_for_model:
                await self._process_single_request(model_id, req_id, request, df, throw_errors)
                
        except Exception as e:
            self._handle_error(e, f"Unexpected error processing requests for model={model_id}", throw_errors)
    
    def _get_requests_for_model(self, model_id: str) -> List[Tuple[uuid.UUID, BaseMetricRequest]]:
        """Get all requests for a specific model."""
        return [
            (req_id, request)
            for req_id, request in self.get_all_requests_flat().items()
            if request.model_id == model_id
        ]
    def _calculate_max_batch_size(self, requests: List[Tuple[uuid.UUID, BaseMetricRequest]]) -> int:
        """Calculate the maximum batch size needed for all requests."""
        return max(
            [request.batch_size for _, request in requests],
            default=self.service_config.get("batch_size", 100),
        )

    async def _get_model_dataframe(self, model_id: str, batch_size: int, throw_errors: bool) -> Optional[DataFrame]:
        """
        Get the dataframe for a model.
        
        Args:
            model_id: The ID of the model.
            batch_size: Maximum batch size needed.
            throw_errors: Whether to raise exceptions or just log them.
            
        Returns:
            DataFrame or None if error occurred.
        """
        try:
            df = await self.data_source.get_organic_dataframe(model_id, batch_size)
            if df.empty:
                logger.warning(f"Empty dataframe returned for model={model_id}, skipping calculations")
                return None
            return df
            
        except DataframeCreateException as e:
            self._handle_error(e, f"Failed to create dataframe for model={model_id}", throw_errors)
            return None
        except DataError as e:
            self._handle_error(e, f"Pandas data error for model={model_id}", throw_errors)
            return None
        except (MissingH5PYDataException, OSError, IOError) as e:
            self._handle_error(e, f"Storage error creating dataframe for model={model_id}", throw_errors)
            return None
    
    async def _process_single_request(
        self, 
        model_id: str, 
        req_id: uuid.UUID, 
        request: BaseMetricRequest, 
        df: DataFrame, 
        throw_errors: bool
    ) -> None:
        """
        Process a single metric request.
        
        Args:
            model_id: The ID of the model.
            req_id: The request ID.
            request: The request object.
            df: The dataframe containing model data.
            throw_errors: Whether to raise exceptions or just log them.
        """
        try:
            batch_size = min(request.batch_size, df.shape[0])
            batch = df.tail(batch_size)
            
            metric_name = request.metric_name
            value = self._calculate_metric(model_id, metric_name, batch, request, throw_errors)
            if value is None:
                return
            
            self._publish_metric_value(model_id, req_id, request, value, throw_errors)
            
        except AttributeError as e:
            self._handle_error(e, f"Invalid request object for request {req_id} on model={model_id}", throw_errors)
    
    def _calculate_metric(
        self, 
        model_id: str, 
        metric_name: str, 
        batch: DataFrame, 
        request: BaseMetricRequest, 
        throw_errors: bool
    ) -> Optional[MetricValueCarrier]:
        """
        Calculate a metric value.
        
        Args:
            model_id: The ID of the model.
            metric_name: Name of the metric to calculate.
            batch: Data batch to calculate metric on.
            request: The request object.
            throw_errors: Whether to raise exceptions or just log them.
            
        Returns:
            Calculated metric value or None if error occurred.
        """
        calculator = self.metrics_directory.get_calculator(metric_name)
        if not calculator:
            logger.warning(f"No calculator found for metric {metric_name}")
            return None
        
        try:
            return calculator(batch, request)
            
        except KeyError as e:
            self._handle_error(e, f"Missing required column for metric {metric_name} on model={model_id}", throw_errors)
            return None
        except ValueError as e:
            self._handle_error(e, f"Invalid data values for metric {metric_name} on model={model_id}", throw_errors)
            return None
        except ZeroDivisionError as e:
            self._handle_error(e, f"Division by zero in metric {metric_name} on model={model_id}", throw_errors)
            return None
        except Exception as e:
            self._handle_error(e, f"Unexpected error in metric calculation for {metric_name} on model={model_id}", throw_errors)
            return None
  
    def _publish_metric_value(
        self, 
        model_id: str, 
        req_id: uuid.UUID, 
        request: BaseMetricRequest, 
        value: MetricValueCarrier, 
        throw_errors: bool
    ) -> None:
        """
        Publish a calculated metric value.
        
        Args:
            model_id: The ID of the model.
            req_id: The request ID.
            request: The request object.
            value: The calculated metric value.
            throw_errors: Whether to raise exceptions or just log them.
        """
        try:
            if value.is_single():
                self.publisher.gauge(
                    model_name=model_id,
                    id=req_id,
                    request=request,
                    value=value.get_value(),
                )
            else:
                self.publisher.gauge(
                    model_name=model_id,
                    id=req_id,
                    request=request,
                    named_values=value.get_named_values(),
                )
        except ValueError as e:
            self._handle_error(
                e, 
                f"Failed to publish metric {request.metric_name} for model={model_id}", 
                throw_errors
            )