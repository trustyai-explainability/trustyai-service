from fastapi import APIRouter, HTTPException, BackgroundTasks
from prometheus_client import Gauge
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging
import uuid
import pandas as pd

from src.core.metrics.fairness.group.group_statistical_parity_difference import GroupStatisticalParityDifference
from src.service.prometheus.metric_value_carrier import MetricValueCarrier
from src.service.data.datasources.data_source import DataSource
from src.service.data.shared_data_source import get_shared_data_source
from src.service.prometheus.shared_prometheus_scheduler import get_shared_prometheus_scheduler
from src.service.payloads.metrics.base_metric_request import BaseMetricRequest

router = APIRouter()
logger = logging.getLogger(__name__)

def get_prometheus_scheduler():
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()

def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class ReconcilableFeature(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class ReconcilableOutput(BaseModel):
    rawValueNodes: Optional[List[Dict[str, Any]]] = None
    rawValueNode: Optional[Dict[str, Any]] = None
    reconciledType: Optional[List[Dict[str, Any]]] = None
    multipleValued: Optional[bool] = None


class GroupMetricRequest(BaseMetricRequest):
    # Use field aliases to accept camelCase from API while keeping snake_case internally
    model_id: str = Field(alias="modelId")
    metric_name: Optional[str] = Field(default=None, alias="metricName")  # Will be set by endpoint
    request_name: Optional[str] = Field(default=None, alias="requestName")
    batch_size: Optional[int] = Field(default=100, alias="batchSize")

    # SPD-specific fields
    protected_attribute: str = Field(alias="protectedAttribute")
    outcome_name: str = Field(alias="outcomeName")
    privileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="privilegedAttribute")
    unprivileged_attribute: Union[ReconcilableFeature, int, float, str] = Field(alias="unprivilegedAttribute")
    favorable_outcome: Union[ReconcilableOutput, int, float, str] = Field(alias="favorableOutcome")
    threshold_delta: Optional[float] = Field(default=None, alias="thresholdDelta")

    def retrieve_tags(self) -> Dict[str, str]:
        """Retrieve tags for this SPD metric request."""
        tags = self.retrieve_default_tags()
        tags["protectedAttribute"] = self.protected_attribute
        tags["outcomeName"] = self.outcome_name
        return tags



class GroupDefinitionRequest(BaseModel):
    modelId: str
    requestName: Optional[str] = None
    metricName: Optional[str] = None
    batchSize: Optional[int] = 100
    protectedAttribute: str
    outcomeName: str
    privilegedAttribute: Union[ReconcilableFeature, int, float, str]
    unprivilegedAttribute: Union[ReconcilableFeature, int, float, str]
    favorableOutcome: Union[ReconcilableOutput, int, float, str]
    thresholdDelta: Optional[float] = None
    metricValue: Dict[str, Any]

    # Additional snake_case properties for metrics processing
    @property
    def protected_attribute(self) -> str:
        return self.protectedAttribute

    @property
    def outcome_name(self) -> str:
        return self.outcomeName

    @property
    def privileged_attribute(self) -> Union[ReconcilableFeature, int, float, str]:
        return self.privilegedAttribute

    @property
    def unprivileged_attribute(self) -> Union[ReconcilableFeature, int, float, str]:
        return self.unprivilegedAttribute

    @property
    def favorable_outcome(self) -> Union[ReconcilableOutput, int, float, str]:
        return self.favorableOutcome

    @property
    def threshold_delta(self) -> Optional[float]:
        return self.thresholdDelta


class ScheduleId(BaseModel):
    requestId: str


# Note: SPDRequest class removed - using GroupMetricRequest for consistency with Java API


def calculate_spd_metric(dataframe: pd.DataFrame, request) -> MetricValueCarrier:
    """
    Calculate SPD metric for the given dataframe and request.
    This function is registered with the metrics directory and called by the scheduler.
    """
    try:
        # Extract data from the reconciled request
        model_id = request.modelId if hasattr(request, 'modelId') else request.model_id
        protected_attr = request.protectedAttribute if hasattr(request, 'protectedAttribute') else request.protected_attribute
        outcome_name = request.outcomeName if hasattr(request, 'outcomeName') else request.outcome_name

        # Handle different types of privilege/unprivilege attributes
        if hasattr(request, 'privilegedAttribute') and hasattr(request.privilegedAttribute, 'reconciledType'):
            # Complex reconciled type
            privileged_values = [item.get('value') for item in request.privilegedAttribute.reconciledType if 'value' in item]
        else:
            # Simple value
            privileged_attr = getattr(request, 'privilegedAttribute', getattr(request, 'privileged_attribute', None))
            privileged_values = [privileged_attr] if not isinstance(privileged_attr, list) else privileged_attr

        if hasattr(request, 'unprivilegedAttribute') and hasattr(request.unprivilegedAttribute, 'reconciledType'):
            unprivileged_values = [item.get('value') for item in request.unprivilegedAttribute.reconciledType if 'value' in item]
        else:
            unprivileged_attr = getattr(request, 'unprivilegedAttribute', getattr(request, 'unprivileged_attribute', None))
            unprivileged_values = [unprivileged_attr] if not isinstance(unprivileged_attr, list) else unprivileged_attr

        if hasattr(request, 'favorableOutcome') and hasattr(request.favorableOutcome, 'reconciledType'):
            favorable_values = [item.get('value') for item in request.favorableOutcome.reconciledType if 'value' in item]
        else:
            favorable_attr = getattr(request, 'favorableOutcome', getattr(request, 'favorable_outcome', None))
            favorable_values = [favorable_attr] if not isinstance(favorable_attr, list) else favorable_attr

        # Filter the dataframe into privileged and unprivileged groups
        privileged_mask = dataframe[protected_attr].isin(privileged_values)
        unprivileged_mask = dataframe[protected_attr].isin(unprivileged_values)

        privileged_data = dataframe[privileged_mask]
        unprivileged_data = dataframe[unprivileged_mask]

        if len(privileged_data) == 0 or len(unprivileged_data) == 0:
            logger.warning(f"Insufficient data for SPD calculation: privileged={len(privileged_data)}, unprivileged={len(unprivileged_data)} samples. Returning NaN.")
            return MetricValueCarrier(float('nan'))

        # Calculate favorable outcome rates
        priv_favorable_count = len(privileged_data[privileged_data[outcome_name].isin(favorable_values)])
        unpriv_favorable_count = len(unprivileged_data[unprivileged_data[outcome_name].isin(favorable_values)])

        priv_favorable_rate = priv_favorable_count / len(privileged_data)
        unpriv_favorable_rate = unpriv_favorable_count / len(unprivileged_data)

        # SPD = P(Y=favorable|A=unprivileged) - P(Y=favorable|A=privileged)
        spd_value = unpriv_favorable_rate - priv_favorable_rate

        logger.debug(f"SPD calculation: privileged_rate={priv_favorable_rate:.4f}, unprivileged_rate={unpriv_favorable_rate:.4f}, spd={spd_value:.4f}")

        return MetricValueCarrier(spd_value)

    except Exception as e:
        logger.error(f"Error calculating SPD: {str(e)}")
        raise e

# Register the SPD calculator with the metrics directory
def register_spd_calculator():
    """Register the SPD calculator with the global metrics directory."""
    scheduler = get_prometheus_scheduler()
    if scheduler and scheduler.metrics_directory:
        scheduler.metrics_directory.register("SPD", calculate_spd_metric)
        logger.info("SPD calculator registered with metrics directory")

# Register on module import
try:
    register_spd_calculator()
except Exception as e:
    logger.warning(f"Could not register SPD calculator on import: {e}")


# Statistical Parity Difference
@router.post("/metrics/group/fairness/spd")
async def compute_spd(request: GroupMetricRequest):
    """Compute the current value of Statistical Parity Difference metric."""
    try:
        logger.info(f"Computing SPD for model: {request.model_id}")

        # Get data source and load dataframe
        data_source = get_data_source()
        batch_size = request.batch_size if request.batch_size else 100

        # Get dataframe for the model
        dataframe = await data_source.get_organic_dataframe(request.model_id, batch_size)

        if dataframe.empty:
            raise HTTPException(status_code=404, detail=f"No data found for model: {request.model_id}")

        # Calculate SPD using our calculator
        result = calculate_spd_metric(dataframe, request)

        return {
            "name": "SPD",
            "value": result.get_value(),
            "type": "FAIRNESS",
            "specificDefinition": f"Statistical Parity Difference value of {result.get_value():.4f}",
            "thresholds": {
                "lowerBound": -0.1,
                "upperBound": 0.1,
                "outsideBounds": abs(result.get_value()) > 0.1
            }
        }
    except Exception as e:
        logger.error(f"Error computing SPD: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error computing metric: {str(e)}"
        ) from e


@router.get("/metrics/group/fairness/spd/definition")
async def get_spd_definition():
    """Provide a general definition of Statistical Parity Difference metric."""
    return {
        "name": "Statistical Parity Difference",
        "description": "Description.",
    }


@router.post("/metrics/group/fairness/spd/definition")
async def interpret_spd_value(request: GroupDefinitionRequest):
    """Provide a specific, plain-english interpretation of a specific value of SPD metric."""
    try:
        logger.info(f"Interpreting SPD value for model: {request.modelId}")
        # TODO: Implement
        return {"interpretation": "The SPD value indicates a small bias in the model."}
    except Exception as e:
        logger.error(f"Error interpreting SPD value: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interpreting value: {str(e)}")


@router.post("/metrics/group/fairness/spd/request")
async def schedule_spd(request: GroupMetricRequest):
    """Schedule a recurring computation of SPD metric."""
    try:
        # Generate UUID for this request
        request_id = uuid.uuid4()
        logger.info(f"Scheduling SPD computation with ID: {request_id}")

        # Set metric name automatically
        request.metric_name = "SPD"

        # Get the scheduler and register the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Register with the scheduler (this will reconcile the request and store it)
        await scheduler.register("SPD", request_id, request)

        logger.info(f"Successfully scheduled SPD computation with ID: {request_id}")
        return {"requestId": str(request_id)}

    except Exception as e:
        logger.error(f"Error scheduling SPD computation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error scheduling metric: {str(e)}"
        ) from e


@router.delete("/metrics/group/fairness/spd/request")
async def delete_spd_schedule(schedule: ScheduleId):
    """Delete a recurring computation of SPD metric."""
    try:
        logger.info(f"Deleting SPD schedule: {schedule.requestId}")

        # Get the scheduler and delete the request
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Convert string ID to UUID
        try:
            request_uuid = uuid.UUID(schedule.requestId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request ID format")

        # Delete from scheduler
        await scheduler.delete("SPD", request_uuid)

        logger.info(f"Successfully deleted SPD schedule: {schedule.requestId}")
        return {"status": "success", "message": f"Schedule {schedule.requestId} deleted"}

    except Exception as e:
        logger.error(f"Error deleting SPD schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting schedule: {str(e)}")


@router.get("/metrics/group/fairness/spd/requests")
async def list_spd_requests():
    """List the currently scheduled computations of SPD metric."""
    try:
        # Get the scheduler and list SPD requests
        scheduler = get_prometheus_scheduler()
        if not scheduler:
            raise HTTPException(status_code=500, detail="Prometheus scheduler not available")

        # Get all requests for SPD
        spd_requests = scheduler.get_requests("SPD")

        # Convert to list format expected by client
        requests_list = []
        for request_id, request in spd_requests.items():
            requests_list.append({
                "requestId": str(request_id),
                "modelId": request.model_id,
                "metricName": "SPD",
                "batchSize": request.batch_size,
                "protectedAttribute": request.protected_attribute,
                "outcomeName": request.outcome_name
            })

        return {"requests": requests_list}

    except Exception as e:
        logger.error(f"Error listing SPD requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing requests: {str(e)}")


# Deprecated SPD endpoints
@router.post("/spd", deprecated=True)
async def compute_spd_deprecated(request: GroupMetricRequest):
    """Compute the current value of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd instead.
    """
    return await compute_spd(request)


@router.get("/spd/definition", deprecated=True)
async def get_spd_definition_deprecated():
    """Provide a general definition of Statistical Parity Difference metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/definition instead.
    """
    return await get_spd_definition()


@router.post("/spd/definition", deprecated=True)
async def interpret_spd_value_deprecated(request: GroupDefinitionRequest):
    """Provide a specific interpretation of a SPD metric value (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/definition instead.
    """
    return await interpret_spd_value(request)


@router.post("/spd/request", deprecated=True)
async def schedule_spd_deprecated(request: GroupMetricRequest):
    """Schedule a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/request instead.
    """
    return await schedule_spd(request)


@router.delete("/spd/request", deprecated=True)
async def delete_spd_schedule_deprecated(schedule: ScheduleId):
    """Delete a recurring computation of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/request instead.
    """
    return await delete_spd_schedule(schedule)


@router.get("/spd/requests", deprecated=True)
async def list_spd_requests_deprecated():
    """List the currently scheduled computations of SPD metric (deprecated).

    This endpoint is deprecated. Please use /metrics/group/fairness/spd/requests instead.
    """
    return await list_spd_requests()
