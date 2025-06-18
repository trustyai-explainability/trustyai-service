import logging
import numbers
import pickle
from typing import Any, List

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.service.data.storage import get_storage_interface

logger = logging.getLogger(__name__)


class RowMatcher(BaseModel):
    """Represents a row matching condition for data filtering."""

    columnName: str
    operation: str  # "EQUALS" or "BETWEEN"
    values: List[Any]


class DataRequestPayload(BaseModel):
    """Request payload for data download operations."""

    modelId: str
    matchAny: List[RowMatcher] = Field(default_factory=list)
    matchAll: List[RowMatcher] = Field(default_factory=list)
    matchNone: List[RowMatcher] = Field(default_factory=list)


class DataResponsePayload(BaseModel):
    """Response payload containing filtered data as CSV."""

    dataCSV: str


def get_storage() -> Any:
    """Get storage interface instance."""
    return get_storage_interface()


async def load_model_dataframe(model_id: str) -> pd.DataFrame:
    try:
        storage = get_storage_interface()
        print(f"DEBUG: storage type = {type(storage)}")
        input_data, input_cols = await storage.read_data(f"{model_id}_inputs")
        output_data, output_cols = await storage.read_data(f"{model_id}_outputs")
        metadata_data, metadata_cols = await storage.read_data(f"{model_id}_metadata")
        if input_data is None or output_data is None or metadata_data is None:
            raise HTTPException(404, f"Model {model_id} not found")
        df = pd.DataFrame()
        if len(input_data) > 0:
            input_df = pd.DataFrame(input_data, columns=input_cols)
            df = pd.concat([df, input_df], axis=1)
        if len(output_data) > 0:
            output_df = pd.DataFrame(output_data, columns=output_cols)
            df = pd.concat([df, output_df], axis=1)
        if len(metadata_data) > 0:
            logger.debug(f"Metadata data type: {type(metadata_data)}")
            logger.debug(f"First row type: {type(metadata_data[0]) if len(metadata_data) > 0 else 'empty'}")
            logger.debug(
                f"First row dtype: {metadata_data[0].dtype if hasattr(metadata_data[0], 'dtype') else 'no dtype'}"
            )
            metadata_df = pd.DataFrame(metadata_data, columns=metadata_cols)
            trusty_mapping = {
                "ID": "trustyai.ID",
                "MODEL_ID": "trustyai.MODEL_ID",
                "TIMESTAMP": "trustyai.TIMESTAMP",
                "TAG": "trustyai.TAG",
                "INDEX": "trustyai.INDEX",
            }
            for orig_col in metadata_cols:
                trusty_col = trusty_mapping.get(orig_col, orig_col)
                df[trusty_col] = metadata_df[orig_col]
        return df
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model dataframe: {e}")
        raise HTTPException(500, f"Error loading model data: {str(e)}")


def apply_filters(df: pd.DataFrame, payload: DataRequestPayload) -> pd.DataFrame:
    """
    Apply all filters to DataFrame with performance optimization.
    """
    if not any([payload.matchAll, payload.matchAny, payload.matchNone]):
        return df
    has_timestamp_filter = _has_timestamp_filters(payload)
    if has_timestamp_filter:
        logger.debug("Using boolean mask approach for timestamp filters")
        return _apply_filters_with_boolean_masks(df, payload)
    else:
        logger.debug("Using query approach for non-timestamp filters")
        return _apply_filters_with_query(df, payload)


def _has_timestamp_filters(payload: DataRequestPayload) -> bool:
    """Check if payload contains any timestamp filters."""
    for matcher_list in [payload.matchAll or [], payload.matchAny or [], payload.matchNone or []]:
        for matcher in matcher_list:
            if matcher.columnName == "trustyai.TIMESTAMP":
                return True
    return False


def _apply_filters_with_query(df: pd.DataFrame, payload: DataRequestPayload) -> pd.DataFrame:
    """Apply filters using pandas query (optimized for non-timestamp filters)."""
    query_expr = _build_query_expression(df, payload)
    if query_expr:
        logger.debug(f"Executing query: {query_expr}")
        try:
            df = df.query(query_expr)
        except Exception as e:
            logger.error(f"Query execution failed: {query_expr}")
            raise HTTPException(status_code=400, detail=f"Filter execution failed: {str(e)}")
    return df


def _apply_filters_with_boolean_masks(df: pd.DataFrame, payload: DataRequestPayload) -> pd.DataFrame:
    """Apply filters using boolean masks (optimized for timestamp filters)."""
    final_mask = pd.Series(True, index=df.index)
    if payload.matchAll:
        for matcher in payload.matchAll:
            matcher_mask = _get_matcher_mask(df, matcher, negate=False)
            final_mask &= matcher_mask
    if payload.matchNone:
        for matcher in payload.matchNone:
            matcher_mask = _get_matcher_mask(df, matcher, negate=True)
            final_mask &= matcher_mask
    if payload.matchAny:
        any_mask = pd.Series(False, index=df.index)
        for matcher in payload.matchAny:
            matcher_mask = _get_matcher_mask(df, matcher, negate=False)
            any_mask |= matcher_mask
        final_mask &= any_mask
    return df[final_mask]


def _get_matcher_mask(df: pd.DataFrame, matcher: RowMatcher, negate: bool = False) -> pd.Series:
    """
    Get boolean mask for a single matcher with comprehensive validation.
    """
    column_name = matcher.columnName
    values = matcher.values
    if matcher.operation not in ["EQUALS", "BETWEEN"]:
        raise HTTPException(status_code=400, detail="RowMatch operation must be one of [BETWEEN, EQUALS]")
    if column_name not in df.columns:
        raise HTTPException(status_code=400, detail=f"No feature or output found with name={column_name}")
    if matcher.operation == "EQUALS":
        mask = df[column_name].isin(values)
    elif matcher.operation == "BETWEEN":
        mask = _create_between_mask(df, column_name, values)
    if negate:
        mask = ~mask

    return mask


def _create_between_mask(df: pd.DataFrame, column_name: str, values: List[Any]) -> pd.Series:
    """Create boolean mask for BETWEEN operation with type-specific handling."""
    errors = []
    if len(values) != 2:
        errors.append(
            f"BETWEEN operation must contain exactly two values, describing the lower and upper bounds of the desired range. Received {len(values)} values"
        )
    if column_name == "trustyai.TIMESTAMP":
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        try:
            start_time = pd.to_datetime(str(values[0]))
            end_time = pd.to_datetime(str(values[1]))
            df_times = pd.to_datetime(df[column_name])
            return (df_times >= start_time) & (df_times < end_time)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Timestamp value is unparseable as an ISO_LOCAL_DATE_TIME: {str(e)}"
            )
    elif column_name == "trustyai.INDEX":
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        min_val, max_val = sorted([int(v) for v in values])
        return (df[column_name] >= min_val) & (df[column_name] < max_val)
    else:
        if not all(isinstance(v, numbers.Number) for v in values):
            errors.append(
                "BETWEEN operation must only contain numbers, describing the lower and upper bounds of the desired range. Received non-numeric values"
            )
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        min_val, max_val = sorted(values)
        try:
            if df[column_name].dtype in ["int64", "float64", "int32", "float32"]:
                return (df[column_name] >= min_val) & (df[column_name] < max_val)
            else:
                numeric_column = pd.to_numeric(df[column_name], errors="raise")
                return (numeric_column >= min_val) & (numeric_column < max_val)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' contains non-numeric values that cannot be compared with BETWEEN operation.",
            )


def _build_query_expression(df: pd.DataFrame, payload: DataRequestPayload) -> str:
    """Build optimized pandas query expression for all filters."""
    conditions = []
    if payload.matchAll:
        for matcher in payload.matchAll:
            condition = _build_condition(df, matcher, negate=False)
            if condition:
                conditions.append(condition)
    if payload.matchNone:
        for matcher in payload.matchNone:
            condition = _build_condition(df, matcher, negate=True)
            if condition:
                conditions.append(condition)
    if payload.matchAny:
        any_conditions = []
        for matcher in payload.matchAny:
            condition = _build_condition(df, matcher, negate=False)
            if condition:
                any_conditions.append(condition)
        if any_conditions:
            any_expr = " | ".join(f"({cond})" for cond in any_conditions)
            conditions.append(f"({any_expr})")
    return " & ".join(f"({cond})" for cond in conditions) if conditions else ""


def _build_condition(df: pd.DataFrame, matcher: RowMatcher, negate: bool = False) -> str:
    """Build a single condition for pandas query."""
    column_name = matcher.columnName
    values = matcher.values
    if matcher.operation not in ["EQUALS", "BETWEEN"]:
        raise HTTPException(status_code=400, detail="RowMatch operation must be one of [BETWEEN, EQUALS]")
    if column_name not in df.columns:
        raise HTTPException(status_code=400, detail=f"No feature or output found with name={column_name}")
    safe_column = _sanitize_column_name(column_name)
    if matcher.operation == "EQUALS":
        condition = _build_equals_condition(safe_column, values, df[column_name].dtype)
    elif matcher.operation == "BETWEEN":
        condition = _build_between_condition(safe_column, values, column_name, df[column_name].dtype)
    if negate:
        condition = f"~({condition})"
    return condition


def _sanitize_column_name(column_name: str) -> str:
    """Sanitize column name for pandas query syntax."""
    if "." in column_name or column_name.startswith("trustyai"):
        return f"`{column_name}`"
    return column_name


def _build_equals_condition(safe_column: str, values: List[Any], dtype) -> str:
    """Build EQUALS condition for query with optimization."""
    if len(values) == 1:
        val = _format_value_for_query(values[0], dtype)
        return f"{safe_column} == {val}"
    else:
        formatted_values = [_format_value_for_query(v, dtype) for v in values]
        values_str = "[" + ", ".join(formatted_values) + "]"
        return f"{safe_column}.isin({values_str})"


def _build_between_condition(safe_column: str, values: List[Any], original_column: str, dtype) -> str:
    """Build BETWEEN condition for query with comprehensive validation."""
    errors = []
    if len(values) != 2:
        errors.append(
            f"BETWEEN operation must contain exactly two values, describing the lower and upper bounds of the desired range. Received {len(values)} values"
        )
    if original_column == "trustyai.TIMESTAMP":
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        try:
            start_time = pd.to_datetime(str(values[0]))
            end_time = pd.to_datetime(str(values[1]))
            return f"'{start_time}' <= {safe_column} < '{end_time}'"
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Timestamp value is unparseable as an ISO_LOCAL_DATE_TIME: {str(e)}"
            )
    elif original_column == "trustyai.INDEX":
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        min_val, max_val = sorted([int(v) for v in values])
        return f"{min_val} <= {safe_column} < {max_val}"
    else:
        if not all(isinstance(v, numbers.Number) for v in values):
            errors.append(
                "BETWEEN operation must only contain numbers, describing the lower and upper bounds of the desired range. Received non-numeric values"
            )
        if errors:
            raise HTTPException(status_code=400, detail=", ".join(errors))
        min_val, max_val = sorted(values)
        return f"{min_val} <= {safe_column} < {max_val}"


def _format_value_for_query(value: Any, dtype) -> str:
    """Format value appropriately for pandas query syntax."""
    if isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        escaped = str(value).replace("'", "\\'")
        return f"'{escaped}'"