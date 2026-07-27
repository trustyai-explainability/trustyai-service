"""LM Evaluation Harness endpoint for language model benchmarking and testing."""

import contextlib
import datetime
import os
import queue
import shlex
import sys
import threading
from contextlib import asynccontextmanager
from enum import Enum

try:
    from fastapi_utils.tasks import repeat_every
    from lm_eval.__main__ import setup_parser as lm_eval_setup_parser
except ImportError as e:
    msg = "The TrustyAI service was not built with LM-Evaluation-Harness support, use `pip install .[eval]`"
    raise ImportError(msg) from e

import logging
import subprocess  # nosec B404 - Used with shlex.split() for safe argument handling
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, TextIO

from fastapi import APIRouter, HTTPException
from fastapi.applications import FastAPI
from pydantic import BaseModel, create_model, model_validator

from trustyai_service.endpoints.evaluation._env_security import (
    build_subprocess_env,
    validate_env_vars_model,
)

logger = logging.getLogger(__name__)

API_PREFIX = "/eval/lm-evaluation-harness"


# === STATIC API OBJECTS ===========================================================================
class JobStatus(Enum):
    """Status of an LM evaluation job."""

    RUNNING = "Running"
    FAILED = "Failed"
    COMPLETED = "Completed"
    QUEUED = "Queued"
    STOPPED = "Stopped"


class LMEvalJobSummary(BaseModel):
    """Summary information for an LM evaluation job."""

    job_id: int
    argument: str
    status: JobStatus
    timestamp: str | None
    exit_code: int | None
    inference_progress_pct: int


class LMEvalJobDetail(LMEvalJobSummary):
    """Detailed information for an LM evaluation job including output streams."""

    stdout: list[str]
    stderr: list[str]


class AllLMEvalJobs(BaseModel):
    """Container for a list of LM evaluation job summaries."""

    jobs: list[LMEvalJobSummary]


# === Dynamic API Object from LM-Eval CLI ==========================================================
# Executable path is server-side only — never user-controllable (CWE-78 mitigation).
# Override via LM_EVAL_PATH env var at deployment time if needed.
LM_EVAL_EXECUTABLE = os.environ.get("LM_EVAL_PATH", f"{sys.executable} -m lm_eval")

NON_CLI_ARGUMENTS = {
    "env_vars": (dict[str, str], {}),
}


def get_lm_eval_arguments() -> dict[str, dict[str, Any]]:
    """Grab all fields from an argparse specification into a dictionary."""
    parser = lm_eval_setup_parser()  # grab lm-eval argparse specification

    args = {}
    # Access private argparse internals for dynamic CLI generation
    for action in parser._positionals._actions:
        arg = {
            "cli": action.option_strings[0],
            "argparse_type": action.__class__.__name__,
        }
        if action.__class__.__name__ == "_StoreTrueAction":
            arg["type"] = bool
            arg["default"] = False
        elif action.__class__.__name__ == "_StoreFalseAction":
            arg["type"] = bool
            arg["default"] = True
        elif action.__class__.__name__ == "_HelpAction":
            continue
        else:
            arg["default"] = action.default
            arg["type"] = str if action.type == str.upper else action.type
        args[action.dest] = arg
    return args


def get_model() -> type[BaseModel]:
    """Build a Pydantic model from the lm-eval argparse arguments, adding in a few config variables of our own as well."""
    args = get_lm_eval_arguments()
    model_args = {k: (v["type"], v["default"]) for k, v in args.items()}
    model_args.update(NON_CLI_ARGUMENTS)
    return create_model(
        "LMEvalRequest",
        __validators__={
            "_check_env_vars": model_validator(mode="after")(validate_env_vars_model),
        },
        **model_args,
    )


# Dynamically create the lm-eval-harness job request from the library's argparse
LMEvalRequest = get_model()


# === Registry of running lm-eval-harness jobs =====================================================
class LMEvalJob:
    """Manages the lifecycle and state of an LM evaluation job."""

    def __init__(self, job_id: int, request: BaseModel, argument: str) -> None:
        """Initialize an LM evaluation job.

        :param job_id: Unique identifier for the job
        :param request: The LMEvalRequest containing job parameters
        :param argument: CLI argument string for lm-eval command
        """
        self.job_id = job_id
        self.process = None
        self.request = request
        self.argument = argument
        self.start_time = None
        self.cumulative_out = []
        self.cumulative_err = []
        self.progress = 0
        self.is_in_queue = True
        self.has_been_stopped = False
        self.output_lock = threading.Lock()  # Protects cumulative_out/err/progress
        self.stdout_thread = None
        self.stderr_thread = None

    def _read_stream(
        self, stream: TextIO, output_list: list[str], stream_name: str
    ) -> None:
        """Read lines from a stream and append to output list (runs in background thread).

        :param stream: The stream to read from (stdout or stderr)
        :param output_list: The list to append lines to (cumulative_out or cumulative_err)
        :param stream_name: Name for logging purposes
        """
        try:
            for line in stream:
                stripped = line.strip()
                with self.output_lock:
                    output_list.append(stripped)
                    # Update progress from stderr
                    if stream_name == "stderr" and stripped.startswith(
                        "Requesting API:"
                    ):
                        with contextlib.suppress(ValueError, IndexError):
                            self.progress = int(
                                stripped.split("Requesting API:")[1]
                                .split("%")[0]
                                .strip()
                            )
        except Exception:  # noqa: BLE001, S110  # nosec B110 - Thread should not crash on stream errors
            pass  # Stream closed or process terminated

    def mark_launch(self, process: subprocess.Popen, start_time: str) -> None:
        """Mark the job as launched with process and timestamp.

        :param process: The subprocess.Popen object running the job
        :param start_time: ISO format timestamp of job launch
        """
        self.start_time = start_time
        self.process = process
        self.is_in_queue = False

        # Start background reader threads for stdout/stderr
        self.stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(process.stdout, self.cumulative_out, "stdout"),
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(process.stderr, self.cumulative_err, "stderr"),
            daemon=True,
        )
        self.stdout_thread.start()
        self.stderr_thread.start()

    def dequeue(self) -> None:
        """Remove the job from the queue and mark it as stopped."""
        self.has_been_stopped = True
        self.is_in_queue = False


job_registry: dict[int, LMEvalJob] = {}  # track all jobs
job_registry_lock = threading.Lock()  # protects job_registry from concurrent access
job_queue = queue.Queue()


class JobIdGenerator:
    """Thread-safe job ID generator."""

    _lock = threading.Lock()
    _last_id: int = 0

    @classmethod
    def generate(cls) -> int:
        """Generate a unique job ID.

        Returns:
            Unique job ID

        """
        with cls._lock:
            cls._last_id += 1
            return cls._last_id

    @classmethod
    def reset(cls) -> None:
        """Reset ID counter (useful for testing)."""
        with cls._lock:
            cls._last_id = 0


# === ENV VARIABLE PARSING =========================================================================
try:
    MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
except ValueError:
    MAX_CONCURRENCY = 4

try:
    QUEUE_PROCESS_INTERVAL = int(os.environ.get("QUEUE_PROCESS_INTERVAL", "15"))
except ValueError:
    QUEUE_PROCESS_INTERVAL = 15


# === HELPERS ======================================================================================
def convert_to_cli(request: BaseModel) -> str:
    """Convert an LMEvalRequest json object into an lm-eval cli argument."""
    args = (
        get_lm_eval_arguments()
    )  # grab the cli argument spec to translate json to cli

    cli_cmd = LM_EVAL_EXECUTABLE
    for field in request.model_fields_set:
        if field in NON_CLI_ARGUMENTS:
            continue

        arg = args[field]
        if arg["argparse_type"] in {"_StoreTrueAction", "_StoreFalseAction"}:
            # Only add flag when actual value matches the action type
            field_value = getattr(request, field)
            if (arg["argparse_type"] == "_StoreTrueAction" and field_value) or (
                arg["argparse_type"] == "_StoreFalseAction" and not field_value
            ):
                cli_cmd += f" {args[field]['cli']}"
        else:
            field_value = getattr(request, field)
            field_value = (
                shlex.quote(field_value)
                if isinstance(field_value, str)
                else field_value
            )
            cli_cmd += f" {args[field]['cli']} {field_value}"

    return cli_cmd


def _get_job_status(job: LMEvalJob) -> tuple[JobStatus, int | None]:
    """Poll a running subprocess for exit codes."""
    if job.is_in_queue:
        status = JobStatus.QUEUED
        status_code = None
    elif job.has_been_stopped:
        status = JobStatus.STOPPED
        status_code = None
    elif job.process is None:
        # Transitional state: dequeued by _process_queue but Popen() has not returned yet
        status = JobStatus.QUEUED
        status_code = None
    else:
        status_code = job.process.poll()
        if status_code == 0:
            status = JobStatus.COMPLETED
        elif status_code is None:
            status = JobStatus.RUNNING
        else:
            status = JobStatus.FAILED

    return status, status_code


def _generate_job_id() -> int:
    """Generate a unique job ID."""
    return JobIdGenerator.generate()


# === Queuing ======================================================================================
def _get_num_running_jobs() -> int:
    """Get the number of currently running jobs (lightweight check, no I/O)."""
    count = 0
    with job_registry_lock:
        for job in job_registry.values():
            # Count jobs that are launched and still running (no I/O draining)
            if (
                not job.is_in_queue
                and job.process is not None
                and job.process.poll() is None
            ):
                count += 1
    return count


@repeat_every(seconds=QUEUE_PROCESS_INTERVAL, logger=logger)
async def _process_queue() -> None:
    """Check the job queue for pending jobs and launch them if there are available execution slots."""
    logger.debug(
        "Queue processing: num_running_jobs: %s, job queue size: %s",
        _get_num_running_jobs(),
        job_queue.qsize(),
    )
    while _get_num_running_jobs() < MAX_CONCURRENCY and job_queue.qsize() > 0:
        job_id = job_queue.get()
        # Check job state atomically while holding lock, extract job to launch
        job_to_launch = None
        with job_registry_lock:
            job_to_run = job_registry.get(job_id)
            # Verify job is still in queue (not dequeued by another thread)
            if job_to_run is not None and job_to_run.is_in_queue:
                job_to_launch = job_to_run
                # Mark as no longer in queue BEFORE launching to prevent race
                # with stop_lm_eval_job() dequeuing while Popen() is running
                job_to_run.is_in_queue = False
            elif job_to_run is None:
                logger.warning("Job %s not found in registry, skipping", job_id)
            else:
                logger.debug("Job %s already dequeued, skipping launch", job_id)

        # Launch job outside the lock (process creation doesn't need lock)
        if job_to_launch is not None:
            logger.info("Launching job %s", job_to_launch.job_id)
            _launch_job(job_to_launch)


def _launch_job(job: LMEvalJob) -> None:
    """Launch a job."""
    logger.debug("Launching lm-eval job %s", job.job_id)
    logger.debug("Launching with custom env vars: %s keys", len(job.request.env_vars))

    env = build_subprocess_env(job.request.env_vars)

    # Executable is a server-side constant (LM_EVAL_EXECUTABLE); CLI args are shlex.quote()'d
    p = subprocess.Popen(  # noqa: S603  # executable hardcoded, args quoted
        shlex.split(job.argument),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )  # nosec B603

    # register the subprocess in the global registry
    with job_registry_lock:
        job_registry[job.job_id].mark_launch(
            process=p, start_time=datetime.datetime.now(datetime.UTC).isoformat()
        )


# === ROUTER =======================================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Launch recurring queue processing function."""
    await _process_queue()
    yield


router = APIRouter(lifespan=lifespan)


# === API ==========================================================================================
@router.post(API_PREFIX + "/job", summary="Launch an lm-evaluation-harness job")
def lm_eval_job(request: LMEvalRequest) -> dict[str, int | str]:
    """Launch an lm-evaluation-harness job according to the inbound arguments.

    These match the CLI arguments to lm-evaluation-harness, just in json form.
    """
    # convert the json to cli arguments
    cli_cmd = convert_to_cli(request)

    # store job
    job_id = _generate_job_id()
    queued_job = LMEvalJob(job_id=job_id, request=request, argument=cli_cmd)

    # Register job before enqueuing to prevent race condition
    with job_registry_lock:
        job_registry[job_id] = queued_job
    job_queue.put(job_id)

    return {
        "status": "success",
        "message": f"Job {job_id} successfully queued.",
        "job_id": job_id,
    }


# === METADATA =====================================================================================
@router.get(API_PREFIX + "/jobs", summary="List all running jobs")
def list_running_lm_eval_jobs(*, include_finished: bool = True) -> AllLMEvalJobs:
    """Provide a list of all lm-evaluation-harness jobs with attached summary information."""
    jobs = []
    with job_registry_lock:
        job_ids = list(job_registry.keys())
    for pid in job_ids:
        job_information = check_lm_eval_job(pid)

        if not include_finished and job_information.status not in {
            JobStatus.RUNNING,
            JobStatus.QUEUED,
        }:
            continue
        jobs.append(
            LMEvalJobSummary(**job_information.model_dump(exclude={"stdout", "stderr"}))
        )

    return AllLMEvalJobs(jobs=jobs)


@router.get(
    API_PREFIX + "/job/{job_id}", summary="Get information about a specific job"
)
def check_lm_eval_job(job_id: int) -> LMEvalJobDetail:
    """Get detailed report of an lm-evaluation-harness job by ID."""
    with job_registry_lock:
        if job_id not in job_registry:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"No lm-evaluation-harness job with ID={job_id} found.",
            )
        job = job_registry[job_id]
    status, status_code = _get_job_status(job)

    # Progress is updated by background reader threads in mark_launch
    # Just read the current progress value under lock
    with job.output_lock:
        progress = job.progress
        cumulative_out = job.cumulative_out.copy()
        cumulative_err = job.cumulative_err.copy()

    return LMEvalJobDetail(
        job_id=job_id,
        argument=job.argument,
        timestamp=job.start_time,
        status=status,
        exit_code=status_code,
        inference_progress_pct=progress,
        stdout=cumulative_out,
        stderr=cumulative_err,
    )


# === DELETE DATA ==================================================================================
@router.delete(
    API_PREFIX + "/job/{job_id}",
    summary="Delete an lm-evaluation-harness job's data from the server.",
)
def delete_lm_eval_job(job_id: int) -> dict[str, str]:
    """Delete an lm-evaluation-harness job's data from the server by ID, terminating the job if it's still running."""
    with job_registry_lock:
        if job_id not in job_registry:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"No lm-evaluation-harness job with ID={job_id} found.",
            )

    stop_lm_eval_job(job_id)
    with job_registry_lock:
        del job_registry[job_id]
    return {"status": "success", "message": f"Job {job_id} deleted successfully."}


@router.delete(
    API_PREFIX + "/jobs",
    summary="Delete data from all lm-evaluation-harness jobs from the server.",
)
def delete_all_lm_eval_job() -> dict[str, str]:
    """Delete data from all lm-evaluation-harness job's data from the server, terminating any job that its still running."""
    deleted = []
    with job_registry_lock:
        job_ids = list(job_registry.keys())
    for job_id in job_ids:
        stop_lm_eval_job(job_id)
        deleted.append(job_id)
    with job_registry_lock:
        # Only remove the jobs we actually stopped, not all jobs (prevents race with new submissions)
        for job_id in deleted:
            job_registry.pop(job_id, None)
    return {"status": "success", "message": f"Jobs {deleted} deleted successfully."}


# === STOP JOBS ====================================================================================
@router.get(
    API_PREFIX + "/job/{job_id}/dequeue",
    summary="Stop a running lm-evaluation-harness job.",
)
def stop_lm_eval_job(job_id: int) -> dict[str, str]:
    """Stop an lm-evaluation-harness job by ID."""
    with job_registry_lock:
        if job_id not in job_registry:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"No lm-evaluation-harness job with ID={job_id} found.",
            )
        job = job_registry[job_id]
    if job.is_in_queue:
        job.dequeue()
        return {"status": "success", "message": f"Job {job_id} dequeued successfully."}
    if job.has_been_stopped:
        return {
            "status": "success",
            "message": f"Job {job_id} has already been stopped.",
        }
    if job.process is None:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} is not in queue but has no process",
        )
    if not job.has_been_stopped and job.process.poll() is None:
        job.process.terminate()
        try:
            job.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Process didn't terminate gracefully, force kill
            job.process.kill()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Process still won't die - log and continue
                logger.warning("Job %s did not terminate after SIGKILL", job_id)
        job.has_been_stopped = True
        return {"status": "success", "message": f"Job {job_id} stopped successfully."}
    return {"status": "success", "message": f"Job {job_id} has already completed."}


@router.get(
    API_PREFIX + "/jobs/dequeue", summary="Stop all running lm-evaluation-harness jobs."
)
def stop_all_lm_eval_job() -> dict[str, str]:
    """Stop all lm-evaluation-harness jobs."""
    stopped = []
    with job_registry_lock:
        job_ids = list(job_registry.keys())
    for job_id in job_ids:
        stop_lm_eval_job(job_id)
        stopped.append(job_id)

    return {"status": "success", "message": f"Jobs {stopped} stopped successfully."}
