import datetime
import os
import queue
import shlex
import threading
from contextlib import asynccontextmanager

from enum import Enum
import sys
from typing import Optional, List, Dict

try:
    from lm_eval.__main__ import setup_parser as lm_eval_setup_parser
    from fastapi_utils.tasks import repeat_every
except ImportError:
    raise ImportError(
        "The TrustyAI service was not built with LM-Evaluation-Harness support, use `pip install .[eval]`"
    )

from pydantic import BaseModel, create_model

from fastapi import HTTPException
from fastapi import APIRouter
import subprocess
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
logger = logging.getLogger(__name__)
API_PREFIX = "/eval/lm-evaluation-harness"


# === STATIC API OBJECTS ===========================================================================
class JobStatus(Enum):
    RUNNING = "Running"
    FAILED = "Failed"
    COMPLETED = "Completed"
    QUEUED = "Queued"
    STOPPED = "Stopped"


class LMEvalJobSummary(BaseModel):
    job_id: int
    argument: str
    status: JobStatus
    timestamp: Optional[str]
    exit_code: Optional[int]
    inference_progress_pct: int


class LMEvalJobDetail(LMEvalJobSummary):
    stdout: List[str]
    stderr: List[str]


class AllLMEvalJobs(BaseModel):
    jobs: List[LMEvalJobSummary]


# === Dynamic API Object from LM-Eval CLI ==========================================================
NON_CLI_ARGUMENTS = {"env_vars": (Dict[str, str], {}), "lm_eval_path": (str, f"{sys.executable} -m lm_eval")}


def get_lm_eval_arguments():
    """Grab all fields from an argparse specification into a dictionary"""
    parser = lm_eval_setup_parser()  # grab lm-eval argparse specification

    args = {}
    for action in parser._positionals._actions:
        arg = {"cli": action.option_strings[0], "argparse_type": action.__class__.__name__}
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


def get_model():
    """Build a Pydantic model from the lm-eval argparse arguments, adding in a few config variables of our own as well"""
    args = get_lm_eval_arguments()
    model_args = {k: (v["type"], v["default"]) for k, v in args.items()}
    model_args.update(NON_CLI_ARGUMENTS)
    return create_model("LMEvalRequest", **model_args)


# Dynamically create the lm-eval-harness job request from the library's argparse
LMEvalRequest = get_model()


# === Registry of running lm-eval-harness jobs =====================================================
class LMEvalJob:
    def __init__(self, job_id, request, argument):
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

    def mark_launch(self, process, start_time):
        self.start_time = start_time
        self.process = process
        self.is_in_queue = False

    def dequeue(self):
        self.has_been_stopped = True
        self.is_in_queue = False


job_registry: Dict[int, LMEvalJob] = {}  # track all jobs
job_queue = queue.Queue()
ID_LOCK = threading.Lock()
LAST_ID = 0

# === ENV VARIABLE PARSING =========================================================================
try:
    MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 4))
except ValueError:
    MAX_CONCURRENCY = 4

try:
    QUEUE_PROCESS_INTERVAL = int(os.environ.get("QUEUE_PROCESS_INTERVAL", 15))
except ValueError:
    QUEUE_PROCESS_INTERVAL = 15


# === HELPERS ======================================================================================
def convert_to_cli(request: LMEvalRequest):
    """Convert an LMEvalRequest json object into an lm-eval cli argument"""
    args = get_lm_eval_arguments()  # grab the cli argument spec to translate json to cli

    cli_cmd = request.lm_eval_path
    for field in request.model_fields_set:
        if field in NON_CLI_ARGUMENTS:
            continue

        cli_cmd += " "
        arg = args[field]
        if arg["argparse_type"] in {"_StoreTrueAction", "_StoreFalseAction"}:
            cli_cmd += args[field]["cli"]
        else:
            field_value = getattr(request, field)
            field_value = shlex.quote(field_value) if isinstance(field_value, str) else field_value
            cli_cmd += f"{args[field]['cli']} {field_value}"

    return cli_cmd


def _get_job_status(job):
    """Poll a running subprocess for exit codes"""
    if job.is_in_queue:
        status = JobStatus.QUEUED
        status_code = None
    elif job.has_been_stopped:
        status = JobStatus.STOPPED
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


def _generate_job_id():
    """Generate a unique job id"""
    with ID_LOCK:
        global LAST_ID
        LAST_ID += 1
        id_to_return = LAST_ID
    return id_to_return


# === Queuing ======================================================================================
def _get_num_running_jobs():
    """Get the number of currently running jobs"""
    jobs = list_running_lm_eval_jobs(include_finished=False).jobs
    jobs = [job for job in jobs if job.status == JobStatus.RUNNING]
    return len(jobs)


@repeat_every(seconds=QUEUE_PROCESS_INTERVAL, logger=logger)
async def _process_queue():
    """Check the job queue for pending jobs and launch them if there are available execution slots"""
    logger.debug(f"Queue processing: num_running_jobs: {_get_num_running_jobs()}, job queue size: {job_queue.qsize()}")
    while _get_num_running_jobs() < MAX_CONCURRENCY and job_queue.qsize() > 0:
        job_to_run = job_registry.get(job_queue.get())
        if job_to_run is not None and job_to_run.is_in_queue:
            logger.info(f"Launching job {job_to_run.job_id}")
            _launch_job(job_to_run)


def _launch_job(job: LMEvalJob):
    """Launch a job"""
    logger.debug(f"Running command:       {job.argument}")
    logger.debug(f"Environment variables: {job.request.env_vars}")

    # todo: verify that you can't inject commands here
    p = subprocess.Popen(shlex.split(job.argument), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    os.set_blocking(p.stdout.fileno(), False)
    os.set_blocking(p.stderr.fileno(), False)

    # register the subprocess in the global registry
    job_registry[job.job_id].mark_launch(process=p, start_time=datetime.datetime.now(datetime.timezone.utc).isoformat())


# === ROUTER =======================================================================================
@asynccontextmanager
async def lifespan(app):
    """Launch recurring queue processing function"""
    await _process_queue()
    yield


router = APIRouter(lifespan=lifespan)


# === API ==========================================================================================
@router.post("/job", summary="Launch an lm-evaluation-harness job")
def lm_eval_job(request: LMEvalRequest):
    """Launch an lm-evaluation-harness job according to the inbound arguments. These
    match the CLI arguments to lm-evaluation-harness, just in json form."""

    # convert the json to cli arguments
    cli_cmd = convert_to_cli(request)

    # store job
    job_id = _generate_job_id()
    queued_job = LMEvalJob(job_id=job_id, request=request, argument=cli_cmd)
    job_queue.put(job_id)
    job_registry[job_id] = queued_job

    return {"status": "success", "message": f"Job {job_id} successfully queued.", "job_id": job_id}


# === METADATA =====================================================================================
@router.get(API_PREFIX + "/jobs", summary="List all running jobs")
def list_running_lm_eval_jobs(include_finished: bool = True) -> AllLMEvalJobs:
    """Provide a list of all lm-evaluation-harness jobs with attached summary information"""

    jobs = []
    for pid, job in job_registry.items():
        job_information = check_lm_eval_job(pid)

        if not include_finished and job_information.status not in {JobStatus.RUNNING, JobStatus.FAILED}:
            continue
        jobs.append(LMEvalJobSummary(**job_information.model_dump(exclude={"stdout", "stderr"})))

    return AllLMEvalJobs(jobs=jobs)


@router.get(API_PREFIX + "/job/{job_id}", summary="Get information about a specific job")
def check_lm_eval_job(job_id: int) -> LMEvalJobDetail:
    """Get detailed report of an lm-evaluation-harness job by ID"""

    if job_id not in job_registry:
        raise HTTPException(status_code=400, detail=f"No lm-evaluation-harness job with ID={job_id} found.")

    job = job_registry[job_id]
    status, status_code = _get_job_status(job)

    progress = 0
    if not job.is_in_queue and job.process is not None:
        job.cumulative_out += [line.strip() for line in job.process.stdout]
        job.cumulative_err += [line.strip() for line in job.process.stderr]

        for line in reversed(job.cumulative_err):
            if line.startswith("Requesting API:"):
                progress = int(line.split("Requesting API:")[1].split("%")[0].strip())
                break

        job.progress = progress

    return LMEvalJobDetail(
        job_id=job_id,
        argument=job.argument,
        timestamp=job.start_time,
        status=status,
        exit_code=status_code,
        inference_progress_pct=job.progress,
        stdout=job.cumulative_out,
        stderr=job.cumulative_err,
    )


# === DELETE DATA ==================================================================================
@router.delete(API_PREFIX + "/job/{id}", summary="Delete an lm-evaluation-harness job's data from the server.")
def delete_lm_eval_job(job_id: int):
    """Delete an lm-evaluation-harness job's data from the server by ID, terminating the job if it's still running"""
    if job_id not in job_registry:
        raise HTTPException(status_code=400, detail=f"No lm-evaluation-harness job with ID={job_id} found.")

    stop_lm_eval_job(job_id)
    del job_registry[job_id]
    return {"status": "success", "message": f"Job {job_id} deleted successfully."}


@router.delete(API_PREFIX + "/jobs", summary="Delete data from all lm-evaluation-harness jobs from the server.")
def delete_all_lm_eval_job():
    """Delete data from all lm-evaluation-harness job's data from the server, terminating any job that its still running"""

    deleted = []
    for job_id in job_registry.keys():
        stop_lm_eval_job(job_id)
        deleted.append(job_id)
    job_registry.clear()
    return {"status": "success", "message": f"Jobs {deleted} deleted successfully."}


# === STOP JOBS ====================================================================================
@router.get(API_PREFIX + "/job/{job_id}/dequeue", summary="Stop a running lm-evaluation-harness job.")
def stop_lm_eval_job(job_id: int):
    """Stop an lm-evaluation-harness job by ID"""

    if job_id not in job_registry:
        raise HTTPException(status_code=400, detail=f"No lm-evaluation-harness job with ID={job_id} found.")

    job = job_registry[job_id]
    if job.is_in_queue:
        job.dequeue()
        return {"status": "success", "message": f"Job {job_id} dequeued successfully."}
    else:
        if not job.has_been_stopped and job.process.poll() is None:
            job.process.terminate()
            job.has_been_stopped = True
            return {"status": "success", "message": f"Job {job_id} stopped successfully."}
        else:
            return {"status": "success", "message": f"Job {job_id} has already completed."}


@router.get(API_PREFIX + "/jobs/dequeue", summary="Stop all running lm-evaluation-harness jobs.")
def stop_all_lm_eval_job():
    """Stop all lm-evaluation-harness jobs"""

    stopped = []
    for job_id in job_registry.keys():
        stop_lm_eval_job(job_id)
        stopped.append(job_id)

    return {"status": "success", "message": f"Jobs {stopped} stopped successfully."}
