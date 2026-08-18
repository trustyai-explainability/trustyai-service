"""Constants used throughout the TrustyAI service."""

# data constants
INPUT_SUFFIX = "_inputs"
OUTPUT_SUFFIX = "_outputs"
METADATA_SUFFIX = "_metadata"
PROTECTED_DATASET_SUFFIX = "trustyai_internal_"
PARTIAL_PAYLOAD_DATASET_NAME = "partial_payloads"
GROUND_TRUTH_SUFFIX = "-ground-truths"
METADATA_FILENAME = "metadata.json"
INTERNAL_DATA_FILENAME = "internal_data.csv"
# Payload parsing
TRUSTYAI_TAG_PREFIX = "_trustyai"
SYNTHETIC_TAG = f"{TRUSTYAI_TAG_PREFIX}_synthetic"
UNLABELED_TAG = f"{TRUSTYAI_TAG_PREFIX}_unlabeled"
TAGS_COLUMN = "tags"
BIAS_IGNORE_PARAM = "bias-ignore"
DATA_TAG_PARAM = "_trustyai_data_tag"  # Stores tag in partial payload parameters
# Prometheus constants
PROMETHEUS_METRIC_PREFIX = "trustyai_"
