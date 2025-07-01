import os

# Paths will be read from environment variables, with sensible defaults.
# This makes the code portable between your local machine and Kaggle.
INPUT_DATA_ROOT = os.getenv("INPUT_DATA_ROOT", "/kaggle/input/roomsdataset/500_empty_staged_384px")
# We will use this as the primary data directory for scripts that need a single path.
DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/kaggle/working/processed_data") 

PAIRED_DATA_DIR_INPUT = INPUT_DATA_ROOT

PAIRED_DATA_DIR_OUTPUT = os.path.join(DATA_DIR, "500_empty_staged_384px_processed")

# Constants
STAGED = "staged"
EMPTY = "empty"
AGNOSTIC = "agnostic"
MASK = "mask"
CAPTION = "caption"
PNG = "png"