import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Optional local copies of the official CSV files; splits.official_csv streams a missing file from the public dataset bucket.
RAW_DIR = Path(os.environ.get("IDSVAE_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DAY_FILES = {
    "02-14": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-15": "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-21": "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-23": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-28": "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
}
# Byte size of each official CSV file that the pipeline reads (protocol days and extra_days.EXTRA_DAY_FILES): a truncated or
# replaced file would change the splits without an error.
SOURCE_FILE_BYTES = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": 358223333,
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv": 375945899,
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv": 328893673,
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv": 382840456,
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv": 209249758,
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": 333723605,
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv": 4054925350,
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv": 382636202,
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv": 107842858,
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": 352368373,
}

# Day roles follow Sect. 4.1 (Dataset and Protocol) of the paper.
SPLIT_DAYS = {
    "fit": ["02-14", "02-15"],
    "vae_val": ["02-28"],
    "calib": ["02-23"],
    "test": ["02-21"],
}

LABEL_COLUMN = "Label"
TIME_COLUMN = "Timestamp"
TIME_FORMAT = "%d/%m/%Y %H:%M:%S"
# The Timestamp column uses a 12-hour clock without AM/PM, and the capture runs 08:00-17:59, so a parsed hour below 8 is an afternoon hour.
CAPTURE_FIRST_HOUR = 8
CAPTURE_YEAR = 2018
# Each attack tool in the capture targets one fixed port, so Dst Port is a near-label shortcut.
EXCLUDED_COLUMNS = [TIME_COLUMN, "Dst Port"]
# On the fit days these TCP fields identify the attacker machines (most attack rows carry a value with P(attack) > 0.95), so the main runs exclude them as they exclude Dst Port.
FINGERPRINT_COLUMNS = ["Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min"]
BENIGN = "Benign"
# Label of a flow that Stage 3 calls Benign and the unseen-attack detector flags.
UNKNOWN_ATTACK = "Unknown attack"

# Fraction of each (day, class) group, latest flows first, that forms the seen-class hold-out.
SEEN_HOLDOUT_FRACTION = 0.2

# A benign rate inside an attack window this many times the rate outside the windows marks the benign rows there as probable unlabeled attack traffic (02-21 HOIC: about 16,000 against 6 flows per minute).
BENIGN_BURST_RATIO = 10.0
# Fields that describe a flow template; most rows of a benign burst share one template (02-21 HOIC window: 100 %; normal benign traffic: about 20 %).
BURST_TEMPLATE_COLUMNS = ["Protocol", "Tot Fwd Pkts", "Tot Bwd Pkts", "Init Fwd Win Byts", "Init Bwd Win Byts"]
BURST_TEMPLATE_SHARE = 0.5


def project_path(argument, default):
    """A command-line path relative to the project root, or the default when the argument is absent."""
    return default if argument is None else PROJECT_ROOT / argument
