"""
Project Settings
================
Edit this file to configure the platform.
All scripts read from here — change once, affects everything.
"""
from pathlib import Path


class Settings:
    # ------------------------------------------------------------------
    # Study Area
    # ------------------------------------------------------------------
    PLACE        = "Surabaya, Indonesia"
    BBOX         = (-7.3545, 112.6085, -7.1975, 112.8273)  # south, west, north, east
    NETWORK_TYPE = "drive"   # "drive" | "walk" | "bike" | "all"

    # ------------------------------------------------------------------
    # Directories  (auto-created on first run)
    # ------------------------------------------------------------------
    BASE_DIR  = Path(__file__).parent
    DATA_DIR  = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    LOG_DIR   = BASE_DIR / "logs"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL   = "INFO"    # DEBUG | INFO | WARNING | ERROR
    LOG_TO_FILE = True      # writes timestamped .log file to LOG_DIR

    # ------------------------------------------------------------------
    # OSM / Network
    # ------------------------------------------------------------------
    OSM_TIMEOUT   = 300     # seconds before Overpass API times out
    OSM_USE_CACHE = True    # cache downloaded data (speeds up reruns)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    N_SCENARIOS              = 2     # two focused scenarios: emergency + terminal
    N_COVERAGE_SAMPLES       = 100   # random points for coverage analysis
    MAX_FACILITIES_PER_CAT   = 50    # cap per category (keeps coverage fast)

    # ------------------------------------------------------------------
    # Facility Categories
    # Each entry: tags (OSM key→values) and priority (1=highest)
    # ------------------------------------------------------------------
    FACILITY_CATEGORIES = {
        "healthcare": {
            "tags":     {"amenity": ["hospital", "clinic", "doctors", "pharmacy", "dentist"]},
            "priority": 1,
        },
        "education": {
            "tags":     {"amenity": ["school", "university", "college", "kindergarten", "library"]},
            "priority": 2,
        },
        "emergency": {
            "tags":     {"amenity": ["police", "fire_station"]},
            "priority": 1,
        },
        "government": {
            "tags":     {"amenity": ["townhall", "courthouse", "post_office"],
                         "office": ["government"]},
            "priority": 3,
        },
        "transport": {
            "tags":     {"amenity": ["bus_station", "ferry_terminal", "fuel"]},
            "priority": 3,
        },
        "community": {
            "tags":     {"amenity": ["place_of_worship", "community_centre", "marketplace"]},
            "priority": 4,
        },
    }

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
