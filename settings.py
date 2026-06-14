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
            "tags":     {"amenity": ["bus_station", "ferry_terminal", "fuel"],
                         "highway": ["bus_stop"],
                         "public_transport": ["station", "platform", "stop_position"]},
            "priority": 3,
        },
        "community": {
            "tags":     {"amenity": ["place_of_worship", "community_centre", "marketplace"]},
            "priority": 4,
        },
    }

# ------------------------------------------------------------------
    # Vehicle & Fuel Constraint
    # ------------------------------------------------------------------

    # Jenis kendaraan yang tersedia.
    # range_km = tank_liters × fuel_efficiency_kpl
    # width_m  = lebar fisik kendaraan (m) — dipakai untuk constraint lebar jalan:
    #            edge dengan lebar jalan <= width_m tidak bisa dilewati.
    VEHICLE_TYPES = {
        "motor": {
            "label":               "Motor",
            "tank_liters":         4.5,    # kapasitas tangki (liter)
            "fuel_efficiency_kpl": 40.0,   # km per liter
            "range_km":            150.0,  # jarak maksimum full tank
            "width_m":             0.8,    # lebar fisik motor (m)
            "avg_speed_kph":       40.0,   # kecepatan rata-rata (untuk insight)
        },
        "mobil": {
            "label":               "Mobil",
            "tank_liters":         40.0,
            "fuel_efficiency_kpl": 12.0,
            "range_km":            375.0,
            "width_m":             1.8,    # lebar fisik mobil (m)
            "avg_speed_kph":       40.0,
        },
    }

    # Kendaraan default jika tidak di-set via CLI
    DEFAULT_VEHICLE = "motor"

    # Safety buffer: isi bensin kalau sisa range < threshold ini (km).
    # Mencegah kehabisan bensin sebelum SPBU ditemukan.
    FUEL_REFILL_THRESHOLD_KM = 10.0

    # ------------------------------------------------------------------
    # Team Orienteering Problem (TOP) — maksimasi titik dikunjungi
    # ------------------------------------------------------------------
    # Objektif: maksimasi jumlah titik distinct yang dikunjungi dari depot
    # kembali ke depot, di bawah batas jam kerja (shift). Bukan lagi
    # shortest-path/TSP min-time.

    SERVICE_TIME_S      = 600       # 10 menit berhenti per titik untuk cek fasilitas
    SHIFT_SECONDS       = 6 * 3600  # 1 shift = 6 jam jam kerja
    N_SHIFTS            = 2         # 2 shift per kendaraan (6 jam + 6 jam)
    TRAINING_ITERATIONS = 10        # tiap model dijalankan 10x per kendaraan
    MIN_POINTS_TARGET   = 55        # target minimum titik per kendaraan (2 shift)

    # Armada default dalam satu training. Jumlah kendaraan = variabel insight.
    # Titik distinct lintas-kendaraan & lintas-shift (no-overlap).
    FLEET = {"motor": 1, "mobil": 1}

    # Constraint lebar jalan: edge tanpa data width dianggap bisa dilewati?
    WIDTH_MISSING_PASSABLE = True

    # Node depot (start/return) — markas Polda Jatim (sudah dipakai di benchmark).
    DEPOT_NODE = 9156956728

    # Pool titik kandidat (objektif maksimasi). Eliminasi data lain.
    # Tiap grup dipilih SEJUMLAH `count` titik yang TERSEBAR MERATA secara
    # geografis di Surabaya (farthest-point sampling) — bukan titik bebas/acak.
    # Default: 60 emergency + 60 transport = 120 titik tetap yang bisa dikunjungi.
    POOL_GROUPS = {
        "emergency": {
            "match_categories":     ["emergency"],          # polisi + pemadam
            "count":                60,
        },
        "transport": {
            "match_facility_types": ["bus_stop", "bus_station"],
            "count":                60,
        },
    }

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
