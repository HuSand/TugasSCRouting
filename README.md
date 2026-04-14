# Panduan Penggunaan — Surabaya Routing Platform
> Platform perbandingan algoritma routing untuk monitoring fasilitas publik di Surabaya.

---

## Daftar Isi
1. [Persiapan Awal (Install)](#1-persiapan-awal)
2. [Cara Menjalankan Program](#2-cara-menjalankan-program)
3. [Alur Pipeline](#3-alur-pipeline)
4. [Cara Ganti Konfigurasi](#4-cara-ganti-konfigurasi)
5. [Cara Tambah Model Tim Kamu](#5-cara-tambah-model-tim-kamu)
6. [Hasil Output](#6-hasil-output)
7. [Struktur Folder](#7-struktur-folder)
8. [FAQ](#8-faq)

---

## 1. Persiapan Awal

### Yang dibutuhkan
- Python 3.10 atau 3.11 (download di [python.org](https://www.python.org/downloads/))  
  **Centang "Add Python to PATH" saat install**
- Atau Anaconda / Miniconda (jika pakai conda)
- Koneksi internet (untuk download data peta dari OpenStreetMap)

### Install semua library
Cukup **double-click** file `install.bat`.

Program akan otomatis:
- Deteksi apakah kamu pakai conda atau Python biasa
- Buat virtual environment
- Install semua library yang dibutuhkan
- Verifikasi instalasi berhasil

```
install.bat  ← double-click ini, selesai
```

---

## 2. Cara Menjalankan Program

### Cara termudah — pakai menu
**Double-click** `run.bat`, lalu pilih angka:

```
=========================================================
  Surabaya Public Facility Routing Platform
=========================================================

  1   Extract facilities + road network  (butuh internet, ~10 menit)
  2   Explore dan profile data
  3   Routing demonstrations
  4   Algorithm comparison benchmark
  ─────────────────────────────────────────────────────
  5   Jalankan semua (1 → 2 → 3 → 4)

  0   Keluar
```

### Atau lewat terminal
Aktifkan environment dulu, lalu:
```bash
python main.py extract    # download data dari OpenStreetMap
python main.py explore    # lihat statistik dan peta
python main.py demo       # demo routing dasar
python main.py compare    # bandingkan semua algoritma
python main.py all        # jalankan semua sekaligus
```

> **Catatan:** Jalankan `extract` dulu sebelum yang lain.  
> Data akan tersimpan di folder `data/` dan tidak perlu download ulang.

---

## 3. Alur Pipeline

```
[ 1. extract ]  →  Download fasilitas (RS, sekolah, polisi, dll)
                   dan jaringan jalan dari OpenStreetMap
                   Output: data/road_network.graphml
                           data/facilities_with_network.geojson

[ 2. explore ]  →  Profil data, cek kualitas, buat peta interaktif
                   Output: data/surabaya_facilities_map.html  ← buka di browser

[ 3. demo    ]  →  Demo routing: rute tercepat vs terpendek,
                   cari RS terdekat, analisis coverage
                   Output: data/demo_shortest_path.html

[ 4. compare ]  →  Jalankan semua algoritma tim pada skenario yang sama,
                   bandingkan hasilnya
                   Output: data/comparison_chart.png
                           data/comparison_map_*.html
                           data/comparison_results.csv
```

---

## 4. Cara Ganti Konfigurasi

Buka file **`settings.py`** — ini satu-satunya file yang perlu diedit untuk mengubah pengaturan.

```python
# Ganti kota target
PLACE = "Surabaya, Indonesia"   # ← ubah ke kota lain kalau perlu

# Ganti tipe jaringan jalan
NETWORK_TYPE = "drive"   # "drive" = jalan mobil
                         # "walk"  = jalan kaki
                         # "bike"  = jalur sepeda

# Jumlah skenario yang diuji saat compare
N_SCENARIOS = 5          # ← tambah angkanya untuk lebih banyak perbandingan
```

---

## 5. Cara Tuning GA Tim Kamu

> Ini bagian terpenting. Baca pelan-pelan ya.

### Konsep dasarnya — Genetic Algorithm (GA)

GA meniru proses evolusi untuk menemukan rute terbaik:

| Istilah GA | Artinya di routing ini |
|---|---|
| **Individu / kromosom** | Satu rute (urutan node dari asal ke tujuan) |
| **Populasi** | Sekumpulan rute kandidat |
| **Fitness** | Total `travel_time` rute — makin kecil makin bagus |
| **Seleksi** | Pilih rute terbaik sebagai "orang tua" (tournament selection) |
| **Crossover** | Gabungkan dua rute di node persimpangan yang sama |
| **Mutasi** | Ganti satu segmen rute dengan jalur alternatif |
| **Elitisme** | Rute terbaik selalu lolos ke generasi berikut |

Setelah N generasi, GA mengembalikan rute terbaik yang ditemukan.

---

### Langkah-langkahnya

#### Langkah 1 — Buka file `src/routing/algorithms.py`

Cari bagian `TeamAGA` (Tim A) atau `TeamBGA` (Tim B).

#### Langkah 2 — Ubah parameter di TUNING ZONE

```python
class TeamAGA(BaseRoutingAlgorithm):
    name        = "team_a_ga"
    description = "Team A — Genetic Algorithm"

    # ── TUNING ZONE Tim A ─────────────────────────────────────
    POPULATION_SIZE = 30     # jumlah rute per generasi (lebih besar = lebih beragam)
    GENERATIONS     = 50     # berapa kali evolusi (lebih banyak = lebih matang)
    CROSSOVER_RATE  = 0.8    # peluang crossover terjadi (0.0–1.0)
    MUTATION_RATE   = 0.3    # peluang mutasi terjadi  (0.0–1.0)
    TOURNAMENT_SIZE = 3      # peserta tournament selection (lebih besar = lebih ketat)
    RANDOM_SEED     = 42     # set None → hasil non-deterministik setiap run
    # ─────────────────────────────────────────────────────────
```

> **Tim B** — sama persis, tapi ubah di bagian `TeamBGA`.  
> Coba parameter yang berbeda dari Tim A supaya perbandingannya menarik!

#### Tips tuning parameter

| Parameter | Naikkan jika... | Turunkan jika... |
|---|---|---|
| `POPULATION_SIZE` | hasil kurang beragam | benchmark terlalu lambat |
| `GENERATIONS` | rute masih bisa membaik | benchmark terlalu lambat |
| `CROSSOVER_RATE` | eksplorasi kurang | sudah konvergen terlalu cepat |
| `MUTATION_RATE` | terjebak di rute lokal | hasil tidak stabil/acak |
| `TOURNAMENT_SIZE` | ingin seleksi lebih ketat | ingin lebih banyak variasi |

#### Langkah 3 — Jalankan compare

```bash
python main.py compare
```

atau pilih **angka 4** di `run.bat`.

#### Hasilnya otomatis muncul di `data/`
- Peta rute Tim A GA vs Tim B GA vs baseline Dijkstra/A* (buka `.html` di browser)
- Grafik perbandingan `travel_time` dan kecepatan komputasi
- Tabel CSV dengan semua angka

---

### Skenario yang diuji (fasilitas Surabaya nyata)

| # | Nama | Dari | Ke | Konteks |
|---|---|---|---|---|
| 1 | `darmo_to_rsu_haji` | RS Darmo | RSU Haji Surabaya | Transfer pasien |
| 2 | `polsek_genteng_to_rs_darmo` | Polsek Genteng | RS Darmo | Respons darurat polisi |
| 3 | `national_to_rs_ramelan` | National Hospital | RS Ramelan | Lintas kota barat→selatan |
| 4 | `polsek_rungkut_to_rs_onkologi` | Polsek Rungkut | RS Onkologi | Darurat area timur |
| 5 | `ciputra_to_rsu_haji` | Ciputra Hospital | RSU Haji | Rute terpanjang lintas kota |

Node ID diambil dari `data/facilities_with_network.csv`.  
Semua tim diuji pada skenario yang **sama persis** — tidak ada yang random.

---

## 6. Hasil Output

Semua hasil tersimpan di folder `data/`.

| File | Isi |
|------|-----|
| `surabaya_facilities_map.html` | Peta interaktif semua fasilitas (buka di browser) |
| `comparison_map_hosp_to_school.html` | Rute semua algoritma dari RS → Sekolah di atas peta nyata |
| `comparison_map_police_to_hosp.html` | Rute semua algoritma dari Polisi → RS |
| `comparison_chart.png` | Grafik: waktu tempuh & kecepatan komputasi per algoritma |
| `comparison_results.csv` | Data mentah: waktu, jarak, lama hitung untuk setiap skenario |
| `comparison_summary.csv` | Ringkasan rata-rata per algoritma |
| `logs/run_*.log` | Log lengkap setiap kali program dijalankan |

### Cara baca peta perbandingan
Buka file `.html` di browser (Chrome/Firefox). Setiap warna = satu algoritma.  
Hover ke garis rute untuk lihat waktu tempuh dan jarak.

---

## 7. Struktur Folder

```
📁 open street map/
│
├── 📄 settings.py          ← Konfigurasi (kota, parameter, dll)
├── 📄 main.py              ← Entry point utama
├── 📄 run.bat              ← Menu jalankan program (double-click)
├── 📄 install.bat          ← Install semua library (double-click sekali)
├── 📄 requirements.txt     ← Daftar library Python
│
├── 📁 src/
│   ├── 📄 extract.py       ← Download & proses data dari OpenStreetMap
│   ├── 📄 explore.py       ← Visualisasi & profil data
│   └── 📁 routing/
│       ├── 📄 algorithms.py  ← ⭐ FILE INI yang diisi tim kamu
│       ├── 📄 base.py        ← Class dasar (jangan diubah)
│       ├── 📄 benchmark.py   ← Sistem perbandingan algoritma
│       ├── 📄 visualize.py   ← Buat peta & grafik
│       └── 📄 demos.py       ← Demo routing dasar
│
├── 📁 data/                ← Semua output tersimpan di sini
├── 📁 logs/                ← Log setiap kali program dijalankan
└── 📁 cache/               ← Cache data OSM (supaya tidak re-download)
```

> **File yang perlu kamu sentuh:**
> - `settings.py` — kalau mau ganti kota atau parameter
> - `src/routing/algorithms.py` — untuk masukkan model tim kamu

---

## 8. FAQ

**Q: Waktu extract berapa lama?**  
A: Sekitar 5–10 menit tergantung koneksi internet. Setelah selesai, data tersimpan dan tidak perlu download ulang.

**Q: Bisa ganti kota selain Surabaya?**  
A: Bisa. Buka `settings.py`, ganti `PLACE` ke nama kota lain (harus nama yang ada di OpenStreetMap), lalu jalankan `extract` ulang.

**Q: Model tim A dan tim B harus berupa apa?**  
A: Apapun yang bisa menghasilkan satu angka dari informasi jalan. Bisa model ML (scikit-learn, PyTorch, dll), rumus manual, atau logika if-else sederhana.

**Q: Error "data not found" saat jalankan explore/compare?**  
A: Jalankan `extract` dulu (pilih 1 di menu atau `python main.py extract`).

**Q: Bagaimana cara lihat peta hasilnya?**  
A: Buka file `.html` di folder `data/` menggunakan browser Chrome atau Firefox.

**Q: Log tersimpan di mana?**  
A: Di folder `logs/` dengan nama `run_TANGGAL_JAM.log`. Berguna untuk debug kalau ada error.

**Q: Boleh tambah algoritma lebih dari 2 tim?**  
A: Boleh. Buat class baru di `algorithms.py` (copy skeleton di bagian bawah file), lalu tambahkan `registry.register(NamaClass())` di `benchmark.py`.
