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

## 5. Cara Tambah Model Tim Kamu

> Ini bagian terpenting. Baca pelan-pelan ya.

### Konsep dasarnya
Program akan tanya ke model kamu: **"berapa biaya untuk melewati jalan ini?"**

Kamu jawab dengan angka. Makin kecil angkanya, makin dipilih oleh router.  
Setelah itu, program otomatis cari rute terbaik dan tampilkan di peta.

### Langkah-langkahnya

#### Langkah 1 — Buka file `src/routing/algorithms.py`

Cari bagian `TeamAModel` dan `TeamBModel`.

#### Langkah 2 — Isi fungsi `predict_edge_weight`

```python
class TeamAModel(BaseRoutingAlgorithm):
    name        = "team_a"
    description = "Model Tim A"   # ← ganti deskripsi

    def predict_edge_weight(self, u, v, edge_data):
        #
        # Di sini kamu masukkan model kamu.
        # edge_data berisi informasi tentang jalan yang sedang dinilai:
        #
        #   edge_data["length"]       → panjang jalan (meter)
        #   edge_data["travel_time"]  → waktu tempuh (detik)
        #   edge_data["speed_kph"]    → kecepatan (km/jam)
        #   edge_data["highway"]      → tipe jalan ("primary", "residential", dll)
        #
        # Kembalikan satu angka float.
        # Makin kecil = makin disukai algoritma.
        #

        # Contoh 1: pakai travel_time langsung (sama kayak Dijkstra biasa)
        return edge_data.get("travel_time", 9999)

        # Contoh 2: pakai model ML yang sudah kamu train
        # fitur = [edge_data["length"], edge_data["speed_kph"]]
        # return float(model_kamu.predict([fitur])[0])

        # Contoh 3: kombinasi bobot manual
        # return edge_data["travel_time"] * 0.7 + edge_data["length"] * 0.3
```

> **Tim B** — sama persis, tapi isi di bagian `TeamBModel`.

#### Langkah 3 — Jalankan compare

```bash
python main.py compare
```

atau pilih **angka 4** di `run.bat`.

#### Hasilnya otomatis muncul di `data/`
- Peta rute Tim A vs Tim B vs baseline (buka `.html` di browser)
- Grafik perbandingan waktu dan kecepatan komputasi
- Tabel CSV dengan semua angka

---

### Contoh nyata: model pakai bobot prioritas fasilitas

Misal Tim A punya ide: *"jalan dekat RS dan sekolah harus diprioritaskan"* — implementasinya:

```python
def predict_edge_weight(self, u, v, edge_data):
    base_time = edge_data.get("travel_time", 9999)

    # Kurangi bobot untuk jalan utama (lebih disukai)
    highway = edge_data.get("highway", "")
    if highway in ("primary", "trunk"):
        base_time *= 0.85    # 15% lebih ringan

    return base_time
```

---

### Contoh nyata: model ML sudah di-train di luar

```python
import joblib   # untuk load model sklearn

class TeamAModel(BaseRoutingAlgorithm):
    name = "team_a"
    description = "Model Tim A — Random Forest"

    def __init__(self):
        # Load model yang sudah di-train
        self.model = joblib.load("models/team_a_model.pkl")

    def predict_edge_weight(self, u, v, edge_data):
        fitur = [[
            edge_data.get("length", 0),
            edge_data.get("travel_time", 0),
            edge_data.get("speed_kph", 30),
        ]]
        return float(self.model.predict(fitur)[0])
```

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
