# Surabaya Routing Platform

Platform Python untuk mengambil data jalan dan fasilitas publik dari OpenStreetMap, membangun skenario rute multi-stop di Surabaya, lalu membandingkan beberapa algoritma routing pada skenario yang sama.

Dokumentasi ini menjelaskan alur lengkap dari instalasi, ekstraksi data, eksplorasi, benchmark algoritma, sampai visualisasi hasil.

---

## Daftar Isi

1. [Ringkasan Sistem](#1-ringkasan-sistem)
2. [Kebutuhan Awal](#2-kebutuhan-awal)
3. [Instalasi](#3-instalasi)
4. [Cara Menjalankan](#4-cara-menjalankan)
5. [Pipeline Lengkap](#5-pipeline-lengkap)
6. [Data dan Skenario](#6-data-dan-skenario)
7. [Algoritma yang Dibandingkan](#7-algoritma-yang-dibandingkan)
8. [Cara Kerja Pemilihan Urutan Tujuan](#8-cara-kerja-pemilihan-urutan-tujuan)
9. [Output dan Visualisasi](#9-output-dan-visualisasi)
10. [Konfigurasi](#10-konfigurasi)
11. [Struktur Project](#11-struktur-project)
12. [Menambah atau Mengubah Algoritma](#12-menambah-atau-mengubah-algoritma)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Ringkasan Sistem

Project ini dipakai untuk membandingkan algoritma routing pada jaringan jalan nyata Surabaya.

Sistem melakukan hal berikut:

1. Mengambil data fasilitas publik dari OpenStreetMap.
2. Mengambil jaringan jalan Surabaya dari OpenStreetMap.
3. Menghubungkan setiap fasilitas ke node jalan terdekat.
4. Membuat skenario routing multi-stop.
5. Menjalankan beberapa algoritma pada skenario yang sama.
6. Menghasilkan tabel, grafik, peta interaktif, dan viewer evolusi algoritma.

Fokus benchmark saat ini adalah rute berbentuk circuit atau patroli:

- `emergency_patrol_circuit`: pos polisi dan pemadam kebakaran.
- `terminal_circuit`: terminal bus, terminal feri, dan SPBU.

Kedua skenario menggunakan `round_trip=True`, artinya rute kembali lagi ke titik awal.

---

## 2. Kebutuhan Awal

Pastikan komputer memiliki:

- Python 3.10 atau 3.11.
- `pip`.
- Koneksi internet untuk tahap `extract`.
- Browser modern untuk membuka file `.html` hasil visualisasi.

Jika memakai Windows, saat instalasi Python sebaiknya centang:

```text
Add Python to PATH
```

Project ini juga bisa dijalankan memakai Conda atau Miniconda.

---

## 3. Instalasi

Cara paling mudah di Windows:

```text
install.bat
```

Double-click file tersebut. Script akan mencoba mendeteksi environment yang tersedia, membuat environment Python, lalu memasang dependency dari `requirements.txt`.

Cara manual:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Jika memakai Conda:

```bash
conda env create -f environment.yml
conda activate surabaya-routing
```

---

## 4. Cara Menjalankan

### Opsi 1 - Menu Windows

Jalankan:

```text
run.bat
```

Menu utama berisi:

| Opsi | Fungsi |
|---|---|
| `1` | Extract fasilitas dan jaringan jalan dari OpenStreetMap |
| `2` | Explore dan profil data |
| `3` | Demo routing dasar |
| `4` | Benchmark semua algoritma |
| `5` | Jalankan semua pipeline |
| `7` | Benchmark dengan parallel legs |
| `8` | Buka `evolution_viewer.html` |

### Opsi 2 - Terminal

```bash
python main.py extract
python main.py explore
python main.py demo
python main.py compare
python main.py compare --parallel-legs
python main.py all
```

Urutan yang disarankan untuk pertama kali:

```bash
python main.py extract
python main.py explore
python main.py demo
python main.py compare
```

Setelah data berhasil diekstrak, tahap `extract` tidak perlu sering diulang.

---

## 5. Pipeline Lengkap

### 5.1 Extract

Command:

```bash
python main.py extract
```

Tahap ini menjalankan `src/extract.py`.

Yang dilakukan:

- Mengambil fasilitas publik dari OpenStreetMap.
- Mengambil jaringan jalan Surabaya.
- Menambahkan atribut `speed_kph` dan `travel_time` pada edge jalan.
- Menentukan `nearest_node` untuk setiap fasilitas.
- Menyimpan hasil ke folder `data/`.

Output utama:

| File | Fungsi |
|---|---|
| `data/road_network.graphml` | Graph jaringan jalan untuk NetworkX/OSMnx |
| `data/road_network_nodes.geojson` | Node jaringan jalan |
| `data/road_network_edges.geojson` | Edge jaringan jalan |
| `data/facilities_with_network.geojson` | Fasilitas publik yang sudah terhubung ke node jalan |
| `data/facilities_with_network.csv` | Versi CSV dari data fasilitas |
| `data/extraction_report.txt` | Ringkasan hasil ekstraksi |

Tahap ini membutuhkan internet dan bisa memakan waktu beberapa menit.

### 5.2 Explore

Command:

```bash
python main.py explore
```

Tahap ini menjalankan `src/explore.py`.

Yang dilakukan:

- Membaca data hasil ekstraksi.
- Mengecek kualitas data fasilitas.
- Melihat distribusi kategori fasilitas.
- Membuat peta interaktif fasilitas.
- Membuat grafik eksplorasi data.

Output utama:

| File | Fungsi |
|---|---|
| `data/surabaya_facilities_map.html` | Peta interaktif semua fasilitas |
| `data/facility_distribution.png` | Grafik jumlah fasilitas per kategori |
| `data/snap_distance_histogram.png` | Histogram jarak fasilitas ke node jalan terdekat |
| `data/exploration_report.txt` | Ringkasan kualitas dan profil data |

### 5.3 Demo

Command:

```bash
python main.py demo
```

Tahap ini menjalankan `src/routing/demos.py`.

Demo dipakai sebagai sanity check sebelum benchmark utama.

Output utama:

| File | Fungsi |
|---|---|
| `data/demo_shortest_path.html` | Perbandingan rute tercepat dan terpendek |
| `data/demo_nearest_hospital.html` | Demo pencarian rumah sakit terdekat |
| `data/demo_coverage_report.txt` | Ringkasan coverage travel time |

### 5.4 Compare

Command:

```bash
python main.py compare
```

Tahap ini menjalankan `src/routing/benchmark.py`.

Yang dilakukan:

1. Load graph jalan dan data fasilitas.
2. Register algoritma benchmark.
3. Bangun skenario multi-stop.
4. Jalankan setiap algoritma pada setiap skenario.
5. Simpan hasil mentah dan ringkasan.
6. Buat peta perbandingan rute.
7. Buat grafik perbandingan performa.
8. Buat evolution viewer untuk algoritma yang punya `gen_history`.

Mode tambahan:

```bash
python main.py compare --parallel-legs
```

Mode ini memecah pekerjaan per leg agar lebih banyak core CPU terpakai. Biasanya lebih cepat, tetapi bisa memakai RAM lebih besar.

### 5.5 All

Command:

```bash
python main.py all
```

Menjalankan pipeline lengkap:

```text
extract -> explore -> demo -> compare
```

Gunakan ini jika ingin membangun ulang seluruh hasil dari awal.

---

## 6. Data dan Skenario

### 6.1 Kategori Fasilitas

Kategori diatur di `settings.py`.

| Kategori | Contoh fasilitas |
|---|---|
| `healthcare` | hospital, clinic, doctors, pharmacy, dentist |
| `education` | school, university, college, kindergarten, library |
| `emergency` | police, fire_station |
| `government` | townhall, courthouse, post_office, government office |
| `transport` | bus_station, ferry_terminal, fuel |
| `community` | place_of_worship, community_centre, marketplace |

### 6.2 Skenario Benchmark Utama

Benchmark default membuat dua skenario dari fasilitas nyata:

| Skenario | Isi | Bentuk |
|---|---|---|
| `emergency_patrol_circuit` | Pos polisi dan pemadam kebakaran, maksimal 50 node unik | Multi-stop round trip |
| `terminal_circuit` | Terminal bus, terminal feri, dan SPBU | Multi-stop round trip |

Kedua skenario dibuat oleh:

```text
src/routing/benchmark.py -> build_category_scenarios()
```

Tahap pembentukan skenario melakukan:

- Menghapus duplikat berdasarkan `nearest_node`.
- Memilih subset yang tersebar secara geografis jika jumlah emergency terlalu besar.
- Membuat urutan awal yang masuk akal secara geografis.
- Mengaktifkan `optimize_order=True`.
- Mengaktifkan `round_trip=True`.

Urutan awal hanya dipakai sebagai input awal. Saat benchmark berjalan, model tetap diberi kesempatan memilih urutan kunjungan sesuai mekanisme masing-masing.

---

## 7. Algoritma yang Dibandingkan

Algoritma yang aktif didaftarkan di `run_platform()` dalam `src/routing/benchmark.py`.

| Nama di hasil | Class | Peran |
|---|---|---|
| `ga` | `GeneticAlgorithm` | GA untuk path point-to-point dan TSP-GA untuk multi-stop |
| `aco_routing` | `AntColonyRouting` | Ant Colony Optimization untuk mencari path antar dua node |
| `gerald_sa` | `GeraldSimulatedAnnealing` | Simulated Annealing untuk optimasi path |
| `particle_swarm` | `ParticleSwarmRouting` | Particle Swarm Optimization untuk optimasi path |

Setiap algoritma mengembalikan `RouteResult` yang berisi:

- `route`: daftar node jalan yang dilalui.
- `total_time_s`: estimasi waktu tempuh.
- `total_distance_m`: total jarak.
- `nodes_in_route`: jumlah node dalam rute.
- `computation_ms`: waktu komputasi algoritma.
- `metadata`: informasi tambahan seperti urutan stop, leg, dan riwayat generasi.

---

## 8. Cara Kerja Pemilihan Urutan Tujuan

Bagian ini penting karena project sekarang memakai multi-stop routing.

Semua model hanya boleh memilih tujuan dari daftar `route_nodes` milik skenario. Model tidak membuat tujuan baru di luar daftar tersebut.

### 8.1 Genetic Algorithm

`GeneticAlgorithm` memiliki method khusus:

```text
_route_multi_stop()
```

Untuk multi-stop, GA:

1. Mengunci stop pertama sebagai titik mulai.
2. Menganggap stop lain sebagai daftar tujuan yang boleh diurutkan ulang.
3. Membuat populasi berisi permutation urutan stop.
4. Menggunakan crossover dan mutation khusus TSP.
5. Memilih urutan stop dengan total `travel_time` paling kecil.
6. Mengembangkan urutan stop terbaik menjadi rute jalan nyata.
7. Kembali ke titik awal.

Jadi GA benar-benar memilih urutan kunjungan sendiri.

### 8.2 ACO, SA, dan PSO

ACO, SA, dan PSO adalah optimizer path antar dua node. Karena itu benchmark memberi mereka lapisan order selection sebelum menjalankan leg-by-leg routing.

Saat `optimize_order=True`, benchmark:

1. Mengambil semua stop dari `route_nodes`.
2. Menghitung biaya antar stop memakai shortest path pada graph.
3. Memilih urutan kunjungan.
4. Menjalankan algoritma pada setiap leg sesuai urutan terpilih.
5. Menyambung semua leg menjadi satu rute penuh.
6. Menambahkan return leg jika `round_trip=True`.

Strategi pemilihan order:

| Jumlah stop | Strategi |
|---|---|
| 2 stop atau kurang | Tidak perlu reorder |
| Sampai 9 stop | Exact permutation search dengan start tetap |
| Lebih dari 9 stop | Greedy next-stop agar tetap aman untuk skenario besar |

Objective order mengikuti jenis algoritma:

- Jika nama algoritma mengandung `distance`, order memakai `length`.
- Selain itu, order memakai `travel_time`.

Pada benchmark saat ini, skenario kategori sudah memakai:

```python
optimize_order=True
round_trip=True
```

Artinya semua model sekarang dapat memilih urutan kunjungan dari daftar tujuan yang tersedia.

---

## 9. Output dan Visualisasi

### 9.1 Output Compare

Setelah menjalankan:

```bash
python main.py compare
```

hasil utama ada di folder `data/`.

| File | Fungsi |
|---|---|
| `comparison_results.csv` | Hasil mentah per algoritma dan skenario |
| `comparison_summary.csv` | Ringkasan rata-rata performa per algoritma |
| `comparison_chart.png` | Grafik travel time dan computation time |
| `comparison_map_emergency_patrol_circuit.html` | Peta rute semua algoritma untuk skenario emergency |
| `comparison_map_terminal_circuit.html` | Peta rute semua algoritma untuk skenario terminal |
| `evolution_viewer.html` | Viewer interaktif evolusi rute untuk algoritma dengan history |

Log tambahan ada di folder `logs/`.

| File | Fungsi |
|---|---|
| `run_*.log` | Log lengkap eksekusi command |
| `evolution_<algo>_<scenario>.txt` | Log generasi untuk algoritma yang punya `gen_history` |

### 9.2 Cara Membaca CSV

Kolom penting di `comparison_results.csv`:

| Kolom | Arti |
|---|---|
| `scenario` | Nama skenario |
| `algorithm` | Nama algoritma |
| `found` | Apakah rute berhasil ditemukan |
| `travel_time_min` | Estimasi waktu tempuh dalam menit |
| `distance_km` | Total jarak rute dalam kilometer |
| `nodes_in_route` | Jumlah node jalan dalam rute |
| `computation_ms` | Waktu komputasi algoritma |
| `meta_visit_order_nodes` | Urutan stop yang dipakai |
| `meta_order_objective` | Objective pemilihan order |
| `meta_order_score` | Skor order terpilih |

### 9.3 Cara Membaca Peta

Buka file `comparison_map_*.html` di browser.

Di peta:

- Setiap warna mewakili satu algoritma.
- Garis menunjukkan rute jalan yang dipilih.
- Pin menunjukkan daftar stop.
- Klik rute untuk melihat detail.
- Popup/legend menampilkan waktu, jarak, dan informasi rute.

### 9.4 Evolution Viewer

Buka:

```text
data/evolution_viewer.html
```

Viewer ini dipakai untuk melihat perubahan rute dari generasi ke generasi pada algoritma yang menyimpan `gen_history`, terutama GA.

Fitur utama:

- Pilih skenario.
- Pilih algoritma.
- Putar timeline generasi.
- Lihat kandidat rute dan best-so-far.
- Bandingkan overlay beberapa algoritma jika data tersedia.

---

## 10. Konfigurasi

Konfigurasi utama ada di:

```text
settings.py
```

Bagian penting:

```python
PLACE = "Surabaya, Indonesia"
BBOX = (-7.3545, 112.6085, -7.1975, 112.8273)
NETWORK_TYPE = "drive"
OSM_TIMEOUT = 300
OSM_USE_CACHE = True
N_SCENARIOS = 2
N_COVERAGE_SAMPLES = 100
MAX_FACILITIES_PER_CAT = 50
```

Penjelasan:

| Setting | Fungsi |
|---|---|
| `PLACE` | Nama area untuk query OpenStreetMap |
| `BBOX` | Bounding box fallback jika query place gagal |
| `NETWORK_TYPE` | Jenis jaringan jalan: `drive`, `walk`, `bike`, atau `all` |
| `OSM_TIMEOUT` | Batas waktu query Overpass |
| `OSM_USE_CACHE` | Menggunakan cache agar rerun lebih cepat |
| `N_COVERAGE_SAMPLES` | Jumlah titik acak untuk demo coverage |
| `MAX_FACILITIES_PER_CAT` | Batas jumlah fasilitas per kategori |

Jika mengganti kota atau area, jalankan ulang:

```bash
python main.py extract
```

---

## 11. Struktur Project

```text
TugasSCRouting/
|
|-- main.py
|-- settings.py
|-- requirements.txt
|-- environment.yml
|-- install.bat
|-- run.bat
|-- README.md
|-- CLAUDE.md
|
|-- src/
|   |-- extract.py
|   |-- explore.py
|   |
|   |-- routing/
|       |-- base.py
|       |-- algorithms.py
|       |-- benchmark.py
|       |-- demos.py
|       |-- visualize.py
|       |-- evolve_viz.py
|
|-- data/
|   |-- road_network.graphml
|   |-- road_network_nodes.geojson
|   |-- road_network_edges.geojson
|   |-- facilities_with_network.geojson
|   |-- facilities_with_network.csv
|   |-- comparison_results.csv
|   |-- comparison_summary.csv
|   |-- comparison_chart.png
|   |-- comparison_map_*.html
|   |-- evolution_viewer.html
|
|-- logs/
|   |-- run_*.log
|   |-- evolution_*.txt
|
|-- cache/
```

Penjelasan file utama:

| File | Fungsi |
|---|---|
| `main.py` | Entry point command line |
| `settings.py` | Konfigurasi project |
| `src/extract.py` | Download dan proses data OSM |
| `src/explore.py` | Profil data dan peta fasilitas |
| `src/routing/base.py` | Dataclass dan kontrak algoritma |
| `src/routing/algorithms.py` | Implementasi semua algoritma |
| `src/routing/benchmark.py` | Registry, skenario, runner benchmark |
| `src/routing/demos.py` | Demo routing dasar |
| `src/routing/visualize.py` | Peta dan chart hasil benchmark |
| `src/routing/evolve_viz.py` | Viewer evolusi generasi |

---

## 12. Menambah atau Mengubah Algoritma

Semua algoritma ada di:

```text
src/routing/algorithms.py
```

Minimal sebuah algoritma harus:

1. Subclass `BaseRoutingAlgorithm`.
2. Memiliki `name`.
3. Memiliki `description`.
4. Mengimplementasikan `find_route()`.
5. Mengembalikan `RouteResult`.

Contoh sederhana:

```python
class MyRoutingAlgorithm(BaseRoutingAlgorithm):
    name = "my_algo"
    description = "Algoritma routing percobaan"

    def find_route(self, G, source_node, target_node, scenario_name=""):
        start = time.perf_counter()

        route = nx.shortest_path(
            G,
            source_node,
            target_node,
            weight="travel_time",
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return RouteResult.build(
            G,
            self.name,
            scenario_name,
            source_node,
            target_node,
            route,
            elapsed_ms,
            metadata={"strategy": "shortest_path_travel_time"},
        )
```

Daftarkan algoritma di `src/routing/benchmark.py`:

```python
registry.register(MyRoutingAlgorithm())
```

Lalu jalankan:

```bash
python main.py compare
```

### Multi-stop Khusus

Jika algoritma ingin mengatur multi-stop sendiri seperti GA, tambahkan method:

```python
def _route_multi_stop(self, G, nodes, scenario_name="", source_node=None, target_node=None):
    ...
```

Jika method ini ada, benchmark akan memberikan seluruh daftar stop langsung ke algoritma tersebut.

Jika method ini tidak ada, benchmark akan:

1. Memilih order stop.
2. Memanggil `find_route()` per leg.
3. Menyambung semua leg menjadi satu rute.

Dengan begitu algoritma point-to-point tetap bisa ikut benchmark multi-stop.

---

## 13. Troubleshooting

### Error: data not found

Penyebab:

- File hasil ekstraksi belum ada.

Solusi:

```bash
python main.py extract
```

### Extract lama atau timeout

Penyebab:

- Query OpenStreetMap besar.
- Koneksi internet lambat.
- Overpass API sedang padat.

Solusi:

- Jalankan ulang beberapa saat lagi.
- Pastikan `OSM_USE_CACHE=True`.
- Perkecil area dengan mengubah `BBOX`.

### Compare lambat

Penyebab:

- Graph jalan besar.
- Skenario memiliki banyak stop.
- Algoritma metaheuristic butuh banyak iterasi.

Solusi:

```bash
python main.py compare --parallel-legs
```

Atau turunkan parameter algoritma di `src/routing/algorithms.py`.

### Peta HTML tidak terbuka

Solusi:

- Buka file `.html` langsung di browser.
- Gunakan Chrome, Edge, atau Firefox.
- Pastikan file ada di folder `data/`.

### Hasil antar run berbeda

Beberapa algoritma memakai randomness. Jika ingin hasil lebih konsisten, cek `RANDOM_SEED` pada class algoritma di `src/routing/algorithms.py`.

### Ingin reset hasil output

Output benchmark, peta, chart, dan log bisa dibuat ulang dengan menjalankan:

```bash
python main.py compare
```

Untuk membangun ulang semua dari awal:

```bash
python main.py all
```

---

## Ringkasan Cepat

Untuk menjalankan project dari awal:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py extract
python main.py explore
python main.py demo
python main.py compare
```

Untuk melihat hasil:

```text
data/comparison_results.csv
data/comparison_summary.csv
data/comparison_chart.png
data/comparison_map_*.html
data/evolution_viewer.html
```

Untuk mengubah model:

```text
src/routing/algorithms.py
```

Untuk mengubah area, kategori, atau parameter umum:

```text
settings.py
```
