# Laporan Bulanan Pengembangan Aplikasi

## Periode Mei 2026

## Pembahasan

### Ringkasan Aktivitas Bulanan

Pada bulan Mei 2026, kegiatan pengembangan difokuskan pada inisiasi pembuatan **prototipe demo sistem deteksi plat nomor kendaraan berbasis video**, yang dirancang dengan konsep dashboard operasional **Mobil Samsat Keliling**. Prototipe ini merupakan langkah awal eksplorasi teknis sebelum menentukan arsitektur dan perangkat deployment yang akan digunakan pada implementasi sesungguhnya.

Secara garis besar, prototipe ini memvisualisasikan skenario di mana sebuah unit Samsat Keliling dilengkapi dua buah dashcam. Rekaman video dari dashcam tersebut diunggah ke sistem dan dianalisis secara otomatis untuk mendeteksi serta membaca plat nomor kendaraan, kemudian hasilnya diintegrasikan dengan pengecekan status pajak kendaraan secara real-time. Pengembangan demo ini menggunakan project **Panah Pasopati** sebagai referensi teknologi vision berbasis mobile, dengan adaptasi pada sisi arsitektur agar sesuai dengan kebutuhan pemrosesan video secara batch.

Aktivitas Mei juga mencakup refactor menyeluruh terhadap struktur direktori project **plate-recognition** — yang merupakan project AI model dari aplikasi Panah Pasopati — dari susunan flat yang sulit dikelola menjadi arsitektur berlapis yang lebih terstruktur dan siap untuk pengembangan multi-model ke depannya.

---

### Aktivitas yang Dilakukan

### 1. Inisiasi Prototipe Demo — Sistem Deteksi Plat Nomor Dashcam Samsat Keliling

Pada periode ini dimulai pembuatan prototipe demo yang mensimulasikan skenario operasional unit Samsat Keliling yang dilengkapi teknologi vision untuk mendeteksi plat nomor kendaraan dari rekaman video dashcam.

**Latar Belakang dan Konsep:**

Konsep ini muncul dari kebutuhan untuk mengoptimalkan operasional Samsat Keliling dengan memanfaatkan teknologi pengenalan plat nomor secara otomatis. Dalam skenario operasional, sebuah unit kendaraan Samsat Keliling akan beroperasi di lapangan dengan dua dashcam yang merekam kondisi lalu lintas di sekitarnya. Rekaman video kemudian dianalisis oleh sistem untuk mengidentifikasi plat nomor kendaraan yang terekam dan mencocokkannya dengan data status pajak.

Referensi utama prototipe ini adalah project **Panah Pasopati**, sebuah aplikasi mobile yang memiliki kemampuan vision untuk mengenali plat nomor secara real-time. Teknologi inti dari Panah Pasopati diadaptasi ke dalam konteks pemrosesan video berbasis batch, sehingga rekaman dari dashcam dapat dianalisis setelah proses perekaman selesai.

**Ruang Lingkup Prototipe:**

- Pembuatan antarmuka web untuk manajemen video dari dashcam (upload, list, dan monitoring status pemrosesan).
- Implementasi pipeline pemrosesan video: frame sampling, deteksi plat nomor, pembacaan teks OCR, dan pengecekan status pajak.
- Visualisasi hasil deteksi dalam bentuk tabel terstruktur yang menampilkan foto crop kendaraan, nomor plat terdeteksi, confidence score, dan status pajak.
- Integrasi WebSocket untuk monitoring progress pemrosesan video secara real-time.
- Fitur ekspor hasil deteksi dalam format CSV untuk keperluan pelaporan.
- Implementasi antrian review untuk hasil deteksi dengan confidence rendah agar dapat diverifikasi secara manual oleh operator.

**Hasil Awal Prototipe:**

- Dashboard operasional berhasil memvisualisasikan alur kerja dari upload video hingga tampilnya hasil deteksi plat nomor beserta status pajaknya.
- Pipeline deteksi menggunakan model **YOLOv9** untuk deteksi plat dan **fast-plate-ocr** untuk pembacaan teks, terintegrasi melalui library `fast-alpr`.
- Sistem confidence scoring menggunakan rata-rata geometrik antara skor deteksi dan skor OCR, dengan threshold terpisah untuk hasil yang langsung diterima (>=80%) dan yang masuk antrian review (40-80%).
- Validasi format plat nomor Indonesia diterapkan untuk menyaring hasil deteksi yang tidak relevan.
- Demo berhasil menampilkan skenario tipikal: video diproses, plat terdeteksi, status pajak ditampilkan (Patuh / Tidak Ditemukan).

**Catatan Penting:**

Status bulan Mei masih merupakan tahap **demo dan eksplorasi teknis**. Keputusan mengenai arsitektur final dan perangkat deployment (termasuk spesifikasi dashcam, hardware unit Samsat Keliling, dan infrastruktur server) masih dalam tahap pertimbangan dan akan ditentukan seiring dengan perkembangan eksperimen selanjutnya.

---

### 2. Pembangunan Fondasi Arsitektur Teknis Fullstack

Untuk mendukung prototipe demo, dibangun fondasi arsitektur teknis yang memisahkan tanggung jawab antar komponen secara jelas, sehingga masing-masing lapisan dapat dikembangkan dan dioptimalkan secara independen ke depannya.

**Stack Teknologi yang Digunakan:**

| Layer | Teknologi |
|---|---|
| Frontend | Next.js 14 (App Router) + Tailwind CSS + shadcn/ui |
| API Backend | FastAPI (Python 3.11) |
| Task Queue | Celery 5 + Redis 7 |
| Object Storage | MinIO (self-hosted, S3-compatible) |
| Database | PostgreSQL 16 + SQLAlchemy (ORM) |
| AI — ALPR | fast-alpr (YOLOv9 detection + fast-plate-ocr) |
| Auth | JWT (python-jose) + bcrypt |
| Real-time | WebSocket (FastAPI native) |
| Container | Docker + Docker Compose |

**Komponen yang Dibangun:**

- **Backend (FastAPI):** API endpoint untuk autentikasi, manajemen video, hasil deteksi, statistik dashboard, antrian review, dan manajemen pengguna. Dilengkapi middleware untuk rate limiting, request ID, dan CORS.
- **Worker (Celery):** Task asinkron untuk pemrosesan video secara background, meliputi frame sampling dengan Laplacian variance untuk menyaring frame buram, inferensi ALPR, deduplication hasil deteksi, dan pengambilan data pajak.
- **Frontend (Next.js):** Dashboard dengan halaman upload video, daftar video dengan status badge, detail video dengan tabel deteksi, dan halaman review plat dengan confidence rendah.
- **Infrastruktur (Docker Compose):** Semua service (PostgreSQL, Redis, MinIO, backend, Celery worker, frontend) terkontainerisasi dan dapat dijalankan dalam satu perintah.

**Capaian Teknis:**

- Sistem berhasil berjalan end-to-end dari upload video hingga tampilnya hasil deteksi beserta status pajak.
- Health check endpoint memastikan semua service dependencies (database, Redis, MinIO) berjalan dengan baik sebelum menerima request.
- Superadmin seeding otomatis saat pertama kali sistem dijalankan.
- Presigned URL diterapkan untuk akses objek di MinIO agar tidak mengekspos storage secara publik.

---

### 3. Refactor Struktur Folder dan Kode Project plate-recognition (AI Model Panah Pasopati)

Di luar pengerjaan demo Samsat Keliling, pada periode ini juga dilakukan refactor menyeluruh terhadap project **plate-recognition** — yaitu project pengembangan model AI dari aplikasi **Panah Pasopati** yang menjadi referensi teknologi vision pada poin sebelumnya. Refactor ini bertujuan mentransformasi struktur direktori dari susunan flat yang tidak terorganisir menjadi arsitektur berlapis yang lebih production-ready dan mudah dikembangkan oleh tim.

**Latar Belakang Refactor:**

Project plate-recognition merupakan repositori pengembangan model AI yang mendukung kemampuan vision pada aplikasi Panah Pasopati. Seiring bertambahnya jumlah model yang dikembangkan (plate detector, color classifier, OCR custom, dan anti-spoofing), struktur project yang sebelumnya bersifat flat — di mana seluruh script training, model, dataset, dan konfigurasi tersebar tanpa pemisahan yang jelas — mulai menimbulkan masalah: sulit menemukan file yang relevan, tidak ada pemisahan tanggung jawab antar komponen, kode preprocessing diulang berulang kali untuk setiap model yang berbeda, dan onboarding anggota tim baru menjadi tidak efisien.

**Sebelum Refactor — Struktur Flat:**

```
plate-recognition/
├── label_tool.py
├── train_color_classifier.py
├── train_native_rotation.py
├── dataset/
│   ├── plate_detection_yolo/
│   └── plate_colors/
├── runs/
└── models/
    ├── best.pt
    └── yolo11n.pt
```

**Setelah Refactor — Arsitektur Hybrid "Mobile-First MLOps":**

```
plate-recognition/
└── teman_pamor_anpr/          <- Project terstruktur
    ├── 00_platform/           <- Shared infrastructure (validasi, preprocessing, model registry)
    ├── 01_data_platform/      <- Manajemen data (datasets, labeling tools, notebooks)
    ├── 02_training_platform/  <- Pipeline training 4 model AI
    │   ├── 01_plate_detector_yolo/
    │   ├── 02_color_classifier/
    │   ├── 03_ocr_custom/
    │   └── 04_anti_spoofing/
    ├── 03_deployment_platform/<- Mobile deployment (konversi TFLite, inferensi, monitoring)
    ├── 04_ci_cd/              <- Otomasi pipeline
    ├── docs/                  <- Architecture Decision Record + panduan
    └── tests/                 <- Testing suites
```

**Komponen Platform yang Dibuat:**

- **`00_platform/data_validation/`:** Utilitas validasi kualitas gambar (cek resolusi, korupsi file, ukuran) yang digunakan bersama oleh semua pipeline model.
- **`00_platform/preprocessing/`:** Fungsi preprocessing gambar (normalisasi, resize dengan aspect ratio, konversi color space) yang terpusat agar tidak ada duplikasi kode antar model.
- **`00_platform/model_registry/`:** Registry JSON untuk melacak versi, metrik akurasi, dan status deployment setiap model AI (plate detector, color classifier, OCR custom, anti-spoofing).
- **`01_data_platform/labeling_tools/`:** Tool pelabelan data yang dipindahkan dari root project ke dalam struktur yang lebih terorganisir.
- **`02_training_platform/`:** Setiap dari 4 model AI mendapatkan direktori training mandiri dengan konfigurasi, source code, dan hasil yang terisolasi.

**Dampak Refactor:**

- Pembersihan storage sebesar **9.72 GB** (35% pengurangan) dari file duplikat, backup lama, dan artefak sementara yang tidak lagi diperlukan.
- Migrasi **87 direktori** dan **26 file** dengan tingkat keberhasilan 100% tanpa kehilangan data.
- Dokumentasi teknis sebesar **2.500+ baris** dihasilkan meliputi Architecture Decision Record (ADR), panduan getting started, dan README untuk setiap komponen.
- Struktur baru memungkinkan pengelolaan **4 model AI sekaligus** (plate detector, color classifier, OCR custom, anti-spoofing) dengan pipeline yang jelas dan tidak saling mengganggu.

**Hasil Utama:**

- Alur kerja data hingga deployment menjadi lebih mudah dipahami: `data platform → training platform → deployment platform`.
- Shared utilities menghilangkan duplikasi kode preprocessing yang sebelumnya diulang di setiap script training.
- Model registry memberikan visibilitas terpusat terhadap status dan versi setiap model yang sedang dikembangkan.
- Project siap untuk pengembangan lanjutan dengan pendekatan yang lebih terstruktur dan dapat diaudit.

---

### 4. Capaian Bulan Mei

Capaian yang dapat dilaporkan pada Mei 2026:

- Terselesaikannya prototipe demo sistem deteksi plat nomor kendaraan berbasis video dengan konsep dashboard Samsat Keliling.
- Berjalannya pipeline end-to-end dari upload video dashcam, deteksi plat, pengecekan status pajak, hingga visualisasi hasil.
- Terbentuknya fondasi arsitektur fullstack yang terkontainerisasi dan dapat dikembangkan lebih lanjut.
- Terselesaikannya refactor menyeluruh project plate-recognition ke arsitektur Hybrid "Mobile-First MLOps" dengan pemisahan 5 lapisan platform yang jelas.
- Pembersihan 9.72 GB storage dari artefak lama serta migrasi 87 direktori ke struktur baru dengan tingkat keberhasilan 100%.
- Tersedianya fitur antrian review untuk penanganan hasil deteksi dengan confidence rendah.

---

### 5. Kendala dan Mitigasi

Dalam pelaksanaan Mei, terdapat beberapa tantangan:

- **Keputusan arsitektur dan deployment masih terbuka:** Spesifikasi dashcam, hardware kendaraan, dan infrastruktur server lapangan belum ditentukan, sehingga prototipe dirancang se-fleksibel mungkin tanpa bergantung pada hardware tertentu.
- **Variasi kualitas video rekaman:** Kondisi pencahayaan, sudut pengambilan, dan kualitas kamera dashcam sangat bervariasi dan memengaruhi akurasi deteksi. Mitigasi sementara dilakukan melalui threshold confidence yang terpisah antara deteksi otomatis dan antrian review manual.
- **Integrasi API pajak yang belum produksi:** Data status pajak pada prototipe masih menggunakan endpoint uji. Integrasi ke sistem pajak produksi akan dilakukan setelah arsitektur deployment final ditentukan.

Mitigasi yang diterapkan:

- Pendekatan kontainerisasi Docker memastikan prototipe mudah dipindahkan ke berbagai lingkungan deployment tanpa konfigurasi ulang yang besar.
- Sistem antrian review memastikan hasil deteksi dengan confidence rendah tidak langsung ditampilkan sebagai data valid, melainkan menunggu konfirmasi operator.
- Threshold dan parameter inferensi dibuat dapat dikonfigurasi melalui environment variable sehingga mudah disesuaikan tanpa perubahan kode.

---

### 6. Rencana Lanjutan Juni

Rencana kerja lanjutan yang disiapkan:

- Melakukan evaluasi dan pemilihan arsitektur deployment yang sesuai, termasuk pertimbangan spesifikasi hardware dashcam dan unit kendaraan.
- Mengembangkan skenario uji dengan rekaman video dari kondisi lapangan nyata untuk mengukur akurasi deteksi secara lebih representatif.
- Memperkuat pipeline pemrosesan dengan kemampuan menangani dua sumber video (dua dashcam) secara bersamaan dan mengasosiasikan hasilnya ke satu sesi operasional.
- Mengeksplorasi kemungkinan mode pemrosesan semi real-time untuk skenario di mana koneksi jaringan memungkinkan data langsung dikirim dari kendaraan ke server.
- Melanjutkan integrasi API pajak ke lingkungan yang lebih mendekati produksi.
