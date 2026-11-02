# Laporan Bulanan Pengembangan Aplikasi

## Periode Juni 2026

## Pembahasan

### Ringkasan Aktivitas Bulanan

Pada bulan Juni 2026, kegiatan pengembangan difokuskan pada dua lini utama: pemutakhiran platform mobile (aplikasi Atos Pamor) dan realisasi pengembangan sistem dashboard operasional berbasis deteksi plat nomor kendaraan (ALPR). 

Pada aplikasi **Atos Pamor**, fokus utamanya adalah memastikan stabilitas aplikasi masa depan melalui migrasi besar-besaran ke Flutter 3.44.2 dan Dart 3.12.2, serta standarisasi berbagai komponen inti seperti sinkronisasi data reaktif pada modul NJAB, dan integrasi sistem debugging (API & Error Logger) internal yang dirancang khusus untuk mempermudah operasional pengembangan aplikasi.

Sementara itu, pada proyek eksperimental **Plate Recognition**, konsep prototipe dari bulan Mei telah diwujudkan menjadi ekosistem *fullstack application* yang utuh dan terkontainerisasi. Sistem ini kini mencakup antarmuka web, backend pemrosesan video asinkron, hingga integrasi pengecekan pajak, yang semuanya dikelola dalam satu orkestrasi infrastruktur Docker.

---

### Aktivitas yang Dilakukan

### 1. Pengembangan Ekosistem Sistem Deteksi Plat Nomor & Tax Checker (ALPR)
Melanjutkan inisiasi prototipe pada bulan Mei, pada periode ini sistem telah direalisasikan menjadi ekosistem fullstack yang siap untuk tahapan uji coba operasional (di bawah direktori `detection/`). 

**Infrastruktur & Arsitektur Fullstack:**
- **Frontend (Next.js 14 & Tailwind CSS):** Pembangunan antarmuka operasional yang mencakup halaman autentikasi, dashboard metrik, manajemen video (upload dengan *drag-and-drop*), dan tabel detail deteksi yang dilengkapi *progress bar* real-time (melalui WebSocket) serta *badge* status pajak.
- **API Backend (FastAPI):** Penyediaan REST API dan WebSocket gateway untuk menangani manajemen pengguna (JWT Auth), video, dan distribusi hasil deteksi.
- **Asynchronous Task Queue (Celery & Redis):** Mengisolasi beban berat pemrosesan AI dari API utama. Pemrosesan video dilakukan di latar belakang secara asinkron.
- **Storage & Database:** Implementasi PostgreSQL (via SQLAlchemy) untuk manajemen relasi data pengguna dan video, serta *self-hosted* MinIO (S3-compatible) untuk menyimpan file video asli dan potongan gambar plat nomor hasil deteksi secara aman (*presigned URL*).

**Pipeline Deteksi Cerdas (AI & Processing):**
- **Smart Frame Sampling:** Video yang diunggah diekstrak framenya, lalu di-filter secara otomatis menggunakan perhitungan *Laplacian Variance* untuk membuang frame yang buram (blur).
- **ALPR Engine:** Menjalankan model `fast-alpr` yang memadukan YOLOv9 (detektor) dan *fast-plate-ocr* untuk mengenali plat nomor secara akurat, disertai normalisasi teks dan deduplikasi data tangkapan.
- **Integrasi API Pajak:** Plat nomor yang telah di-deduplikasi diteruskan ke modul pengecekan pajak Bapenda Jabar untuk mengekstrak status "ACTIVE", "EXPIRED", atau "NOT_FOUND".

**Deployment & CI/CD:**
- Kontainerisasi menyeluruh seluruh *services* menggunakan `docker-compose`, dilengkapi dengan Nginx sebagai reverse proxy.
- Otomasi pengujian (*linting* dan integrasi) setiap kali *push* dilakukan melalui GitHub Actions.

---

### 2. Upaya Pemutakhiran Ekosistem Atos Pamor (Flutter 3.44.2 Migration)
Untuk menjaga kompatibilitas aplikasi terhadap standar terbaru Android dan *environment* pengembangan Flutter, saat ini sedang dilakukan tahapan migrasi dependensi skala besar. Proses ini merupakan inisiatif berkelanjutan dan masih memerlukan serangkaian pengujian (*verification phase*) lebih lanjut sebelum dapat dinyatakan stabil seutuhnya untuk produksi. 

Adapun beberapa cakupan pekerjaan *upgrade* yang telah diinisiasi pada bulan ini meliputi:

- **Upgrade Lingkungan Build & Arsitektur Android:** Menyesuaikan konfigurasi tingkat proyek untuk menargetkan Android SDK versi 35 (Android 15). Ini mencakup pembaruan krusial pada level sistem *build*, yaitu peningkatan versi *Gradle* ke versi 8.9 serta *Android Gradle Plugin* (AGP) ke versi 8.7.3 yang diiringi dengan penyesuaian kompatibilitas Java ke versi 17 untuk mendukung optimasi kompilasi terbaru.
- **Modernisasi Widget UI dan Manajemen Transisi Layar:** Melakukan transisi fungsi navigasi dari `PopScope` lawas beralih sepenuhnya ke arsitektur `onPopInvokedWithResult` untuk menyesuaikan standar rilis API stabil Flutter terbaru. Selain itu, terdapat perombakan struktur format *styling* gambar vektor, yakni migrasi dari sintaks `SvgPicture.color` yang sudah usang (*deprecated*) menjadi implementasi `ColorFilter.mode(..., BlendMode.srcIn)` secara komprehensif di seluruh modul antarmuka.
- **Resolusi Deprecations pada Third-Party Libraries:** Upaya migrasi juga berdampak pada berbagai komponen eksternal. Kami telah membuang dependensi *pull-to-refresh* pihak ketiga yang sudah tidak dipelihara lagi (*abandonware*) dan menyusun ulang kodenya untuk menggunakan komponen *native* bawaan Flutter, yakni `RefreshIndicator`. Library besar lainnya yang mengalami *breaking changes* dan memerlukan intervensi penulisan ulang API adalah `map_launcher` (yang telah bermigrasi melampaui v3+) serta pembaruan logika *tooltip API* pada pustaka pembuat grafik `fl_chart`. Selain itu, kami juga terus mengevaluasi *compatibility issue* dari paket *core* lainnya seperti `flutter_map` (v7) dan `device_info_plus` (v11).
- **Pembersihan Log Analisis Statik:** Mengatasi peringatan-peringatan *deprecation* pada kode internal, termasuk melakukan mitigasi pada fungsi-fungsi `RegExp` di utilitas pemformatan *string* agar tetap sejalan dengan protokol keamanan analisis kode.

Proses migrasi ini masih akan terus berjalan karena banyaknya *library* yang terdampak, sehingga tim akan melakukan tahap regresi (*regression testing*) secara menyeluruh pada sprint berikutnya.

---

### 3. Optimalisasi Manajemen Data & UI Konsistensi (Atos Pamor)
Melakukan perbaikan fungsionalitas dan reaktivitas status secara konsisten, agar versi mobile bekerja persis seperti web desktop.

- **Sinkronisasi Reaktif NJAB:** Penyelesaian bug kritis pada dialog penambahan data NJAB versi seluler. Sinkronisasi *state* antar *widget* saat penambahan/perubahan (CRUD) data distandarisasi menggunakan metode callback (`onTableRefresh`), menggantikan pengelolaan state lama yang tidak stabil sehingga UI dapat melakukan pengambilan data otomatis pasca-transaksi.
- **Error Handling Dashboard:** Penanganan *exception* seperti error parsing numerik (`Invalid radix-10`) di atasi dengan parsing angka aman (*safe-double conversion*) dan penggunaan metode formatter standar desimal Indonesia melalui utilitas `StringUtils.separatorNumber`. 
- **Pengamanan BuildContext:** Menyelesaikan masalah *LateInitializationError* pada layer controller saat aplikasi mengalami pergantian status antarmuka sebelum fungsi asinkron rampung, dengan menambahkan pengecekan `mounted` pada UI/Login controller.

---

### 4. Implementasi Sistem Debugging Kustom Terpusat (Atos Pamor)
Memperkuat kemudahan dalam pemeliharaan (maintenance) aplikasi dengan membangun komponen analitik dan *debugging* yang berjalan secara *real-time*. 

- **Pembangunan Modul DebugApiLogger:** Mengembangkan sistem deteksi *bug* mandiri yang diintegrasikan langsung ke dalam kerangka kerja aplikasi Atos Pamor, yang berfungsi layaknya inspektur jaringan tersembunyi.
  - **Antarmuka Overlay Inspector:** Membuat *floating button* dinamis yang dapat digeser (draggable) yang membuka panel inspeksi layar penuh. Desain antarmuka (UI) dibangun agar *seamless* dan mematuhi aturan sistem desain warna internal Atos Pamor (palet *blueGray* dan *blue*), sehingga tidak terasa seperti aplikasi pihak ketiga.
  - **Dio Interceptors Kustom:** Mengubah alur *middleware* HTTP (Dio) agar seluruh siklus jaringan—mulai dari *request*, *response*, hingga *error*—ditangkap dan dikirim ke *singleton manager*. Fitur ini mempermudah pelacakan beban *payload* tanpa harus menghubungkan perangkat keras ke komputer.
  - **Global Error Logger Service:** Menerapkan pengawas *error* secara hierarkis yang mengaitkan diri ke `FlutterError.onError` dan `PlatformDispatcher.instance.onError`. Hal ini memastikan bahwa *exception* UI maupun kegagalan kode asinkron (Dart) langsung ditangkap, dikelola dalam sebuah *queue* (antrean), dan dicatat untuk keperluan forensik kode di masa mendatang.
  - **Manajemen State Berbasis Hive:** Menghubungkan fitur visibilitas *debugger* ini ke dalam penyimpanan memori persisten (`mainStorage` Hive), sehingga preferensi aktif/nonaktif fitur pelacakan dapat dipertahankan di berbagai sesi penggunaan oleh pengembang yang memiliki hak akses (*Role ID Admin*).

---

### Capaian Bulan Juni
 
 1. Tuntasnya pembangunan sistem "Vehicle License Plate Detection & Tax Checker" berbasis dashboard web *fullstack*, siap di-deploy melalui Docker ke lingkungan produksi/staging.
 2. Dimulainya inisiasi migrasi proyek Atos Pamor ke Flutter versi 3.44.2 (Dart 3.12.2) secara besar-besaran, yang mencakup modernisasi puluhan komponen *deprecated* dan penyesuaian lingkungan *build* ke Android SDK 35 (proses saat ini sedang dalam fase verifikasi).
 3. Terselesaikannya integrasi fitur "Debug / API Inspector" kustom yang persisten antar sesi aplikasi yang memungkinkan tim pengembang memantau lalu lintas API secara visual tanpa IDE debugger.
 4. Perbaikan siklus UX (konsistensi state reaktif & *error boundary*) pada modul pengelolaan NJAB seluler dan laporan dashboard.

---

### Rencana Lanjutan Juli

- Menyusun skenario pengujian lapangan untuk sistem prototipe pendeteksi plat nomor kendaraan (Video Detection).
- Memastikan performa inferensi deteksi plat dari Celery Worker stabil saat dibebani beberapa tugas pemrosesan video beresolusi tinggi.
- Evaluasi integrasi backend sistem *plate detection* dengan perangkat keras (dashcam dan mini-server/laptop) dalam unit operasi lapangan sebenarnya.
- Observasi stabilitas versi *Atos Pamor* terbaru (Flutter 3.44.2) pada lingkungan pengguna aktif untuk mendeteksi kemungkinan perlunya rilis tambalan.
