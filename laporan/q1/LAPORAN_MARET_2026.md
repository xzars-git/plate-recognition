# Laporan Bulanan Pengembangan Aplikasi

## Periode Maret 2026

## Pembahasan

### Ringkasan Aktivitas Bulanan

Pada bulan Maret 2026, kegiatan pengembangan difokuskan pada pematangan hasil eksperimen model AI OCR dan peningkatan efektivitas labeling tools untuk mendukung kesiapan implementasi operasional. Fokus ini dijalankan bersamaan dengan pemeliharaan stabilitas aplikasi Panah Pasopati agar transformasi teknologi tetap sejalan dengan kebutuhan layanan harian pengguna.

Secara garis besar, pekerjaan Maret diarahkan pada penguatan performa model pada data menantang, penyempurnaan quality control anotasi, validasi integrasi teknis, serta penyiapan rekomendasi rollout bertahap yang aman.

### Aktivitas yang Dilakukan

### 1. Optimasi Lanjutan Fitur Aplikasi dan Reliabilitas Sistem

Pada periode ini dilakukan penyempurnaan terhadap modul-modul inti untuk menjaga kualitas layanan saat pengembangan AI semakin intensif.

Pekerjaan yang dilakukan meliputi:

- Penyempurnaan alur penelusuran dan penguatan penanganan error pada kondisi response tidak lengkap.
- Penyesuaian proses caching dan refresh data agar konsistensi tampilan hasil tetap terjaga.
- Perapihan komponen UI untuk meningkatkan kejelasan informasi status proses kepada pengguna.
- Peningkatan kualitas logging untuk memudahkan investigasi issue lintas modul.

Hasil utama:

- Stabilitas operasional tetap terjaga selama fase validasi AI.
- Proses troubleshooting menjadi lebih cepat berkat struktur log yang lebih informatif.
- Pengalaman pengguna tetap konsisten pada alur utama aplikasi.

### 2. Pengembangan Model AI OCR (Tahap Validasi dan Tuning)

Kegiatan AI di Maret berfokus pada peningkatan akurasi dan konsistensi hasil inferensi melalui tuning terarah.

Ruang lingkup pekerjaan:

- Evaluasi model kandidat terhadap baseline pada beberapa kategori data sulit.
- Penyempurnaan konfigurasi training dan parameter inferensi untuk menekan false detection.
- Analisis error per kategori kasus untuk menentukan prioritas perbaikan model.
- Pengukuran performa berulang menggunakan metrik yang sama agar tren peningkatan dapat dipantau.

Progres model AI:

- Candidate model menunjukkan peningkatan kestabilan deteksi pada sebagian besar skenario internal.
- Gap performa pada kondisi ekstrem mulai berkurang melalui tuning bertahap.
- Tim memperoleh peta prioritas perbaikan berbasis bukti evaluasi.

Catatan:
Status Maret masih pada tahap validasi internal dan belum menjadi implementasi produksi penuh.

### 3. Penguatan Labeling Tools dan Tata Kelola Data Anotasi

Pada Maret, labeling tools ditingkatkan agar mendukung siklus eksperimen yang lebih cepat dan terukur.

Komponen yang disempurnakan:

- Pengelolaan antrian labeling berdasarkan prioritas kebutuhan eksperimen model.
- Validasi anotasi yang lebih ketat sebelum data dinyatakan siap training.
- Pencatatan audit trail terhadap revisi label agar histori perubahan dapat ditelusuri.
- Peningkatan alur export dataset untuk meminimalkan potensi mismatch format.

Quality control yang diterapkan:

- Reviewer check untuk sampel berisiko tinggi.
- Random sampling audit untuk memonitor konsistensi antar labeler.
- Kalibrasi berkala terhadap guideline anotasi agar standar tetap seragam.

Hasil utama:

- Kualitas dataset meningkat dan lebih siap digunakan untuk eksperimen lanjutan.
- Waktu siklus dari labeling ke training menjadi lebih ringkas.
- Tingkat konsistensi anotasi antar batch membaik.

### 4. Validasi Integrasi AI dan Kesiapan Rollout Bertahap

Sepanjang Maret dilakukan validasi teknis agar komponen AI dapat diintegrasikan ke alur aplikasi secara terkontrol.

Aktivitas yang dilakukan:

- Uji alur inference pada lingkungan internal dengan skenario penggunaan representatif.
- Penyesuaian threshold confidence untuk menyeimbangkan akurasi dan reliabilitas hasil.
- Penerapan fallback flow pada kondisi hasil OCR belum memenuhi ambang kualitas.
- Monitoring hasil uji integrasi untuk mengidentifikasi potensi dampak terhadap modul lain.

Hasil utama:

- Integrasi teknis menunjukkan kesiapan untuk tahap limited rollout.
- Risiko regresi pada fitur utama dapat dikelola melalui pendekatan bertahap.
- Tersusun rekomendasi implementasi berdasarkan hasil uji internal.

### 5. Pengujian Menyeluruh dan Quality Assurance

Untuk memastikan kesiapan transisi ke tahap berikutnya, pengujian dilakukan secara menyeluruh pada komponen aplikasi dan AI.

Aktivitas QA:

- Pengujian regresi fungsional pada alur login, telusur, detail, riwayat, dan pelaporan.
- Uji konsistensi output OCR pada batch data internal dengan variasi kualitas gambar.
- Validasi kualitas hasil anotasi setelah perbaikan guideline labeling.
- Pengujian reliabilitas sistem pada sesi penggunaan berulang dan skenario beban menengah.

Hasil utama:

- Alur kritikal tetap stabil selama fase validasi AI.
- Kualitas output OCR menunjukkan tren peningkatan pada skenario mayoritas.
- Kesiapan teknis untuk pengujian terbatas tahap lanjut dinilai memadai.

### 6. Capaian Bulan Maret

Capaian yang dapat dilaporkan pada Maret 2026:

- Terselesaikannya fase tuning dan validasi lanjutan model AI berbasis evaluasi terstruktur.
- Meningkatnya kematangan labeling tools dari sisi workflow, QA, dan auditability.
- Terbentuknya rekomendasi threshold dan fallback untuk integrasi AI yang aman.
- Terjaganya stabilitas aplikasi selama proses peningkatan teknologi berjalan.
- Tersusunnya readiness plan untuk pelaksanaan limited rollout pada tahap berikutnya.

### 7. Kendala dan Mitigasi

Dalam pelaksanaan Maret, terdapat beberapa tantangan:

- Data ekstrem masih memerlukan penanganan khusus agar performa model lebih konsisten.
- Kebutuhan kecepatan labeling tinggi berpotensi memengaruhi kualitas jika tanpa kontrol ketat.
- Integrasi lintas modul membutuhkan sinkronisasi berkelanjutan antar tim.

Mitigasi yang diterapkan:

- Prioritisasi data sulit sebagai fokus tuning berikutnya.
- Penerapan QA berlapis dan audit sampling terjadwal pada proses anotasi.
- Penguatan koordinasi teknis dan review berkala lintas fungsi.

### 8. Rencana Lanjutan April

Rencana kerja lanjutan yang disiapkan:

- Menjalankan limited rollout komponen AI pada skenario terkontrol.
- Melanjutkan optimasi model berdasarkan feedback hasil rollout terbatas.
- Memperluas cakupan data labeling untuk meningkatkan generalisasi model.
- Menyempurnakan dashboard monitoring performa model dan kualitas data.
- Menyiapkan dokumen evaluasi untuk keputusan implementasi produksi bertahap.
