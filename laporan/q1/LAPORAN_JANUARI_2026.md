# Laporan Bulanan Pengembangan Aplikasi

## Periode Januari 2026

## Pembahasan

### Ringkasan Aktivitas Bulanan

Pada bulan Januari 2026, kegiatan pengembangan difokuskan pada dua arus pekerjaan utama, yaitu stabilisasi dan penyempurnaan aplikasi Panah Pasopati, serta inisiasi proyek pengembangan model AI untuk meningkatkan akurasi proses penelusuran kendaraan berbasis OCR. Pendekatan ini dilakukan agar aplikasi tetap berjalan stabil di sisi pengguna sekaligus menyiapkan fondasi inovasi fitur cerdas yang dapat diimplementasikan bertahap pada rilis berikutnya.

Secara umum, pekerjaan Januari diarahkan pada peningkatan kualitas layanan, penguatan reliabilitas proses deteksi, konsistensi branding aplikasi, serta persiapan pipeline teknis untuk pengembangan model AI yang lebih adaptif terhadap variasi data lapangan.

### Aktivitas yang Dilakukan

### 1. Stabilisasi dan Penyempurnaan Fitur Aplikasi

Pada periode ini dilakukan perbaikan berkelanjutan pada modul-modul inti untuk memastikan pengalaman pengguna tetap konsisten selama fase transisi pengembangan.

Pekerjaan yang dilakukan meliputi:

- Penyempurnaan alur penelusuran agar lebih robust pada kondisi jaringan tidak stabil.
- Perapihan proses validasi input untuk meminimalkan kegagalan request ke layanan backend.
- Peningkatan konsistensi komponen antarmuka agar tampilan antarlayar lebih seragam.
- Penyesuaian pesan error dan feedback pengguna agar lebih informatif serta mudah dipahami.
- Perbaikan minor bug pada proses filtering, rendering list, dan refresh data.

Hasil utama:

- Alur penggunaan harian menjadi lebih stabil.
- Penurunan potensi error karena input tidak valid.
- Pengalaman pengguna lebih rapi dan konsisten pada berbagai skenario penggunaan.

### 2. Maintenance Teknis dan Refactoring Terarah

Untuk menjaga maintainability, dilakukan refactoring pada bagian kode yang berdampak langsung terhadap kecepatan pengembangan berikutnya.

Ruang lingkup pekerjaan:

Penyederhanaan struktur fungsi dan pemisahan tanggung jawab logika agar lebih modular.
Perbaikan pola logging untuk mempermudah pelacakan issue di lingkungan pengujian.
Penataan ulang beberapa utilitas agar reusable dan tidak duplikasi.
Pembersihan kode lama yang sudah tidak relevan terhadap arsitektur terbaru.

Hasil utama:

- Struktur kode lebih mudah dirawat.
- Waktu analisis bug lebih cepat karena kualitas log meningkat.
- Risiko regresi berkurang saat menambahkan fitur lanjutan.

### 3. Penguatan Kualitas OCR dan Integrasi Data Pendukung

Perbaikan pada komponen OCR tetap dilanjutkan untuk mempertahankan performa deteksi di lapangan.

Aktivitas yang dilakukan:

- Penyesuaian state handling saat proses deteksi berjalan kontinu.
- Penyempurnaan mekanisme reset agar deteksi ulang tidak menimbulkan konflik state.
- Perbaikan response handling ketika hasil OCR kosong atau confidence rendah.
- Penyelarasan integrasi data pendukung seperti parameter lokasi dan metadata penelusuran.

Hasil utama:

- Proses OCR lebih stabil pada sesi penggunaan panjang.
- Penanganan edge case lebih baik pada kondisi hasil deteksi ambigu.
- Data penelusuran lebih lengkap untuk keperluan analitik lanjutan.

### 4. Pengembangan Model AI

Selain pengembangan aplikasi, pada Januari dimulai pekerjaan khusus untuk pengembangan model AI sebagai fondasi peningkatan akurasi deteksi di fase berikutnya.

Ruang lingkup inisiatif AI:

- Penyusunan tujuan model: peningkatan akurasi pembacaan plat, ketahanan terhadap variasi cahaya, sudut, dan kualitas kamera.
- Persiapan dataset awal dari sampel internal dan data uji lapangan terkurasi.
- Klasifikasi data berdasarkan kategori kondisi sulit, seperti blur, low light, occlusion, dan noise tinggi.
- Penyusunan skema eksperimen baseline untuk membandingkan performa model existing dan model kandidat.
- Penentuan metrik evaluasi awal seperti precision, recall, F1-score, serta waktu inferensi.
- Perancangan alur integrasi model agar kompatibel dengan kebutuhan aplikasi mobile.

Progres Januari untuk AI:

- Tahap persiapan data dan baseline eksperimen telah dimulai.
- Kerangka evaluasi performa model sudah ditetapkan.
- Pipeline kerja AI dan aplikasi sudah diselaraskan agar proses integrasi bisa dilakukan bertahap tanpa mengganggu kestabilan produksi.

Catatan penting:
Fase Januari berfokus pada fondasi riset dan validasi awal. Implementasi penuh ke produksi dijadwalkan bertahap setelah hasil evaluasi model memenuhi ambang kualitas yang ditentukan.

#### Penambahan Progres Lanjutan Pengembangan Model AI

Pada Januari 2026, pengembangan model AI OCR dilanjutkan dengan fokus pada perapihan fondasi teknis yang telah dibangun pada periode sebelumnya. Baseline model telah tersedia dan digunakan sebagai acuan utama untuk evaluasi ulang performa pada skenario data yang lebih terstruktur. Aktivitas utama mencakup peninjauan ulang metrik baseline, penyesuaian konfigurasi uji internal, serta penataan kembali pipeline data agar alur ingestion, validasi, dan pemisahan dataset berjalan lebih konsisten.

Output awal Januari meliputi baseline terverifikasi ulang, daftar isu kualitas data prioritas, dan perbaikan alur pipeline untuk mendukung siklus eksperimen berikutnya. Dampak dari kegiatan ini adalah meningkatnya kesiapan eksperimen 2026 karena proses evaluasi menjadi lebih terukur dan dapat direplikasi. Hingga akhir Januari, status pekerjaan masih pada tahap validasi internal dan belum dilakukan implementasi ke aplikasi mobile production.

#### Penambahan Progres Lanjutan Labeling Tools

Pada aspek labeling tools, Januari difokuskan pada penguatan kualitas anotasi dan konsistensi operasional proses pelabelan. Tim melanjutkan perangkat yang sudah ada melalui standardisasi ulang pedoman anotasi, perapihan struktur label, serta penyelarasan aturan quality check antar-annotator. Sejalan dengan itu, dilakukan pembersihan dan penandaan ulang sebagian sampel untuk mengurangi noise label dan meningkatkan reliabilitas data latih OCR.

Capaian utama bulan ini adalah meningkatnya keseragaman hasil anotasi, tersusunnya daftar validasi kualitas anotasi internal, dan meningkatnya kesiapan dataset berlabel untuk eksperimen lanjutan 2026. Secara status, pekerjaan tetap berada pada fase validasi internal dan belum diarahkan ke penggunaan produksi.

### 5. Pengujian, Validasi, dan Quality Control

Untuk memastikan perubahan aman diterapkan, dilakukan pengujian berkala pada skenario utama pengguna.

Aktivitas QA:

- Uji alur login, penelusuran, detail hasil, dan riwayat.
- Uji kondisi jaringan lambat, timeout, serta retry request.
- Uji perilaku aplikasi pada data kosong, data parsial, dan response tidak lengkap.
- Validasi tampilan pada variasi ukuran layar dan orientasi perangkat.

Hasil utama:

- Mayoritas alur utama berjalan sesuai ekspektasi.
- Isu kritikal yang ditemukan telah ditangani dalam siklus perbaikan Januari.
- Risiko gangguan pada rilis lanjutan dapat ditekan melalui validasi regresi.

### 6. Capaian Bulan Januari

Capaian yang dapat dilaporkan pada Januari 2026:

- Terselesaikannya rangkaian stabilisasi modul inti aplikasi.
- Meningkatnya kualitas maintainability melalui refactoring terarah.
- Peningkatan reliabilitas OCR pada kondisi penggunaan berulang.
- Dimulainya proyek pengembangan model AI beserta pipeline evaluasinya.
- Tersusunnya rencana teknis integrasi AI ke aplikasi secara bertahap.

### 7. Kendala dan Mitigasi

Dalam pelaksanaan Januari, terdapat beberapa tantangan:

- Variasi kualitas data lapangan yang tinggi menyulitkan standardisasi evaluasi model AI.
- Perbedaan karakteristik perangkat pengguna mempengaruhi konsistensi performa OCR.
- Kebutuhan menjaga stabilitas produksi membatasi laju implementasi perubahan besar.

Mitigasi yang diterapkan:

- Penyusunan kategori dataset berbasis tingkat kesulitan untuk evaluasi lebih objektif.
- Pendekatan rilis bertahap agar perubahan besar tidak langsung berdampak ke seluruh pengguna.
- Prioritas pada perbaikan yang berdampak tinggi terhadap stabilitas dan pengalaman pengguna.

### 8. Rencana Lanjutan Februari

Rencana kerja lanjutan yang disiapkan:

- Melanjutkan eksperimen model AI dan benchmark performa terhadap baseline.
- Memulai integrasi terbatas model hasil eksperimen ke skenario pengujian internal.
- Menambah cakupan pengujian regresi untuk modul OCR dan pelaporan.
- Melakukan penyempurnaan UX berbasis umpan balik penggunaan lapangan.
- Menyiapkan dokumentasi teknis untuk fase implementasi AI berikutnya.
