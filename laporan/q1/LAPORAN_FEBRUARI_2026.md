# Laporan Bulanan Pengembangan Aplikasi

## Periode Februari 2026

## Pembahasan

### Ringkasan Aktivitas Bulanan

Pada bulan Februari 2026, fokus pengembangan diarahkan pada percepatan implementasi hasil riset awal model AI OCR serta pembentukan alur operasional labeling tools yang lebih terstruktur. Aktivitas ini dilakukan secara paralel dengan stabilisasi aplikasi Panah Pasopati agar perubahan teknologi dapat diadopsi tanpa mengganggu layanan yang sudah berjalan.

Secara umum, pekerjaan Februari menitikberatkan pada tiga sasaran utama: peningkatan kualitas data, penguatan akurasi dan konsistensi model AI, serta kesiapan integrasi bertahap ke modul produksi. Pendekatan ini dipilih untuk memastikan bahwa setiap peningkatan bersifat terukur, dapat diaudit, dan memiliki dampak langsung terhadap kualitas penelusuran kendaraan.

### Aktivitas yang Dilakukan

### 1. Stabilisasi Aplikasi dan Penyempurnaan Alur Penggunaan

Pada periode ini dilakukan peningkatan kualitas pada modul-modul yang bersinggungan langsung dengan proses penelusuran harian pengguna.

Pekerjaan yang dilakukan meliputi:

- Penyempurnaan alur request-response pada fitur penelusuran untuk mengurangi kegagalan proses saat trafik meningkat.
- Penyesuaian penanganan timeout, retry, dan notifikasi status proses agar pengguna memperoleh umpan balik yang lebih jelas.
- Perbaikan minor bug pada proses filter, tampilan daftar hasil, serta sinkronisasi data riwayat.
- Penataan ulang validasi input agar proses pengiriman data menjadi lebih konsisten.

Hasil utama:

- Penggunaan fitur inti menjadi lebih stabil pada berbagai kondisi koneksi.
- Kegagalan proses akibat kondisi sementara dapat ditangani lebih baik.
- Konsistensi pengalaman pengguna meningkat pada siklus penelusuran berulang.

### 2. Pengembangan Model AI OCR (Tahap Eksperimen Lanjutan)

Pada Februari, kegiatan AI berlanjut dari tahap fondasi Januari menuju fase eksperimen terukur.

Ruang lingkup pekerjaan:

- Penyempurnaan baseline model menggunakan dataset internal yang telah dikurasi.
- Penerapan skenario eksperimen bertahap untuk membandingkan model baseline dan kandidat tuning.
- Penyesuaian preprocessing data agar model lebih adaptif terhadap kondisi low light, blur, dan sudut pengambilan gambar yang ekstrem.
- Pengukuran metrik performa awal, mencakup precision, recall, F1-score, dan waktu inferensi.

Progres model AI:

- Baseline model telah tervalidasi untuk skenario uji internal.
- Eksperimen tuning awal menunjukkan peningkatan konsistensi deteksi pada sampel data sulit.
- Struktur evaluasi per skenario berhasil dibentuk untuk mempercepat siklus improvement berikutnya.

Catatan:
Tahap Februari masih berfokus pada validasi internal dan belum dilakukan rollout penuh ke produksi.

### 3. Implementasi Labeling Tools (Tahap Operasional Awal)

Untuk mendukung kebutuhan data berkualitas tinggi, dilakukan pengembangan dan penyempurnaan labeling tools sebagai bagian dari pipeline AI.

Komponen yang dikembangkan:

- Workflow labeling bertahap: assignment data, proses anotasi, review, revisi, dan approval.
- Penandaan status pekerjaan untuk memudahkan monitoring progres tiap batch data.
- Validasi format label agar dataset tetap konsisten dengan standar pelatihan model.
- Mekanisme export dataset terlabel untuk kebutuhan training dan evaluasi.

Quality control labeling:

- Penerapan review berlapis pada sampel data kritis.
- Audit sampling berkala untuk memeriksa konsistensi anotasi antar labeler.
- Pencatatan koreksi utama sebagai referensi guideline labeling berikutnya.

Hasil utama:

- Kecepatan penyediaan dataset terlabel meningkat dibanding proses manual sebelumnya.
- Kualitas anotasi menjadi lebih konsisten untuk kebutuhan pelatihan model.
- Siklus eksperimen model menjadi lebih efisien karena data siap pakai tersedia lebih cepat.

#### Penambahan Progres Model AI (Penyempurnaan dari Baseline Sebelumnya)

Pada Februari 2026, pengembangan model AI OCR difokuskan pada tuning lanjutan dari baseline yang telah tersedia dari periode sebelumnya. Aktivitas utama diarahkan pada penyetelan parameter pelatihan, penyesuaian strategi preprocessing, dan evaluasi ulang performa per skenario data sulit agar hasil pengujian lebih stabil dan terukur. Tim juga melakukan komparasi antar konfigurasi eksperimen untuk menilai keseimbangan antara akurasi pembacaan, konsistensi inferensi, dan efisiensi waktu proses.

Output tambahan bulan ini berupa daftar konfigurasi kandidat yang lebih robust terhadap variasi kualitas citra, ringkasan metrik evaluasi internal yang lebih konsisten, serta prioritas eksperimen lanjutan untuk siklus berikutnya. Dampak operasionalnya adalah peningkatan kecepatan siklus training-evaluasi karena keputusan eksperimen dapat diambil lebih cepat berbasis data pembanding yang lebih rapi.

#### Penambahan Progres Labeling Tools (Peningkatan Workflow dan QA)

Pada sisi labeling tools, Februari diarahkan pada penguatan workflow operasional dan quality assurance secara berlapis. Perbaikan dilakukan melalui penajaman mekanisme review, penerapan audit sampling yang lebih rutin pada batch kritis, serta refinement guideline anotasi untuk menurunkan variasi interpretasi antar labeler. Selain itu, proses tracking status anotasi dan revisi dipertegas agar alur kerja lebih terpantau dan waktu tindak lanjut lebih singkat.

Hasil awal menunjukkan peningkatan konsistensi anotasi dan berkurangnya temuan koreksi berulang pada tahap validasi internal. Kondisi ini berdampak langsung pada percepatan ketersediaan data latih yang siap digunakan dalam eksperimen model. Hingga akhir Februari, integrasi ke mobile app masih berada pada tahap persiapan teknis dan internal testing, serta belum masuk implementasi final ke produksi.

### 4. Integrasi Bertahap AI ke Alur Aplikasi

Selama Februari dilakukan persiapan teknis untuk menghubungkan hasil eksperimen AI dengan kebutuhan aplikasi mobile secara aman.

Aktivitas yang dilakukan:

- Penyusunan contract data antara komponen inference dan modul aplikasi.
- Penyesuaian penanganan fallback ketika confidence model berada di bawah ambang batas.
- Simulasi alur internal untuk memastikan kompatibilitas dengan proses penelusuran yang sudah berjalan.
- Penguatan logging teknis agar hasil inferensi mudah ditelusuri untuk analisis perbaikan.

Hasil utama:

- Fondasi integrasi AI telah terbentuk untuk tahap uji terbatas.
- Risiko gangguan terhadap alur utama aplikasi dapat diminimalkan.
- Tim memiliki visibilitas lebih baik terhadap perilaku model di skenario pengujian.

### 5. Pengujian, Validasi, dan Quality Assurance

Untuk menjaga kualitas rilis, pengujian dilakukan pada sisi aplikasi dan komponen AI secara terkoordinasi.

Aktivitas QA:

- Pengujian regresi alur login, penelusuran, detail hasil, dan riwayat pengguna.
- Pengujian edge case OCR pada kondisi pencahayaan rendah dan hasil gambar kurang ideal.
- Validasi hasil labeling terhadap pedoman anotasi terbaru.
- Pengujian kestabilan aplikasi pada sesi penggunaan berulang.

Hasil utama:

- Alur utama aplikasi tetap stabil selama penambahan komponen baru.
- Error handling untuk skenario confidence rendah telah berjalan lebih baik.
- Kualitas data labeling terjaga untuk mendukung eksperimen lanjutan.

### 6. Capaian Bulan Februari

Capaian yang dapat dilaporkan pada Februari 2026:

- Terselesaikannya fase eksperimen lanjutan model AI pada baseline internal.
- Beroperasinya labeling tools pada alur kerja awal untuk penyediaan dataset terstruktur.
- Meningkatnya konsistensi quality control data anotasi.
- Terbentuknya fondasi integrasi AI ke aplikasi dengan pendekatan bertahap.
- Tetap terjaganya stabilitas fitur inti aplikasi selama proses peningkatan teknologi.

### 7. Kendala dan Mitigasi

Dalam pelaksanaan Februari, terdapat beberapa tantangan:

- Variasi karakter data lapangan menyebabkan hasil evaluasi model belum sepenuhnya homogen.
- Perbedaan interpretasi anotasi pada awal implementasi labeling tools memerlukan penyesuaian pedoman.
- Integrasi komponen AI membutuhkan sinkronisasi lintas modul agar tidak menimbulkan regresi.

Mitigasi yang diterapkan:

- Penguatan guideline anotasi dan sesi kalibrasi antar reviewer.
- Penetapan skenario uji berbasis tingkat kesulitan data.
- Pendekatan integrasi bertahap dengan fallback mechanism dan monitoring ketat.

### 8. Rencana Lanjutan Maret

Rencana kerja lanjutan yang disiapkan:

- Melanjutkan tuning model berdasarkan hasil evaluasi batch Februari.
- Meningkatkan cakupan data terlabel untuk skenario sulit dan data outlier.
- Melakukan uji internal terkontrol untuk integrasi inference pada alur penelusuran.
- Menyempurnakan dashboard monitoring kualitas data dan performa model.
- Menyiapkan dokumentasi readiness untuk tahap limited rollout.
