# Blueprint OCR Kustom (plat nomor Indonesia)

Status: **gerbang 1 dan 2 LULUS, ekspor TFLite berparitas; berikutnya integrasi di aplikasi (bendera fitur), ukur kecepatan di HP kelas bawah, lalu keputusan bentuk akhir**. Modul: `02_training_platform/03_ocr_custom`.

Keputusan pemilik: bentuk akhir (gabungan vs kustom murni) **ditentukan dari hasil benchmark**, bukan di muka.
Lisensi font plat: dinyatakan tidak bermasalah oleh pemilik (dipakai hanya untuk data latih internal).

## Tujuan dan gerbang keputusan

Memutuskan apakah OCR kustom menggantikan atau mendampingi Google ML Kit di Panah Pasopati.
Hanya diintegrasikan jika **lolos gerbang** di bawah. Angka gerbang ditetapkan SEBELUM melihat hasil.

| # | Gerbang | Angka |
|---|---|---|
| 1 | Benar persis pada soal uji tetap, dibanding ML Kit (versi terbaiknya) | minimal **+15 poin**, tanpa kenaikan "otomatis-salah" |
| 2 | Plat jelas dan dekat (kasus harian) | tidak lebih buruk dari ML Kit |
| 3 | Ukuran / kecepatan | < 5 MB, < 50 ms di HP kelas bawah |

Jika setelah iterasi analisis kesalahan selisihnya < +5 poin: **berhenti**, hanya pakai perbaikan murah di ML Kit.
Bentuk akhir yang dievaluasi: ML Kit saja, kustom saja, dan gabungan (kustom dulu, ML Kit cadangan).

## Pembanding yang sudah diukur (ML Kit, 1.110 plat unik Kaggle)

Detektor = `plat_recognation.tflite` milik aplikasi, crop padding 0,15, batas lebar 800 px.

| Kondisi | Benar persis | Kosong/ditolak | Otomatis-salah |
|---|---|---|---|
| Foto utuh (kondisi produksi) | 34,4% | 21,0% | 18,8% |
| Crop kecil: ML Kit menolak gambar < 32 px | 25,7% | 27,6% (20,8% ditolak) | 15,4% |
| Crop kecil diperbesar (min tinggi 96 px) | 38,5% | 9,7% | 19,4% |

Catatan: dataset ini sulit (resolusi rendah, sebagian label salah). Angka ini BUKAN akurasi lapangan.
Kesalahan baca terbanyak: Q terbaca O. `O` dan `I` dipakai di seri plat (3,6% dan 3,8%), jadi tidak bisa diperbaiki dengan aturan.

## Hasil tahap 0 (dikunci)

Korpus: 3.032 plat unik, 3.941 gambar (`scripts/build_corpus.py`, seed 13). Split per nomor plat dan sumber:
uji 500 plat (250 HP/Kaggle + 250 tol/JMTO, satu gambar per plat), validasi 300, latih 2.232.
358 plat muncul di lebih dari satu sumber; setiap plat ditempatkan sekali.

Audit label soal uji: dari 127 kasus ML Kit membaca plat valid beda 1-2 karakter dari label, 40 disampel dan
ditinjau visual: 34 label benar (ML Kit salah), 2 label jelas salah, 4 ambigu. Enam kasus itu dikeluarkan.
Perkiraan noise label soal uji 1-4%, kecil terhadap gerbang dan menimpa kedua model sama besar.

**Soal uji final: 494 plat (247 HP + 247 tol).** Pembanding ML Kit pada soal ini:

| Sumber | Varian | Benar persis | Benar + normalizer | Otomatis-salah |
|---|---|---|---|---|
| HP | A (seperti aplikasi) | 32,4% | 42,1% | 13,4% |
| HP | B (crop kecil diperbesar) | **39,7%** | 51,0% | 15,8% |
| Tol | A | **40,5%** | 59,9% | 23,5% |
| Tol | B | 34,4% | 55,1% | 20,6% |

Memperbesar crop kecil membantu foto gaya HP tetapi merugikan kamera tol, jadi "ML Kit terbaik" = varian terbaik per sumber:
**40,1% benar persis** (55,5% dengan normalizer).

**Target gerbang 1: OCR kustom minimal 55,1% benar persis pada soal uji ini** (+15 poin), tanpa kenaikan otomatis-salah.
Soal uji terkunci: tidak boleh dilatih, diaugmentasi, atau diubah lagi.

## Hasil pelatihan (tahap 2-3)

Model: CNN penuh-konvolusi + CTC, 1,13 juta parameter, masukan abu-abu 48x192, `src/train.py`.
Data latih: 2.829 gambar asli + 30.000 sintetis (3 font plat), cuplik 40% asli per epoch.

| Versi | Perubahan | Validasi asli (benar persis) |
|---|---|---|
| v1 | resize memeras crop ke 4:1, sintetis tanpa pemotongan | 22,9% (tol 6,5%, Kaggle 25,9%) |
| v2 | resize mempertahankan rasio (letterbox); sintetis meniru crop terpotong di bawah/samping | 78,3% |

Sebab v1 gagal (dari analisis kesalahan di validasi): crop kamera tol terpotong di bawah sehingga huruf seri akhir hilang separuh,
dan crop asli (~2,2:1) diperas ke 4:1 sedangkan sintetis ~2,9:1. Tidak ada kebocoran: irisan nomor plat latih/validasi/uji/sintetis = 0.

**Soal uji terkunci (494 plat), dijalankan SEKALI pada v2:**

| | Benar persis | Salah per karakter | Otomatis-salah |
|---|---|---|---|
| OCR kustom v2 | **76,3%** | 6,3% | 17,2% |
| - HP/Kaggle | 74,1% | 8,1% | 17,0% |
| - Tol/JMTO | 78,5% | 4,6% | 17,4% |
| ML Kit terbaik | 40,1% (55,5% dengan normalizer) | - | 15,8-23,5% per sumber |

Gerbang 1 lulus (target 55,1%; selisih +36 poin dari ML Kit mentah, +21 dari ML Kit + normalizer).
Selang kepercayaan 95% (bootstrap 2.000x): kustom 76,3% [72,5; 80,2]; selisih terhadap ML Kit+normalizer +20,9 poin [+16,0; +25,5],
terhadap ML Kit mentah +36,2 [+31,2; +41,3]; per sumber: Kaggle +23,1 [+17,0; +29,6], tol +18,6 [+10,9; +25,9].

**Gerbang 2 (kasus harian) lulus.** Kelompok ditetapkan dari tinggi crop dan ketajaman (varians Laplacian), tanpa melihat hasil model:

| Kelompok | n | Kustom | ML Kit+normalizer |
|---|---|---|---|
| tinggi >= 64 px dan tajam (tersil atas) | 161 | 85,7% | 67,1% |
| tinggi >= 64 px | 433 | 78,1% | 55,2% |
| tinggi 48-64 px | 13 | 69,2% | 69,2% (seri, n kecil) |
| tinggi < 48 px | 48 | 62,5% | 54,2% |
| buram (tersil bawah) | 163 | 65,0% | 52,1% |

Tidak ada kelompok yang lebih buruk dari ML Kit. Soal uji tidak boleh dipakai lagi untuk iterasi.

**Ekspor TFLite** (`src/export_tflite.py`, port Keras + salin bobot; normalisasi di dalam model): paritas dengan PyTorch diuji pada 494 soal uji.

| Varian | Ukuran | Sama dengan PyTorch | Benar persis | CPU PC 1 thread |
|---|---|---|---|---|
| fp32 | 4,54 MB | 100% | 76,3% | 2,5 ms |
| **fp16** | **2,28 MB** | **100%** | **76,3%** | 2,5 ms |
| int8 dinamis | 1,16 MB | 98,4% | 76,5% | 0,8 ms |

Pilihan awal: **fp16** (paritas penuh). Gerbang 3: ukuran lulus (< 5 MB); kecepatan di HP kelas bawah (< 50 ms) BELUM terukur, angka PC hanya perkiraan.

## Keputusan bentuk akhir: kustom saja, tanpa cadangan ML Kit

Dihitung pada 494 plat uji yang sama. OCR kustom menghasilkan bacaan yang bukan nomor polisi utuh pada 32 plat (6,5%).
Dari 32 itu, ML Kit menebak benar 10, memberi bacaan valid tetapi salah 12, dan kosong/tidak valid 10. Akurasi kustom saja 76,3%,
kustom + cadangan ML Kit 78,3% (+10 plat). Cadangan hanya menambah 2 poin dan 12 kasusnya bisa lolos sebagai nomor sah yang salah,
jadi diputuskan (pemilik, 2 November 2026): hapus cadangan ML Kit. Bacaan tidak valid masuk dialog koreksi dengan teks terisi,
tidak pernah dicari otomatis. ML Kit hanya tinggal di build pembanding (`--dart-define=CUSTOM_OCR=false`); pustakanya baru dihapus
setelah tes lapangan. Berkas bernama "mlkit" di APK sekitar 30 MB (perkiraan dari nama berkas).

## Data

| Sumber | Plat unik berlabel | Label dari |
|---|---|---|
| Kaggle plate_text_dataset | ~1.297 (valid) | `label.csv` |
| JMTO (Roboflow v22) | ~1.900 | nama berkas + kotak YOLO (14 dari 15 sampel cocok) |
| Data sendiri | ~370 | nama berkas + kotak |
| Jurnal | - | font `FE-FONT.TTF`, crop karakter |

Aturan keras:
- Pemisahan **per nomor plat dan per sumber**; soal uji dikunci sebelum pelatihan dan tidak pernah diedit/diaugmentasi.
- Soal uji tidak boleh dipilih dari kasus yang dibaca benar ML Kit (menguntungkan ML Kit).
- Label yang diduga salah: kasus ML Kit tidak sama dengan label ditinjau visual; yang ambigu dikeluarkan dari soal uji.
- Dataset dan model tidak di-commit (data plat termasuk data pribadi). Hanya skrip dan ringkasan angka.

Split: uji ~500 plat (campuran gaya HP dan kamera tol), validasi ~300, sisanya latih.

## Sintetis dan augmentasi (hanya untuk data latih)

- Render dengan font plat (`PlatNomor.ttf`, 95 glyph, A-Z dan 0-9 lengkap; `FE-FONT.TTF` untuk variasi) di latar hitam/putih/kuning/merah, baris masa berlaku kecil, bingkai.
- Nomor diacak mengikuti distribusi kode wilayah asli, tidak menyalin nomor soal uji.
- Degradasi kamera: blur biasa dan gerak, perkecil lalu perbesar, artefak JPEG, noise, terang/gelap/silau, miring +-10 derajat, terpotong di tepi, kotoran/baut.
- Lisensi font tidak tercantum di metadata: hanya dipakai untuk data latih internal, sumber unduhan perlu dicek.

## Model

CRNN kecil (CNN ringan, BiGRU, CTC), karakter `A-Z0-9` + blank. Keluaran diperiksa terhadap pola huruf-angka-huruf.
Dua tahap: pra-latih pada sintetis, setel pada data asli, berhenti dini pada validasi asli.
Ekspor TFLite, dekode CTC greedy di Dart. Cek GPU di awal pelatihan.

## Evaluasi

Satu soal uji untuk ML Kit, kustom, dan gabungan. Metrik: benar persis, salah per karakter, kosong,
otomatis-salah; selang kepercayaan bootstrap; dipecah per sumber, resolusi, warna plat.
Tambahan: kurva ketahanan terhadap blur untuk kedua model.

## Integrasi (hanya jika lulus gerbang)

Model ketiga lewat `tflite_flutter`, tes paritas Python vs Dart, bendera fitur lokal, dua event Firebase
(otomatis-langsung vs dikoreksi petugas) untuk mengukur di lapangan.

## Risiko

| Risiko | Penanganan |
|---|---|
| Kebocoran latih/uji | pemisahan per plat dan sumber, soal uji dikunci |
| Domain tol vs HP | soal uji memuat gaya HP, hasil dipecah per sumber |
| Terlalu cocok dengan sintetis | validasi hanya data asli |
| Label salah | peninjauan kasus berselisih, ambigu dikeluarkan |
| Lisensi font | internal saja, sumber dicek |

## Tahapan

0. Korpus + soal uji + pembanding ML Kit. 1. Pembangkit sintetis. 2. Latih v1 + evaluasi.
3. Analisis kesalahan + iterasi. 4. **Gerbang keputusan.** 5. Integrasi.
