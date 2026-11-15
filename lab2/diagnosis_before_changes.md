# Diagnosis sebelum perubahan implementasi Lab 2

Tanggal: 15 November 2026, Asia/Jakarta.

Catatan kronologi: kode starter `broken_scene.py`, termasuk komentarnya tentang
perbaikan, sudah dibaca saat audit. Dokumen ini mencatat diagnosis sebelum patch
Lab 2 dibuat; bukan klaim diagnosis buta sebelum melihat solusi starter.

| Kasus | Bukti awal | Diagnosis dan perubahan yang akan diuji |
|---|---|---|
| Inside-out | Generator fallback mengklaim CCW keluar, tetapi signed volume kotak pertama -1.4. Backface culling menyeleksi sisi yang berbeda setelah indeks face dibalik. | Winding generator fallback sebenarnya ke dalam. Normalisasikan winding 15 komponen kotak yang diketahui untuk baseline diagnosis, balik face **beserta indeks UV** untuk injeksi bug, lalu balik keduanya kembali untuk perbaikan. Jangan mengubah ref/temple_ref.ppm yang sudah dibekukan. |
| Z-fighting | Dua quad berpotongan di jarak sekitar 15; near 0.001, far 200, depth integer 16-bit. Depth proyeksi mendekati 1. | Kuantisasi menyatukan depth yang semestinya berbeda; draw order memenangkan tie. Uji perubahan near menjadi 1 dengan bit depth yang tetap, bandingkan keduanya ke oracle float yang memakai near sama, dan simpan histogram kode/depth. Perbaiki juga jumlah interval integer menjadi 2**bits-1. |
| Stretched | Translasi nol dan skala seragam pada rantai awal membuat pertukaran faktor komutatif. Kode lalu menambahkan diag(2.2,0.5,1,1). | Distorsi awal disebabkan skala tambahan, bukan hanya urutan. Buat eksperimen eksplisit memakai faktor nonuniform yang **sama**, good=R@S dan bad=S@R, sehingga pembuktian benar-benar mengisolasi urutan. Cetak kedua matriks. |

Temuan tambahan:

- Ray tracer belum menolak hit di luar near/far; tambahkan uji jarak eye-space,
  bukan membandingkan parameter ray t langsung dengan near/far.
- Starter belum memiliki mipmap. Tambahkan piramida empat tingkat dengan rata-rata
  **linear light**, LOD dari turunan UV perspective-correct, dan sampling trilinear.
- 120 dari 180 segitiga fallback mempunyai UV dengan luas nol. Jangan menulis
  ulang baseline historis untuk menyembunyikan temuan ini. Model Lab 1 memiliki
  UV tidak kolaps pada seluruh 636 segitiga dan dipakai untuk eksperimen texture.
- Reference frame lokal dibuat rasteriser starter saat Lab 1. Ini baseline regresi,
  bukan ground truth independen. Ray tracing menjadi cross-check visibilitas dan
  interpolasi barycentric yang dihitung melalui jalur berbeda.
