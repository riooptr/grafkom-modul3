# Lab 2 — Pipeline NumPy dan diagnosis

Tanggal pengukuran: 27 November 2026. Lingkungan: Pop!_OS 24.04, Python 3.12.3, NumPy 2.5.5; `OPENBLAS_NUM_THREADS=1` untuk pengukuran raster/ray. Pengukuran di bawah dapat diulang dengan `python lab2_check.py`, `python broken_scene.py --all`, `python experiments.py`, dan `python tests.py` setelah `python make_model.py`. Berkas hasil ditulis ke `out/`.

## Pipeline dan kecocokan

`transforms.py` menggunakan vektor kolom dan `P @ V @ M`: translasi, skala, rotasi, look-at, perspektif OpenGL, dan viewport. Tes E1–E3 memeriksa titik hasil `(3,3,-3)`, dualitas kamera/objek dan pemetaan near/far. `raster3d.py` melakukan clipping bidang near dalam clip-space (`z+w>=0`), pembagian w, viewport, barycentric pada pusat piksel, z-buffer, interpolasi UV memakai `u/w`, `v/w`, `1/w`, sampling bilinear dalam cahaya linear, dan Lambert plus ambient. `raytrace.py` menembakkan ray pinhole, memilih hit Möller–Trumbore terdekat, lalu menggunakan tekstur, cahaya dan transform model yang sama. Normal kedua renderer diorientasikan menghadap kamera: ini aturan shading dua sisi yang dipilih untuk perbandingan, bukan simulasi material satu sisi fisik.

Pada model contoh `data/pusaka_model.obj`, tekstur `data/relief.ppm`, matriks `Ry(35°) @ S(1.4)`, kamera default, 160 × 160:

| Perbandingan | PSNR | SSIM global | Piksel berbeda |
|---|---:|---:|---:|
| Raster terhadap `ref/temple_ref.ppm` | ∞ dB | 1,000000 | 0 |
| Raster terhadap ray tracer | ∞ dB | 1,000000 | 0 |

**Batas penting:** acuan `ref/temple_ref.ppm` dibuat sebelumnya oleh rasteriser yang sama karena acuan independen tidak tersedia dalam proyek awal. Angka pertama adalah uji regresi, bukan validasi independen terhadap referensi dosen. Ray tracer memberikan pemeriksaan implementasi visibilitas dan interpolasi yang berbeda, walaupun ia masih memakai loader, sampler, model transform dan kode warna yang sama. SSIM di `metrics.py` adalah SSIM global satu jendela, bukan SSIM lokal Gaussian. Satu frame cocok tidak menjamin kecocokan semua scene: ray tracer saat ini tidak menerapkan clipping near/far seperti rasteriser dan khusus hasil ini seluruh permukaan terlihat berada dalam frustum.

## Diagnosis dan bukti

Kode perbaikan sudah ada/terbaca saat pengerjaan awal, sehingga catatan ini **tidak mengklaim** bahwa diagnosis ditulis sebelum membaca perbaikannya. Untuk setiap kasus `broken_scene.py --all` sekarang menyimpan warna sebelum/sesudah, citra referensi, dan depth sebelum/sesudah. Depth PPM berskala absolut NDC, hitam untuk background; kuantisasi 8-bit pada gambar dapat menyembunyikan perbedaan depth yang kecil, maka histogram numerik juga disimpan.

| Kasus | Gejala dan diagnosis dari bukti | Perbaikan | SSIM sebelum → sesudah terhadap acuan |
|---|---|---|---:|
| Inside-out | Depth permukaan yang terlihat berbeda rata-rata 0,082 NDC saat culling aktif; urutan tiga vertex dibalik sehingga muka luar dianggap belakang dan muka dalam terlihat. | Kembalikan winding mesh beserta urutan UV-nya. | 0,789685 → 1,000000 |
| Z-fight | Dua quad hampir sebidang pada jarak sekitar 15 unit dengan `near=0,001`, far=200 dan depth 16-bit jatuh ke kode sama; urutan gambar memutuskan pemenang. | Naikkan near ke 1,0; objek uji tetap dalam frustum. | 0,762879 → 0,984151 |
| Stretched | Dengan skala nonuniform, matriks buruk `S @ R` dan yang diinginkan `R @ S` menukar penerapan rotasi dan skala; matriks yang dicetak memiliki komponen silang berbeda. | Terapkan skala di ruang model lalu rotasi: `R @ S`. | 0,646602 → 1,000000 |

Untuk inside-out dan stretched, citra hasil perbaikan juga dipakai sebagai acuan sehingga skor sesudah = 1 secara definisi; bukan pengukuran terhadap gambar eksternal. Untuk z-fight acuan adalah render depth floating-point pada near=1. Pada near=0,001, histogram `out/zfight_depth_bad.csv` memiliki **1 kode depth** untuk 5.704 piksel terlihat; pada near=1, `out/zfight_depth_fixed.csv` memiliki **13 kode**, bin terbesar 3.036 piksel. Dibanding acuan float, 2.852 piksel vs 184 piksel memiliki perbedaan saluran >60 setelah perubahan near. Naikkan near hanya jika clipping objek dekat tetap dapat diterima.

## Perspektif benar (E4)

`experiments.py` merasterisasi segitiga lantai `y=-0,5` dengan titik tepi pada view `z=-1` dan `z=-9`, sehingga `wL=1`, `wR=9`. Untuk koordinat tekstur tepi `uL=0`, `uR=1`, tepat di tengah **layar**:

```text
u_naif = (0 + 1) / 2 = 0,5
u_benar = ((0,5 × 0/1) + (0,5 × 1/9)) / (0,5/1 + 0,5/9) = 0,1
```

Gambar `out/perspective_naive.ppm` dan `out/perspective_correct.ppm` memakai geometri, tekstur dan kamera yang sama tetapi interpolasi UV berbeda; `out/perspective_diff.ppm` menunjukkan perbedaan. Interpolasi linear layar menempatkan pola pada posisi yang salah di bidang miring, sehingga saat sudut pandang berubah polanya dapat tampak berenang. Gambar ini adalah perbandingan satu frame, bukan pengukuran animasi.

## Mipmap dan pitfall

Piramida terdiri dari 256², 128², 64², 32². Setiap level dihitung dari rata-rata kotak 2 × 2 **setelah decoding sRGB ke linear**, lalu dienkode kembali ke sRGB untuk keluaran. Candi dari jauh dirender dengan tekstur level 0 dan secara eksplisit dengan level 3; gambar serta selisih tersedia di `out/distant_mip_0.ppm`, `out/distant_mip_3.ppm`, dan `out/distant_mip_diff.ppm`. Energi perbedaan kuadrat antar piksel bertetangga yang sama-sama mengenai candi: **3546,08 → 91,94**. Ini menunjukkan pengurangan variasi frekuensi tinggi pada frame tersebut; bukan bukti bahwa seluruh kedipan hilang dalam animasi. Pemilihan LOD di eksperimen ini tetap, belum otomatis dari turunan UV; perbandingannya juga mencakup perbedaan detail/shading pada permukaan.

Perbaikan pitfall yang diukur adalah back-face culling dengan winding keluar yang benar. Pada frame 160 × 160, fragmen yang **lulus depth test** berkurang dari 26.840 menjadi 21.527 (19,80%), sedangkan gambar yang dihasilkan identik (PSNR ∞, SSIM 1). Penghitung ini bukan jumlah semua kandidat fragmen yang diuji. Culling hanya aman untuk mesh tertutup dengan winding konsisten; permukaan satu sisi terbuka dapat hilang.

## Verifikasi

`python tests.py` lulus 11 pemeriksaan termasuk clipping near, cocok raster–ray pada frame 96 × 96, E1–E5, dan pengujian model/UV. `python lab2_check.py`, `python broken_scene.py --all`, dan `python experiments.py` menghasilkan angka di atas. Tidak ada klaim uji referensi eksternal, tes semua posisi kamera, mip LOD otomatis, atau animasi temporal.
