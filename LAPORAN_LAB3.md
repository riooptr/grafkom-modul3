# Lab 3 — Three.js, perbandingan, dan pengiriman

Tanggal: 27 November 2026. Peramban terintegrasi VS Code Electron 43.8.9 / Chromium 150.0.7871.250; Three.js 0.160.0, Python 3.12.3, NumPy 2.5.5, Node 18.19.1 pada Pop!_OS 24.04.

## Viewer dan pembandingan

`web/` memuat OBJ dan tekstur model Lab 1, OrbitControls, satu directional light dan ambient. Framebuffer 512 × 512, pixel ratio 1, keluaran sRGB. UI memakai `MeshLambertMaterial` dan sampling bilinear linear-light yang disamakan dengan NumPy; faktor π mengimbangi normalisasi diffuse Three.js 0.160. Antialias dimatikan agar aturan cakupan lebih sebanding. Uji lokal `python -m http.server 8765 --directory web`: menyeret canvas mengubah frame, dan tombol ekspor menghasilkan blob P6 berukuran 786.447 byte dengan header `P6`, 512 × 512, max 255. Aset harus disajikan melalui HTTP dan Three.js memerlukan CDN.

Pada kamera default, model transform `Ry(35°) S(1.4)`, render NumPy 512² dibanding canvas browser pada frame yang sama memakai `compare.py`:

| PSNR | SSIM global | Mean absolute error | Max |
|---:|---:|---:|---:|
| 50,3 dB | 1,000 (dibulatkan) | 0,3 / 255 | 222 / 255 |

`out/three_diff.ppm` adalah selisih diperbesar 6 kali; `web/mymodel_three.ppm` adalah capture canvas browser yang ditulis sebagai PPM saat pengujian. Pada 154.896 piksel interior yang didefinisikan sebagai foreground kedua gambar dan dikelilingi foreground 3 × 3, rata-rata beda kanal 0,525, persentil 99 = 1, dan 5 piksel memiliki beda kanal >10. Hanya dua piksel berbeda dalam mask foreground. Jadi siluet tidak memperlihatkan pergeseran seluruh model; beberapa selisih besar tersisa pada tepi/sambungan segitiga karena aturan rasterisasi/tie-break dan rounding GPU, serta perbedaan kecil warna dari presisi. Material Lambert tidak memiliki komponen specular sehingga specular bukan penyebab. Ini pemeriksaan satu kamera dan GPU/browser tersebut, bukan jaminan lintas perangkat.

Untuk mengulangi perbandingan, jalankan:

```bash
python make_model.py
python raster3d.py --model web/mymodel.obj --w 512 --h 512 --out out/temple_raster.ppm
npx serve web
# buka URL, jangan orbit; Export PPM lalu simpan sebagai web/mymodel_three.ppm
python compare.py out/temple_raster.ppm web/mymodel_three.ppm --diff out/three_diff.ppm
```

## Prediksi dan pengukuran

Prediksi tertulis sebelum pengukuran ada di `PREDIKSI.md`: raster 0,1 s, ray 0,5 s, ray 5× lebih lambat pada 128². Pengukuran `OPENBLAS_NUM_THREADS=1 python benchmark.py` memakai model 240 segitiga: raster **0,1042 s**, ray **0,2417 s**, rasio **2,32×**, PSNR ∞ dan SSIM 1 untuk frame tersebut. Hasil waktu adalah satu kali pada CPU, bukan benchmark statistik; jumlah thread dan perangkat memengaruhi hasil. Ray lebih cepat daripada perkiraan relatif, tetapi tetap lebih lambat pada implementasi tanpa akselerasi ini. Tidak ada hard shadow atau reflection yang diimplementasikan pada kedua renderer, sehingga kemampuan visual itu adalah alasan teoretis, bukan hasil yang diukur.

Untuk viewer interaktif saya memilih rasterisasi GPU: update kamera dan visibilitas banyak segitiga dapat berjalan paralel. Untuk beauty shot saya memilih ray tracing yang telah diperluas dengan shadow rays, refleksi, material, dan struktur percepatan: ray sekunder secara alami menjawab occlusion dan pantulan. Implementasi ray tracer praktikum ini sendiri **belum** menghasilkan efek tersebut, jadi belum merupakan renderer beauty shot siap produksi.

## Reproduksi dan status publikasi

Workflow `.github/workflows/pages.yml` disiapkan untuk mengunggah direktori `web/` dan deploy melalui GitHub Actions setelah push ke `main`; pengaturan Pages repo harus memilih sumber GitHub Actions. URL yang diharapkan adalah `https://riooptr.github.io/grafkom-modul3/`, tetapi **belum diverifikasi sebagai halaman terbit**. Pada mesin ini SSH agent tidak memiliki identitas dan browser GitHub belum login; perubahan lokal belum dapat didorong atau konfigurasi Pages diaktifkan secara otomatis. Jangan menganggap URL ekspektasi sebagai deliverable yang sudah hidup.

Verified lokal: `make_model.py`, `tests.py`, `benchmark.py`, renderer browser dan ekspor diperiksa pada toolchain di atas. Clone bersih, `pip install -r requirements.txt` dan `npx serve web` dalam lingkungan baru belum diverifikasi; Python yang digunakan berasal dari virtual environment lain dan viewer lokal diuji dengan `python -m http.server`. Lihat README untuk perintah lengkap.

## Refleksi

Three.js dan WebGL mengambil alih upload mesh, transformasi vertex model/view/projection, clipping, pembagian perspektif, viewport, interpolasi atribut, rasterisasi, depth test dan shading di GPU. Saya tetap memilih kamera dan matriks, menyediakan geometri, material dan texture, dan menyesuaikan konvensi data. Salah winding dapat membuang muka saat culling; near/far dapat memotong bangunan atau merusak presisi depth. OBJ dan susunan baris tekstur harus konsisten pada sumbu v agar relief tidak terbalik. Tekstur berwarna harus didekode dari sRGB sebelum penerangan dan output dienkode kembali; salah menandai color space mengubah kecerahan. Perbandingan frame ini juga memerlukan aturan sampling dan kekuatan lampu yang cocok, bukan hanya kamera yang sama.

Pada 200.000 segitiga, hambatan pertama tergantung perangkat: transformasi vertex, jumlah draw call, bandwidth upload atau fill rate dapat menjadi dominan. Renderer Python dengan loop per segitiga dan ray tracer tanpa BVH akan melambat lebih awal; GPU tidak otomatis gagal pada jumlah itu. Profiling frame time dan memori diperlukan sebelum menentukan optimasi seperti batching, frustum culling, LOD, indeks mesh, atau BVH.
