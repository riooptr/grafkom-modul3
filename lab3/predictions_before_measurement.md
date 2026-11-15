# Prediksi sebelum benchmark Lab 3

Dicatat 15 November 2026 sebelum benchmark baru pada resolusi 512x512.
Prediksi ini memakai informasi run Lab 2 pada 256x256: raster 0.397 detik,
ray 5.958 detik. Jadi ini bukan prediksi buta tanpa pengalaman sebelumnya.

| Ukuran | Prediksi |
|---|---|
| Raster NumPy 512x512 | 0.6-1.5 detik/frame |
| Ray caster NumPy 512x512 | 20-35 detik/frame |
| Rasio ray/raster | sekitar 20-35x, titik tengah 25x |

Benchmark: model 636 triangle, texture dan kamera sama, satu warmup kecil,
tiga run terukur, laporkan sampel dan median. Pembandingan CPU-vs-CPU.

Prediksi kualitas: primary rays dan Lambert yang sama seharusnya hampir identik
dengan raster. Kode belum punya secondary shadow/reflection rays, sehingga tidak
bisa mengukur keunggulan kualitas efek yang belum diimplementasikan. Ray tracing
lebih langsung diperluas dengan ray bayangan dan pantulan; raster memerlukan
metode seperti shadow map/probe. Pernyataan ini konseptual, bukan hasil render.

Pilihan awal: viewer interaktif memakai Three.js/WebGL GPU; beauty shot dengan
reflection/GI memakai renderer ray/path tracing yang sudah punya secondary
transport. Pilihan berubah menurut frame-rate, jumlah triangle, kebutuhan efek,
dan perangkat dengan RT acceleration. Primary ray caster sederhana ini belum
merupakan renderer beauty-shot lengkap.
