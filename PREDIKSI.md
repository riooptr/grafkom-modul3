# Prediksi sebelum pengukuran

Pada model 240 segitiga dan frame 128 × 128, saya memprediksi rasterisasi CPU NumPy sekitar 0,1 detik, ray tracing sekitar 0,5 detik, sehingga ray tracing 5 kali lebih lambat. Ini prediksi, bukan pengukuran. Rasterisasi membatasi pekerjaan pada kotak setiap segitiga, sedangkan ray tracer menguji seluruh ray terhadap semua segitiga tanpa struktur percepatan. Bayangan keras dan refleksi lebih mudah diproduksi ray tracing dengan ray tambahan; rasterisasi memerlukan shadow map dan pendekatan refleksi tambahan. Prediksi ditulis sebelum menjalankan pengukuran berikut.
