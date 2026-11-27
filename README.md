# Grafika Komputer — Lab 1: model dan transformasi

Laporan lengkap, perhitungan, lisensi aset, sumber, dan keterbatasan ada di [LAPORAN_LAB1.md](LAPORAN_LAB1.md).

Model [`web/mymodel.obj`](web/mymodel.obj) adalah blockout interpretatif Candi Borobudur yang dibuat secara prosedural dengan `make_model.py`: 240 segitiga dan UV untuk setiap segitiga. Tekstur [`web/relief.ppm`](web/relief.ppm) juga dihasilkan prosedural dan didedikasikan sebagai CC0-1.0. Ini bukan scan atau model Blender; aset bawaan `data/pusaka_model.obj` adalah model contoh terpisah.

## Menjalankan

```bash
python -m pip install -r requirements.txt
python make_model.py
python worked_example.py
python tests.py
npx serve web
```

Buka alamat yang diberikan `serve` dan drag untuk orbit. Viewer mengambil Three.js 0.160.0 dari CDN dan memerlukan akses jaringan. `make_model.py` menghasilkan ulang model, tekstur, dan aset contoh; jangan menjalankannya di atas model yang sudah diedit tanpa menyimpan perubahan terlebih dahulu.

## Toolchain dan verifikasi

27 November 2026: Pop!_OS 24.04 LTS, Python 3.12.3, NumPy 2.5.5, Node 18.19.1, Chrome terpasang 158.0.8127.88, Three.js 0.160.0. Python dan NumPy digunakan dari virtual environment yang sudah tersedia pada mesin ini. Viewer dibuka di browser terintegrasi; versinya tidak tercatat.

`make_model.py`, `worked_example.py`, dan `tests.py` dijalankan pada toolchain di atas; sembilan pemeriksaan lulus. Acuan `ref/temple_ref.ppm` dibuat dari renderer proyek ini sendiri sehingga hanya merupakan pengujian regresi, bukan referensi independen. Uji clone bersih dan pemasangan dependensi pada lingkungan baru belum diverifikasi.
