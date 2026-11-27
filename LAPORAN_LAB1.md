# Lab 1 — Model, coordinate spaces, dan presisi

Tanggal: 27 November 2026 (Asia/Jakarta). Model adalah blockout interpretatif Candi Borobudur, bukan rekonstruksi arkeologis berskala. Toolchain: Pop!_OS 24.04 LTS, Python 3.12.3, NumPy 2.5.5, Node.js 18.19.1, Google Chrome terpasang 158.0.8127.88, Three.js 0.160.0. Viewer sempat dibuka dengan browser terintegrasi; versi browser terintegrasi tidak tercatat, sehingga versi Chrome di atas hanya versi yang terpasang di OS.

## Input dan asal aset

Model [`web/mymodel.obj`](web/mymodel.obj) adalah blockout orisinal yang **dihasilkan dengan `make_model.py`**, bukan hasil ekspor Blender, pemindaian, atau pengukuran bangunan. Ini pendekatan generator yang disediakan tugas; jangan mengklaim pemodelan manual. Terdapat 240 segitiga dan koordinat UV non-degenerat per segitiga, termasuk pada permukaan atas dan samping. Tekstur [`web/relief.ppm`](web/relief.ppm) dihasilkan secara prosedural oleh `relief_texture` di `make_model.py`, bukan diunduh atau dibuat AI. Kode dan tekstur prosedural baru untuk tugas ini boleh digunakan dan diubah dengan lisensi CC0-1.0; tidak ada klaim lisensi atas kode bawaan proyek. Sumber model bukan replika bentuk detail atau relief asli Borobudur.

## Telusuran vertex

Kamera `eye=(2.6, 2.0, 3.4)`, target `(0, 0.7, 0)`; `M=Ry(35°) S(1.4,1.4,1.4)`, `V=look_at(eye,target)`, `P=perspective(45°,1,1,14)`. Vertex pertama OBJ adalah `p=(-1.15,0,-1.15,1)`.

Dua langkah pertama dihitung tangan (dibulatkan):

1. **Object → world:** setelah skala, `(-1.61,0,-1.61)`. Dengan `cos 35°=0.819152` dan `sin 35°=0.573576`, `x=0.819152(-1.61)+0.573576(-1.61)=-2.242293`, `y=0`, `z=-0.573576(-1.61)+0.819152(-1.61)=-0.395377`. Hasil `(-2.242293,0,-0.395377,1)`.
2. **World → view:** kalikan matriks `V` di bawah dengan hasil tersebut. `x=0.794358(-2.242293)-0.607450(-0.395377)=-1.541011`; `y=-0.176535(-2.242293)-0.230853(-0.395377)-0.669788=-0.182671`; `z=0.581232(-2.242293)+0.760073(-0.395377)-4.676685=-6.280494`. Hasil `(-1.541011,-0.182671,-6.280494,1)`; z negatif berarti di depan kamera.

Langkah selanjutnya dihitung dengan [`worked_example.py`](worked_example.py):

| Space | Koordinat |
|---|---|
| Clip | `(-3.720330, -0.441007, 5.092877, 6.280494)` |
| NDC, clip dibagi w | `(-0.592363, -0.070218, 0.810904)` |
| Screen 512 × 512, origin kiri atas | `(104.355144, 273.975934)` |
| Depth tersimpan, `(z_ndc+1)/2` | `0.905452` |

Koordinat layar ini kontinu; pusat piksel raster adalah bilangan bulat ditambah 0,5. Nilai di atas dapat direproduksi dengan `python worked_example.py`.

## Membaca kolom

Kolom adalah citra vektor basis dengan `w=0`, sedangkan kolom terakhir adalah citra origin dengan `w=1`:

```text
M = [ 1.146813   0       0.803007   0 ]
    [ 0          1.4     0          0 ]
    [-0.803007   0       1.146813   0 ]
    [ 0          0       0          1 ]

V = [ 0.794358   0         -0.607450   0        ]
    [-0.176535   0.956840  -0.230853  -0.669788 ]
    [ 0.581232   0.290616   0.760073  -4.676685 ]
    [ 0          0          0          1        ]
```

| Input | Ke world oleh M | Ke view oleh V M |
|---|---|---|
| Arah +x model | `(1.146813, 0, -0.803007)` | `(1.398766, -0.017076, 0.056221)` |
| Arah +y model | `(0, 1.4, 0)` | `(0, 1.339576, 0.406863)` |
| Arah +z model | `(0.803007, 0, 1.146813)` | `(-0.058757, -0.406504, 1.338395)` |
| Origin model | `(0, 0, 0)` | `(0, -0.669788, -4.676685)` |

Untuk membaca `V` sendiri, kolom arah +x, +y, +z **world** berturut-turut adalah `(0.794358,-0.176535,0.581232)`, `(0,0.956840,0.290616)`, dan `(-0.607450,-0.230853,0.760073)`; origin world masuk ke `(0,-0.669788,-4.676685)`. Kolom `V` tidak langsung menyatakan sumbu model yang sudah diputar: itu sebabnya tabel juga memakai `VM`. Hasilnya konsisten dengan citra raster yang diperiksa: bangunan berdiri ke atas, alas dan tingkat-tingkatnya terlihat dalam perspektif, dan +x model bergerak terutama ke kanan layar. Citra statis tidak membuktikan semua arah satu per satu; pemeriksaan numerik kolom adalah bukti arah tersebut.

## Bacaan: LearnOpenGL

Saya membaca [LearnOpenGL, “Coordinate Systems”](https://learnopengl.com/Getting-started/Coordinate-Systems), diakses 27 November 2026. Bab tersebut membuat matriks model, view, dan projection di CPU, lalu mengirimkannya sebagai uniform dan menghitung `projection * view * model * vec4(aPos,1)` di vertex shader. Setelah shader menghasilkan posisi clip, OpenGL menjalankan clipping, pembagian perspektif, dan viewport sebagai bagian pipeline. Sebaliknya, `transforms.py` melakukan perkalian dan pembagian `w` secara eksplisit pada array NumPy, kemudian `viewport` menghitung koordinat layar di CPU. Di kode praktikum, clipping near dilakukan oleh `raster3d.py`, bukan `transforms.py`.

Urutan matematikanya sama: vektor kolom, operasi paling kanan lebih dulu. Perbedaannya adalah pembagian tanggung jawab, bukan hukum proyeksi yang berbeda. API GPU menerima geometri dan transformasi agar vertex dapat diproses paralel dan hasilnya diteruskan ke rasterisasi; NumPy membuat angka setiap tahap terlihat dan mudah diperiksa. Penjelasan bab itu tentang `gl_Position` adalah pengingat penting: nilai tersebut masih clip-space, belum NDC; membagi `w` sebelum menyerahkannya akan mengubah perilaku clipping dan interpolasi perspektif.

## Refleksi insiden: depth dan konvensi

Kasus nyata yang relevan adalah transisi reverse-Z Godot 4.3: [Clay John, 29 April 2024](https://godotengine.org/article/introducing-reverse-z/), menjelaskan bahwa teknik ini mengurangi artefak presisi tetapi mematahkan shader tertentu yang mengasumsikan near berada pada clip-space z=0, termasuk shader quad layar penuh. Masalahnya bukan sekadar nilai depth bergeser: efek layar yang mengandalkan asumsi lama dapat hilang atau salah terlihat. Perbaikan yang didokumentasikan adalah memakai konvensi near baru dan matriks proyeksi/inversnya, serta menambahkan peringatan untuk pola shader yang terdampak. Ini contoh nyata risiko saat memperbaiki akurasi tanpa mengaudit konsumen depth.

[Nathan Reed, “Depth Precision Visualized” (2015)](https://developer.nvidia.com/content/depth-precision-visualized) menunjukkan bahwa depth perspektif adalah pemetaan linear dari `1/z`, sehingga kode depth banyak terkumpul dekat kamera. Untuk near=1 dan far=14 pada viewer ini, dengan jarak positif `z`, `d=14/13 - 14/(13z)`. Pada z=1, d=0; z=3, d≈0.717949; z=10, d≈0.969231; z=14, d=1. Jadi rentang jarak 10–14 hanya memakai sekitar 3,08% rentang kode depth, sedangkan 1–3 memakai sekitar 71,8%. Permukaan berdekatan jauh lebih mudah jatuh ke kode sama. Godot menggunakan reverse-Z dengan depth floating-point: near dipetakan ke 1 dan far ke 0 sehingga sebaran floating-point mengimbangi nonlinieritas; membalik urutan saja pada depth integer tidak menambah jumlah kode.

| Konvensi | OpenGL klasik / contoh ini | Direct3D umum | Scene pengungkap salah konvensi |
|---|---|---|---|
| Handedness view | kanan, kamera memandang −z | sering kiri, kamera +z | Objek asimetris dengan label kiri/kanan dan back-face culling; model cermin atau muka hilang. |
| Urutan vektor/matriks | contoh ini kolom: `P V M p` | contoh DirectXMath lazim memakai baris: `p M V P` | Translasi lalu rotasi 90° pada objek di luar origin; urutan salah memindahkannya ke posisi berbeda. |
| NDC z | `[-1,1]` | `[0,1]` | Dua bidang tepat di near dan far: near hilang atau depth salah jika proyeksi dicampur. |
| Origin UV / citra | OBJ v=0 bawah; loader contoh membalik v terhadap baris citra atas | tekstur biasanya diperlakukan v=0 atas | Tekstur bertuliskan “ATAS” di bagian atas dan panah asimetris akan terbalik. |

Handedness, notasi vektor dan orientasi UV adalah **pilihan aplikasi/aset**, bukan aturan mutlak API; Direct3D juga dapat memakai kamera kanan atau vektor kolom dan OpenGL dapat dikonfigurasi berbeda. Penyimpanan matriks row-major/column-major juga berbeda dari urutan perkalian. Kesalahan terjadi ketika proyeksi, loader, shader, dan aset memakai pilihan yang tidak cocok.

Untuk menangkap pembalikan handedness secara otomatis, `tests.py` memeriksa normal segitiga pertama di permukaan atas: `cross(b-a,c-a).y > 0`. Refleksi sumbu atau pembalikan winding global akan membuat orientasi ini gagal, alih-alih hanya mengandalkan tampilan candi yang simetris. Tes ini terbatas pada orientasi segitiga itu, bukan bukti bahwa semua transformasi kamera atau semua mesh benar; tambahkan objek asimetris dan pengujian determinan `det(M[:3,:3])>0` jika transformasi model berubah.
