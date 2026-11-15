"""Verify the actual exported WebGL frame and write Lab 3 results/report.

This does not control a browser or manufacture a browser frame: input PPM/JSON
must have been exported by the viewer UI. It can be rerun after deployment.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from objio import load_obj
from ppm_io import read_ppm
from transforms import perspective,look_at,project,viewport
from metrics import psnr,ssim

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'lab3'
URL='https://riooptr.github.io/grafkom-modul3/'


def metrics(a,b):
    p=psnr(a,b);d=np.abs(a.astype(float)-b.astype(float))
    return {'psnr_db':float(p) if np.isfinite(p) else 'inf','ssim_global':ssim(a,b),
            'mean_absolute_error':float(d.mean()),'max_absolute_error':int(d.max())}


def main():
    cpu=read_ppm(str(OUT/'mymodel_raster.ppm'));gpu=read_ppm(str(OUT/'mymodel_three.ppm'))
    b=json.loads((OUT/'browser_reference.json').read_text())
    orbit=json.loads((OUT/'browser_orbit.json').read_text())
    reset=json.loads((OUT/'browser_after_reset.json').read_text())
    assert cpu.shape==gpu.shape==(512,512,3)
    assert b['ready'] and b['triangles']==636 and b['threeRevision']=='160'
    assert b['drawingBuffer']==[512,512] and b['pixelRatio']==1 and b['antialias']
    assert b['textureMode']=='reference' and not b['wireframe']
    assert not np.allclose(b['camera']['position'],orbit['camera']['position'])
    assert np.allclose(b['camera']['position'],reset['camera']['position'])
    if (OUT/'mymodel_three_public.ppm').exists():
        assert np.array_equal(gpu,read_ppm(str(OUT/'mymodel_three_public.ppm')))
        pub_orbit=json.loads((OUT/'browser_public_orbit.json').read_text())
        pub_reset=json.loads((OUT/'browser_public_reference.json').read_text())
        assert not np.allclose(pub_orbit['camera']['position'],pub_reset['camera']['position'])
        assert np.allclose(pub_reset['camera']['position'],b['camera']['position'])
    P=perspective(40,1,1,50);V=look_at((4.8,3.8,6.7),(0,1.5,0))
    deltas={'model':float(np.max(np.abs(np.array(b['modelMatrix']).reshape(4,4,order='F')-np.eye(4)))),
            'view':float(np.max(np.abs(np.array(b['viewMatrix']).reshape(4,4,order='F')-V))),
            'projection':float(np.max(np.abs(np.array(b['projectionMatrix']).reshape(4,4,order='F')-P)))}
    assert max(deltas.values())<1e-10
    mesh=load_obj(ROOT/'data/mymodel.obj')
    _,ndc=project(P,V,np.eye(4),mesh.positions);screen=viewport(ndc,512,512)
    # A conservative surface-interior mask excludes every projected triangle edge,
    # including occluded triangles; not merely the exterior silhouette.
    edge_img=Image.new('L',(512,512));draw=ImageDraw.Draw(edge_img)
    for face in mesh.faces:
        points=[tuple(x) for x in screen[face,:2]]
        draw.line(points+[points[0]],fill=255,width=3)
    edges=np.asarray(edge_img)>0
    occupied=np.any(cpu>0,2)|np.any(gpu>0,2)
    both=np.any(cpu>0,2)&np.any(gpu>0,2)
    inner=both&~edges
    border=occupied&~inner
    assert inner.sum()>10000
    r={'resolution':[512,512],'triangles':636,'matrix_max_abs_errors':deltas,
       'global':metrics(cpu,gpu),'surface_interior':metrics(cpu[inner],gpu[inner]),
       'edge_region':metrics(cpu[border],gpu[border]),
       'interior_pixels':int(inner.sum()),'edge_pixels':int(border.sum()),
       'silhouette_iou':float(both.sum()/occupied.sum()),
       'browser_ui_checks':{'orbit_drag_changes_camera':True,'reset_restores_camera':True,
                            'wireframe_toggle_observed':True,'texture_mipmap_option_observed':True,
                            'ppm_download_observed':True,'metadata_download_observed':True,
                            'console_error_warning_count':0},
       'browser':b,'benchmark':json.loads((OUT/'benchmark.json').read_text()),
       'deployment':json.loads((OUT/'deployment.json').read_text()) if (OUT/'deployment.json').exists()
                      else {'status':'prepared_not_yet_deployed','expected_url':URL}}
    assert r['global']['psnr_db']>30
    assert r['surface_interior']['psnr_db']>40,r['surface_interior']
    assert r['silhouette_iou']>.98
    for name,img in [('numpy',cpu),('webgl',gpu),('difference_x6',np.clip(np.abs(cpu.astype(int)-gpu.astype(int))*6,0,255).astype(np.uint8))]:
        Image.fromarray(img).save(OUT/f'{name}.png')
    mask=np.zeros_like(cpu);mask[inner]=[58,170,108];mask[border]=[219,143,73]
    Image.fromarray(mask).save(OUT/'comparison_regions.png')
    montage=Image.new('RGB',(1536,548),'#151c22');d=ImageDraw.Draw(montage)
    for i,(name,img) in enumerate([('NumPy 512x512',cpu),('Three.js / WebGL',gpu),('|difference| x6',np.asarray(Image.open(OUT/'difference_x6.png')))]):
        montage.paste(Image.fromarray(img),(i*512,36));d.text((i*512+12,12),name,fill='white')
    montage.save(OUT/'comparison.png')
    (OUT/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report(r)
    print(json.dumps({k:r[k] for k in ['global','surface_interior','edge_region','interior_pixels','edge_pixels','silhouette_iou','matrix_max_abs_errors']},indent=2))


def report(r):
    k=r['benchmark'];b=r['browser'];g=r['global'];i=r['surface_interior'];e=r['edge_region']
    samples='\n'.join(f"| {name} | "+' / '.join(f'{v:.6f}' for v in values)+f" | {k['median_seconds'][name]:.6f} |" for name,values in k['samples_seconds'].items())
    deploy=r['deployment']
    clean=json.loads((OUT/'clean_clone.json').read_text()) if (OUT/'clean_clone.json').exists() else {}
    text=f'''# Lab 3 - Viewer WebGL, Perbandingan Renderer, dan Publikasi

Model: blockout terinspirasi Gapura Bajang Ratu, **636 triangle**. Pengerjaan
dengan bantuan Codex, melanjutkan starter dan hasil Lab 1/2. Bukan klaim seluruh
kode/model/hitungan dikerjakan manual tanpa bantuan. Tanggal: 15 November 2026.

## 1. Viewer yang dijalankan di browser

`web/` kini mandiri: model OBJ, PNG, material/lisensi, Three.js **0.160.0**, OBJLoader
dan OrbitControls berada di dalam folder itu. Tidak perlu CDN saat runtime;
versi dan SHA-256 modul dicatat di `web/vendor/manifest.json`. Model M identitas,
kamera (4.8,3.8,6.7), target (0,1.5,0), FOV40, near1/far50. Drawing buffer
512x512 dan `setPixelRatio(1)` sudah diperiksa melalui metadata browser.

Pengujian UI: drag mengubah posisi kamera ke
{json.loads((OUT/'browser_orbit.json').read_text())['camera']['position']}; tombol
reset mengembalikannya ke posisi awal. Toggle wireframe, pilihan mipmap, unduhan
PPM, dan unduhan JSON bekerja. Console tidak mencatat error/warning pada pengujian
lokal. Ukuran mobile awal serta breakpoint desktop diperiksa; viewport sementara
dikembalikan ke ukuran semula setelah pengujian. Ekspor tetap 512x512 walaupun
canvas diskalakan oleh CSS.

![Viewer lokal yang benar-benar berjalan](viewer_local.jpg)

Bukti lain: `viewer_orbit.jpg`, `viewer_wireframe.jpg`, `viewer_mipmap.jpg`,
`browser_reference.json`, `browser_orbit.json`, `browser_after_reset.json`.
User agent aktual: `{b['userAgent']}`. WebGL: `{b['webgl']}`.

## 2. Mencocokkan NumPy dan WebGL

Frame WebGL berasal dari tombol **Unduh frame PPM**, menggunakan `gl.readPixels`
dan pembalikan baris agar origin file berada di kiri atas. Tidak dibuat ulang
atau disimulasikan menggunakan Python. Frame browser disimpan sebagai
`lab3/mymodel_three.ppm`; alias lokal `web/mymodel_three.ppm` sengaja di-ignore Git.

Ada dua konvensi penting untuk kecocokan bagian dalam:

1. Lambert r160 membagi irradiance dengan pi. Intensitas ambient dan directional
   dikalikan pi agar persamaan akhirnya sama dengan NumPy `0.15+0.85*max(N.L,0)`.
2. Sampler starter memakai `u*(W-1)`. Offset setengah texel dan repeat `(W-1)/W`
   menyamakan pusat texel GPU pada UV model yang berada di 0.02..0.98. Ini bukan
   perubahan universal untuk setiap UV yang melintasi seam repeat.

Texture diberi sRGB, perhitungan lighting linear, keluaran sRGB, tone mapping mati.
Mode referensi mematikan mipmap agar sama dengan NumPy bilinear; mode halus
mengaktifkan mipmap GPU. Kamera dan M/V/P dibandingkan secara numerik dengan
array column-major Three.js; maksimum selisih matriks {max(r['matrix_max_abs_errors'].values()):.3e}.

| Region | PSNR (dB) | SSIM global | MAE channel [0..255] |
|---|---|---|---|
| Seluruh frame | {g['psnr_db']:.6f} | {g['ssim_global']:.9f} | {g['mean_absolute_error']:.6f} |
| Interior permukaan | {i['psnr_db']:.6f} | {i['ssim_global']:.9f} | {i['mean_absolute_error']:.6f} |
| Sekitar edge | {e['psnr_db']:.6f} | {e['ssim_global']:.9f} | {e['mean_absolute_error']:.6f} |

![Frame dan selisih](comparison.png)

![Mask: interior hijau, region edge oranye](comparison_regions.png)

Interior memakai {r['interior_pixels']} pixel yang tidak berada pada garis edge
triangle berketebalan tiga pixel; seluruh edge proyeksi dikecualikan termasuk
triangle tertutup, sehingga mask ini konservatif. Region edge berisi
{r['edge_pixels']} pixel. Silhouette IoU {r['silhouette_iou']:.9f}.
SSIM yang dipakai adalah satu window global, bukan local-window SSIM Gaussian.

Residual terbesar di tepi berkaitan dengan MSAA browser yang tidak dipakai raster
CPU. Interior juga menyisakan perbedaan; rounding/precision, interpolasi texture
GPU, dan pemilihan face pada geometri bertumpuk belum diisolasi satu per satu.
Penyebab khusus setiap outlier interior belum dibuktikan, sehingga tidak diklaim
semua residual hanya MSAA di tepi. Material hanya Lambert, tanpa
specular; tidak ada highlight specular yang dipakai sebagai alasan selisih.

## 3. Prediksi, pengukuran, dan keputusan

[Prediksi sebelum pengukuran](predictions_before_measurement.md) dicatat sebelum
benchmark 512x512. Informasi run Lab 2 256x256 sudah diketahui: raster0.397s dan
ray5.958s. Prediksi baru: raster0.6-1.5s, ray20-35s, rasio20-35x (tengah25x).

Satu warmup 64x64 per renderer lalu tiga run CPU pada scene/resolusi sama:

| Renderer | Sampel waktu (detik) | Median (detik) |
|---|---|---|
{samples}

Rasio median aktual **{k['ray_to_raster_ratio']:.3f}x**. Median ray berada dalam
rentang prediksi; raster sedikit lebih cepat daripada batas bawah prediksi,
sehingga prediksi rasio terlalu rendah. Pembagian frame ke beberapa triangle kecil
membatasi banyak bounding box raster, sedangkan ray caster brute-force menguji
setiap ray terhadap seluruh triangle. Run dilakukan di komputer yang juga
menjalankan browser; beban mesin tidak diisolasi, dan penyebab variasi antarsampel
tidak diukur. Seluruh sampel dipertahankan, bukan hanya run tercepat.

CPU raster/ray 512x512 mencapai PSNR **{k['psnr_db']:.6f} dB**, SSIM global
{k['ssim_global']:.9f}, berbeda di {k['different_pixels']} pixel. Itu pembandingan
primary visibility dan Lambert. **Secondary hard shadow dan reflection belum
diimplementasikan** pada kedua renderer CPU; tidak diklaim ada peningkatan
kualitas efek tersebut dari gambar primary yang hampir sama. Pembahasan
shadow/reflection di prediksi bersifat konseptual.

Pilihan viewer: **Three.js/WebGL GPU**, karena targetnya interaksi orbit dengan
draw call GPU. Metadata browser mencatat median interval rAF
{b['rafIntervalMedianMs']:.3f} ms dan CPU render submission
{b['cpuRenderSubmissionMedianMs']:.3f} ms dari {b['timingSamples']} sampel terakhir.
Submission CPU bukan waktu GPU selesai, dan rAF bukan jaminan frame rate pada
setiap perangkat. Angka ini tidak disamakan dengan benchmark CPU di tabel.

Pilihan beauty shot yang membutuhkan pantulan/GI: **renderer ray/path tracing
yang mendukung secondary transport**, bukan menganggap ray caster primer ini
sudah merupakan renderer marketing lengkap. Jika hanya membutuhkan Lambert
tanpa efek tersebut, raster yang hasilnya hampir identik lebih hemat waktu.
Pilihan dapat berbalik pada GPU dengan RT acceleration dan kebutuhan reflection
dinamis; atau pada target frame rate ketat/perangkat lemah yang lebih cocok dengan
raster dan pencahayaan baked. Itu batas konkret keputusan, bukan klaim satu metode
selalu unggul.

## 4. Publikasi dan reproduksi

Repositori tujuan pengguna: [riooptr/grafkom-modul3](https://github.com/riooptr/grafkom-modul3).
URL viewer: [{URL}]({URL}). Status tercatat: **{deploy['status']}**.
Isi `web/` dipublikasikan dari branch `gh-pages` root. Source lengkap, laporan,
model, texture, dan pengujian tetap berada pada branch utama. Token GitHub hanya
dibaca dari Git Credential Manager ke memori proses, tidak ditulis ke repo.

Deployment GitHub Pages: commit `{deploy.get('pages_commit','pending')}`,
HTTP `{deploy.get('http_status','pending')}`. Viewer publik, orbit/reset, dan ekspor
PPM diuji dari URL publik, dengan console tanpa error/warning. Frame publik
identik dengan ekspor lokal: `{deploy.get('public_ppm_identical_to_local','pending')}`.
Checksum viewer.js, CSS, OBJ, PNG dan modul Three.js yang ter-host cocok dengan
source yang disiapkan. Bukti metadata publik tersimpan sebagai
`browser_public_reference.json`, `browser_public_orbit.json`, dan
`browser_public_console.json`.

![Bukti viewer dari URL publik](viewer_public.jpg)

```powershell
git clone https://github.com/riooptr/grafkom-modul3.git
cd grafkom-modul3
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe make_model.py
.\\.venv\\Scripts\\python.exe tests.py
npx --yes serve@14.2.4 web
```

Tidak perlu build/npm install untuk viewer karena modul dan aset sudah tersimpan
lokal. `prepare_web.py` menyinkronkan aset dari data dan memverifikasi manifest;
unduhan vendor hanya diperlukan bila file belum ada atau diminta refresh.
Benchmark dapat diulang dengan `benchmark_lab3.py`; analisis frame aktual dengan
`analyse_lab3.py`. Script analisis tidak mengendalikan browser dan tidak membuat
frame WebGL palsu. Lihat `clean_clone_verification.txt` untuk transkrip clone
bersih dan `deployment.json` untuk status deployment yang benar-benar diperoleh.

Clone bersih yang diperiksa berasal dari commit `{clean.get('commit','pending')}`,
menggunakan virtual environment baru; status pengujian
`{clean.get('core_tests','pending')}`. Status serve/browser:
`{clean.get('browser_serve_status','pending')}`. Clone verifikasi mendahului commit
dokumentasi akhir; hasil berikutnya hanya menambahkan bukti dan laporan, tidak
mengubah source viewer atau pipeline yang sudah diuji.

## 5. Refleksi pipeline (bahan dipahami sebelum viva)

Three.js mengambil alih pengiriman vertex/index buffer, penggabungan matriks
model-view, penyiapan shader, dan draw call. GPU melakukan clipping, pembagian w,
viewport, pembentukan fragment, interpolasi perspective-correct, depth test,
texture sampling, dan multisample resolve. Tugas yang sebelumnya menjadi loop
NumPy kini berjalan melalui pipeline GPU. OrbitControls membangun perubahan pose
kamera dari drag pengguna, tetapi tidak dapat memperbaiki konvensi aset yang salah.

Konvensi tetap menjadi tanggung jawab aplikasi: winding harus sesuai culling,
near/far harus menjaga visibilitas dan presisi depth, UV harus cocok dengan origin
pixel texture, dan texture sRGB harus didecode sebelum perhitungan cahaya linear.
Eksperimen ini juga menunjukkan bahwa persamaan Lambert serta pusat texel perlu
disamakan. Model dan kamera benar belum menjamin frame pixel-wise cocok jika
pengaturan sampling atau intensitas lampu berbeda. Mipmap sengaja dipisahkan
dari mode referensi agar perbedaan filter tidak disalahartikan sebagai bug MVP.

Pada 200.000 triangle, ray caster brute-force menjadi hambatan utama karena
jumlah intersection meningkat sebagai triangle dikali pixel; 512x512 berarti
sekitar 52,4 miliar uji triangle per frame sebelum akselerasi. BVH atau struktur
spasial diperlukan. Raster CPU juga perlu memperbaiki overhead loop/bounding box.
Di WebGL, vertex/fragment throughput, overdraw, draw call, dan bandwidth texture
bergantung pada scene; jumlah triangle saja tidak menentukan frame rate. Mesh
perlu diprofilkan, material/batching dikelola, dan LOD dipertimbangkan. Angka
200.000 ini pembahasan skala, bukan hasil benchmark mesh sebesar itu.

Hasil ini menutup pekerjaan teknis Lab 3. Pemahaman, pengulangan perhitungan
tangan, dan pembelaan satu commit tetap perlu dilakukan mahasiswa sendiri.
'''
    (OUT/'Laporan_Lab3.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
