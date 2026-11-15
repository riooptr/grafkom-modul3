"""Run Lab 1 evidence, asset checks, coordinate trace, and report generation.

Run from any directory: Module3/.venv/Scripts/python.exe Module3/run_lab1.py
The report explicitly records Codex assistance and unperformed human activities.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hashlib
import json
import platform
import re
import subprocess
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw

from make_lab1_model import main as make_model
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from raster3d import render
from transforms import translate, rot_y, look_at, perspective, viewport, project

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'lab1'
FIG = OUT / 'figures'


def command(argv):
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    if proc.returncode:
        raise RuntimeError(f'{argv}: {proc.stderr}')
    return proc.stdout.strip()


def toolchain():
    ps = r"""$cv=Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$b=@{}
foreach($p in @('C:\Program Files\Google\Chrome\Application\chrome.exe','C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')) {
 if(Test-Path -LiteralPath $p) { $v=(Get-Item -LiteralPath $p).VersionInfo; $b[$v.ProductName]=$v.ProductVersion }
}
@{build=$cv.CurrentBuild;ubr=$cv.UBR;release=$cv.DisplayVersion;browsers=$b}|ConvertTo-Json -Compress
"""
    win = json.loads(command(['powershell', '-NoProfile', '-Command', ps]))
    try:
        date = datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(timespec='seconds')
    except ZoneInfoNotFoundError:
        from datetime import timezone, timedelta
        date = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    html = (ROOT/'web/index.html').read_text(encoding='utf-8')
    three = re.search(r'three@([\d.]+)/', html).group(1)
    return {'recorded_at':date,'timezone':'Asia/Jakarta',
            'os':f"Windows {'11' if int(win['build']) >= 22000 else '10'} {win['release']} (build {win['build']}.{win['ubr']})",
            'python':platform.python_version(),'python_executable':sys.executable,
            'numpy':np.__version__,'pillow':PIL.__version__,
            'node':command(['node','--version']), 'npm':command(['npm.cmd','--version']),
            'browsers_installed':win['browsers'], 'three_declared_in_module3':three,
            'browser_render_executed':False,
            'three_reading_version':'0.160.0 (pinned upstream source)',
            'pip_check':command([sys.executable,'-m','pip','check'])}


def vec(v):
    return '(' + ', '.join(f'{x:.6f}' for x in v) + ')'


def matrix_md(m):
    return '```text\n'+'\n'.join(' '.join(f'{v:11.6f}' for v in row) for row in m)+'\n```'


def save_render(mesh, tex, name, **kwargs):
    rgb, depth, filled = render(mesh, tex_u8=tex, **kwargs)
    Image.fromarray(rgb).save(FIG/f'{name}.png')
    write_ppm(str(FIG/f'{name}.ppm'), rgb)
    return rgb, depth, filled


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    make_model()
    tc = toolchain()
    metadata = json.loads((ROOT/'data/lab1_model.json').read_text(encoding='utf-8'))
    mesh = load_obj(ROOT/'data/mymodel.obj')
    tri = mesh.triangle_positions()
    twice_area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
    tuv = mesh.triangle_uvs()
    a,b = tuv[:,1]-tuv[:,0], tuv[:,2]-tuv[:,0]
    uv_area = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]) / 2
    assert mesh.n_tris >= 200
    assert mesh.face_uvs.shape == mesh.faces.shape
    assert np.all(twice_area > 1e-10), 'Degenerate geometry'
    assert np.all(uv_area > 1e-10), 'Collapsed UV triangle'
    assert np.all((mesh.uvs >= 0) & (mesh.uvs <= 1))
    # Every component is a closed box: positive signed volume proves outward winding.
    volumes = []
    for part in metadata['components']:
        first = part['first_triangle']
        t = tri[first:first+part['triangle_count']]
        volume = np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
        assert volume > 0, part['name']
        volumes.append(float(volume))

    # First vertex of the asymmetric plaque, read back from the saved OBJ.
    vi = len(mesh.positions)-24
    p = np.r_[mesh.positions[vi],1.0]
    M = rot_y(90) @ translate((1,0,-1))
    V = look_at((0,0,8),(0,0,0))
    P = perspective(60,800/600,1,50)
    world = M @ p
    eye = V @ world
    clip = P @ eye
    ndc = clip[:3]/clip[3]
    scr = viewport(ndc[None,:],800,600)[0]
    x,y,z = p[:3]
    # Independent scalar derivation, without calling the matrix functions.
    translated = np.array([x+1,y,z-1,1.0])
    manual_world = np.array([z-1,y,-x-1,1.0])
    manual_eye = np.array([z-1,y,-x-9,1.0])
    assert np.allclose(world,manual_world,atol=1e-12)
    assert np.allclose(eye,manual_eye,atol=1e-12)
    helper_clip,helper_ndc = project(P,V,M,mesh.positions[vi:vi+1])
    assert np.allclose(helper_clip[0],clip) and np.allclose(helper_ndc[0],ndc)
    assert clip[3] > 0 and np.all(np.abs(clip[:3]) <= clip[3])
    assert np.allclose(V @ np.array([0,0,8,1.]),[0,0,0,1.])

    # A negative-control mirror must be rejected, even if a symmetric silhouette passes.
    mirrored = M @ np.diag([-1,1,1,1])
    def convention_guard(candidate):
        return bool(np.linalg.det(candidate[:3,:3]) > 0 and
                    np.allclose(candidate @ p, manual_world, atol=1e-9))
    assert convention_guard(M)
    assert not convention_guard(mirrored), 'Mirror negative control was not detected'
    mirror_clip = P @ V @ mirrored @ p
    mirror_screen = viewport((mirror_clip[:3]/mirror_clip[3])[None,:],800,600)[0]

    tex = read_ppm(str(ROOT/'data/lab1_brick.ppm'))
    _,depth,filled = save_render(mesh,tex,'model',w=720,h=720,
                                eye=(4.8,3.8,6.7),center=(0,1.50,0),near=1,far=50,
                                fovy=40,cull_back=True)
    assert np.isfinite(depth).sum() > 20000, 'Preview is empty or badly framed'
    trace_rgb,_,_ = save_render(mesh,tex,'trace_scene',w=800,h=600,model=M,
                               eye=(0,0,8),center=(0,0,0),near=1,far=50,fovy=60)
    annotated = Image.fromarray(trace_rgb)
    draw = ImageDraw.Draw(annotated)
    sx,sy = scr[:2]
    draw.ellipse((sx-6,sy-6,sx+6,sy+6),outline='#ffe16a',width=2)
    draw.text((sx+10,sy-17),f'v{vi+1}: ({sx:.3f}, {sy:.3f})',fill='#ffe16a')
    origin = np.array([0,0,0,1.])
    # A labelled projection of the model's origin and three axes, using the same M/V/P.
    for axis,label,color in [(np.array([.5,0,0,0]),'+x','#ff756e'),
                             (np.array([0,.5,0,0]),'+y','#69efad'),
                             (np.array([0,0,.5,0]),'+z','#81b8ff')]:
        coords=[]
        for q in [origin,origin+axis]:
            cc=P@V@M@q
            coords.append(tuple(viewport((cc[:3]/cc[3])[None,:],800,600)[0,:2]))
        draw.line(coords,fill=color,width=3)
        draw.text(coords[1],label,fill=color)
    annotated.save(FIG/'trace_annotated.png')
    uv_img = Image.fromarray(tex).resize((512,512),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(uv_img)
    for t in tuv[-12:]:
        xy=[(float(u*511),float(v*511)) for u,v in t]
        draw.line(xy+[xy[0]],fill='#ffe16a',width=2)
    draw.text((20,20),'UV plaque: 12 triangles / 6 faces',fill='white')
    uv_img.save(FIG/'uv.png')

    near,far=1.0,50.0
    def depth01(distance,n=near,f=far):
        return f/(f-n)-f*n/((f-n)*distance)
    distances=[1,2,5,10,25,49,50]
    rows=[{'distance':d,'depth01':depth01(d),
           'depth24_code':round(depth01(d)*(2**24-1))} for d in distances]
    first_meter=depth01(near+1)-depth01(near)
    bad_near_fraction=depth01(.01+1,.01,50)-depth01(.01,.01,50)
    derivative_far=far*near/((far-near)*far**2)
    far_step=1/((2**24-1)*derivative_far)
    checks={'minimum_200_triangles':True,'all_geometry_non_degenerate':True,
            'all_faces_have_non_degenerate_uv':True,'all_box_winding_outward':True,
            'scalar_vs_matrix_world_view':True,'project_matches_trace':True,
            'traced_vertex_inside_frustum':True,'camera_maps_to_origin':True,
            'normal_pose_accepted':True,'mirrored_negative_control_rejected':True,
            'textured_preview_nonempty':True,'pip_dependencies_consistent':True}
    results={'toolchain':tc,'model':metadata,'checks':checks,
             'geometry_min_twice_area':float(twice_area.min()),
             'uv_min_area':float(uv_area.min()),'minimum_component_signed_volume':min(volumes),
             'vertex_index_zero_based':vi,'object':p.tolist(),'translated':translated.tolist(),
             'world':world.tolist(),'view':eye.tolist(),'clip':clip.tolist(),
             'ndc':ndc.tolist(),'screen_xy':scr[:2].tolist(),
             'depth01':float((ndc[2]+1)/2),'M':M.tolist(),'V':V.tolist(),'P':P.tolist(),
             'depth_samples':rows,'first_meter_fraction':first_meter,
             'first_meter_fraction_near_001':bad_near_fraction,
             'depth24_step_near_far':far_step,
             'mirror_screen_xy':mirror_screen[:2].tolist(),
             'preview_visible_pixels':int(np.isfinite(depth).sum()),
             'preview_fragment_writes':filled}
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    (OUT/'toolchain.json').write_text(json.dumps(tc,indent=2)+'\n',encoding='utf-8')
    trace=command([sys.executable,'worked_example.py','--model','data/mymodel.obj',
                   '--vertex',str(vi),'--width','800','--height','600'])
    (OUT/'vertex_trace.txt').write_text(trace+'\n',encoding='utf-8')
    (OUT/'requirements.lock.txt').write_text(f'numpy=={np.__version__}\npillow=={PIL.__version__}\n',encoding='utf-8')
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'data/mymodel.obj',ROOT/'data/lab1_brick.png',ROOT/'data/lab1_brick.ppm']}
    (OUT/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    depth_table='\n'.join(f"| {r['distance']} | {r['depth01']:.9f} | {r['depth24_code']} |" for r in rows)
    browser_lines='; '.join(f'{k} {v}' for k,v in tc['browsers_installed'].items())
    report=f'''# Lab 1 - Model Bangunan, Telusuran Pipeline, dan Refleksi

Tanggal pencatatan: {tc['recorded_at']} (Asia/Jakarta).

**Transparansi pengerjaan.** Model, skrip, perhitungan, serta draf analisis ini
disiapkan dengan bantuan Codex atas permintaan pengguna. Bukan klaim bahwa mahasiswa
memodelkan atau menghitungnya manual tanpa bantuan. Turunan skalar di bawah disediakan
untuk diulang dengan tangan dan dipahami sebelum pengumpulan/viva. Model merupakan
blockout interpretatif yang terinspirasi Gapura Bajang Ratu, bukan rekonstruksi terukur.
`Blender/House.blend` hanya diinspeksi; tidak diubah dan tidak dipakai sebagai model final.

## 1. Toolchain yang benar-benar diperiksa

| Komponen | Hasil |
|---|---|
| OS | {tc['os']} |
| Python | {tc['python']} |
| NumPy | {tc['numpy']} |
| Pillow | {tc['pillow']} (ekspor/annotasi PNG saja) |
| Node.js / npm | {tc['node']} / {tc['npm']} |
| Browser terpasang | {browser_lines} |
| Three.js di import map Module3 | {tc['three_declared_in_module3']} |
| Pemeriksaan dependensi | {tc['pip_check']} |

Versi browser dibaca dari executable terpasang. Rendering WebGL dan orbit di browser
belum dijalankan karena itu pekerjaan Lab 3. Bacaan shader untuk butir 5 menggunakan
source Three.js tag r160 (0.160.0), dibaca tanpa dieksekusi. Paket Three.js lama di
folder Modul 2 adalah 0.160.1 dan tidak dianggap sebagai instalasi Module3.

## 2. Input model dan texture

- Model: `data/mymodel.obj`, {mesh.n_tris} segitiga, {len(mesh.positions)} vertex,
  {len(metadata['components'])} komponen kotak. Ambang 200 segitiga terpenuhi.
- Komponen: alas bertingkat, dua pilar dengan lorong terbuka, ambang atas, enam
  tingkat atap, puncak, sayap, tangga depan/belakang, dan plakat asimetris.
- Texture: `data/lab1_brick.png` dan `.ppm`, 256 x 256, pola bata prosedural asli.
  Material OBJ/MTL menunjuk PNG; renderer NumPy memakai PPM yang pixel-nya sama.
- UV: setiap sisi kotak memiliki empat sudut UV berbeda; {len(uv_area)} dari
  {mesh.n_tris} segitiga memiliki luas UV positif. Luas UV minimum {uv_area.min():.6f}.
- Sumbu object: right-handed, +y ke atas, depan +z. Loader NumPy membalik v sekali
  untuk citra yang baris pertamanya berada di atas.
- Aset baru didedikasikan CC0-1.0; provenance dan batas lisensi ada di
  `data/ASSET_LICENSE.md`. Tidak mengambil texture pihak ketiga.

![Pratinjau model bertexture](figures/model.png)

Gambar di atas dihasilkan rasteriser starter pada kamera (4.8, 3.8, 6.7), target
(0, 1.5, 0), FOV 40 derajat, M identitas. Ini pratinjau bentuk; kamera telusuran
di bagian berikut sengaja berbeda dan parameternya dicatat terpisah.

![UV yang tidak kolaps pada plakat](figures/uv.png)

Semua sisi memakai patch bata yang sama, sehingga UV saling menumpuk secara sengaja.
Ini tiling/reuse texture, bukan atlas unik. Setiap komponen kotak memiliki winding
keluar yang diperiksa dengan signed volume positif. Komponen saling beririsan dan
belum di-union menjadi manifold tunggal; cukup untuk blockout praktikum, bukan mesh
siap cetak 3D. Dimensi satuan arbitrer dan relief arkeologis belum direkonstruksi.
Referensi identitas situs: [Direktori Majapahit, Candi Bajang Ratu][bajang] (2021,
berdasarkan survei 2018). Proporsi model adalah keputusan desain latihan.

## 3. Satu vertex melewati enam space

Vertex OBJ nomor **{vi+1}**, indeks Python **{vi}**, adalah sudut plakat asimetris.
Tidak memakai titik contoh (2,3,4) yang bukan berasal dari model.

Konvensi: column vector, `p' = M @ p`, operasi paling kanan dijalankan dulu;
right-handed; kamera menghadap -z; NDC z [-1,1]; layar y ke bawah.

`M = Ry(90) @ T(1,0,-1)`:

{matrix_md(M)}

`V = look_at((0,0,8), (0,0,0)) = T(0,0,-8)`:

{matrix_md(V)}

`P = perspective(60, 800/600, 1, 50)`:

{matrix_md(P)}

Turunan skalar sebagai bahan perhitungan tangan, independen dari pemanggilan matriks:

```text
p = {vec(p)}
T p = (x+1, y, z-1, 1) = {vec(translated)}
Ry(90) (T p) = (z-1, y, -(x+1), 1) = {vec(manual_world)}
V (M p) = (z-1, y, -(x+1)-8, 1) = {vec(manual_eye)}
```

| Space | Nilai |
|---|---|
| Object (x,y,z,w) | {vec(p)} |
| World = M p | {vec(world)} |
| View = V M p | {vec(eye)} |
| Clip = P V M p | {vec(clip)} |
| NDC = clip.xyz / clip.w | {vec(ndc)} |
| Screen x,y (kontinu; origin kiri atas) | {vec(scr[:2])} |
| Depth NDC / depth [0,1] | {ndc[2]:.6f} / {(ndc[2]+1)/2:.6f} |

Lebar/tinggi 800/600. `x_screen=(x_ndc+1)*400`,
`y_screen=(1-y_ndc)*300`. `clip.w={clip[3]:.6f}>0` dan
`abs(clip.xyz)<=clip.w`, jadi vertex berada di frustum sebelum pembagian w.
Turunan skalar world/view cocok dengan matriks hingga lebih dari tiga desimal;
`project()` juga cocok. Transkrip aktual: `vertex_trace.txt`.

![Vertex dan arah sumbu pada pose telusuran](figures/trace_annotated.png)

Model tampak dari samping karena yaw 90 derajat. Lingkaran menandai proyeksi vertex
yang ditelusuri, termasuk jika titik tersebut tertutup permukaan lain; ini marker
geometris, bukan klaim fragment itu lolos z-test.

## 4. Membaca kolom M dan V

| Matriks | Kolom +x (arah, w=0) | Kolom +y | Kolom +z | Origin (titik, w=1) |
|---|---|---|---|---|
| M | {vec(M[:,0])} | {vec(M[:,1])} | {vec(M[:,2])} | {vec(M[:,3])} |
| V | {vec(V[:,0])} | {vec(V[:,1])} | {vec(V[:,2])} | {vec(V[:,3])} |

M mengarahkan +x model ke -z world, +y tetap ke atas, dan +z ke +x.
Origin model pindah ke (-1,0,-1), karena vektor translasi juga diputar oleh yaw.
V tidak memutar sumbu; origin world berpindah ke (0,0,-8) pada ruang kamera.
Sebaliknya titik kamera (0,0,8,1) dipetakan V ke (0,0,0,1), sudah diperiksa.
Pada overlay, +y bergerak ke atas layar; +z bergerak ke kanan; +x dominan mengubah
depth karena kini searah -z. Pergeseran kecil x/y layar akibat perspektif konsisten
dengan perubahan jarak, bukan rotasi sumbu yang salah.

## 5. Bacaan shader: satu perbedaan dengan transforms.py

Bacaan yang dipilih adalah [source `project_vertex.glsl.js`, Three.js r160][shader],
salinan lokal bacaan tersimpan sebagai `reading/project_vertex.r160.glsl.js`.
Source ini **dibaca, tidak dijalankan**. Shader membentuk `mvPosition` dari
`transformed` dengan w=1, menerapkan matriks batching/instancing jika fitur itu aktif,
lalu mengalikan `modelViewMatrix` dan `projectionMatrix`. Hasil akhirnya ditulis ke
`gl_Position` sebagai vektor empat komponen di clip space. Di sinilah perbedaan
tanggung jawab yang paling terlihat: fungsi `project()` pada `transforms.py`
segera menghitung `clip.xyz / clip.w` dan mengembalikan NDC ke pemanggil Python,
sedangkan shader tersebut tidak menuliskan pembagian itu sama sekali.

Perbedaan ini diperlukan karena output vertex shader masih harus dapat di-clip
oleh pipeline GPU. Pembagian prematur akan menghilangkan w yang diperlukan untuk
menangani segitiga yang memotong near plane atau pinhole. Hardware kemudian
melakukan perspective divide dan viewport transform. Pada renderer NumPy proyek
ini, alur produksi di `raster3d.render()` tidak menggunakan pembagian prematur dari
helper `project()`; ia mempertahankan clip coordinates, memanggil
`clip_triangle_near()`, lalu membagi koordinat vertex hasil clipping. Artinya
helper `project()` cocok untuk telusuran titik yang sudah berada dalam frustum,
tetapi tidak cukup sebagai pengganti clipping segitiga.

`modelViewMatrix` menggabungkan V dan M, sehingga satu perkalian shader mewakili
dua perubahan ruang yang ditulis terpisah dalam laporan. Instancing menambahkan
pose tiap instance sebelum transformasi model-view tanpa perlu menggandakan
geometri. Namun konvensi matriks, arah kamera, dan nilai near/far tetap harus
dipilih benar oleh aplikasi. Source shader ini tidak memutuskan origin UV atau
lisensi texture; hal tersebut berada di pemuatan aset dan konfigurasi material.

## 6. Refleksi kasus: depth dan port konvensi

Reed menjelaskan distribusi depth sebagai remapping 1/z pada tulisan asli
**3 Juli 2015** [Depth Precision Visualized][reed]. Halaman NVIDIA berjudul
**Visualizing Depth Precision** menampilkan tanggal **21 Oktober 2021** [nvidia];
kedua tanggal ini dibedakan, bukan ditulis seolah halaman NVIDIA bertanggal 2015.
Dengan jarak positif d=-z_view, near=1 dan far=50 milik telusuran ini:

```text
D(d) = far/(far-near) - far*near/((far-near)*d)
     = 50/49 - 50/(49*d),     D(1)=0, D(50)=1
```

| Jarak d | Depth [0,1] | Kode integer 24-bit (pembulatan) |
|---|---|---|
{depth_table}

Meter pertama yang terlihat, d=1 sampai 2, menggunakan **{first_meter*100:.6f}%**
rentang kode, sekitar {round(first_meter*(2**24-1)):,} dari 16.777.216 kode.
Untuk near=0,01 dan far yang sama, meter pertama d=0,01..1,01 memakai
**{bad_near_fraction*100:.6f}%** rentang; ini menunjukkan biaya near terlalu kecil.
Estimasi lokal jarak antar-kode 24-bit dekat far=50 adalah {far_step:.9f}
satuan, dihitung dari turunan D, bukan pengukuran scene z-fighting Lab 2.

Kemen, **28 November 2012**, membahas pembalikan depth bersama format float
[Maximizing Depth Buffer Range and Precision][outerra]. Reverse-Z memetakan near
ke 1 dan far ke 0, mengubah clear ke 0 dan perbandingan ke GREATER; keuntungan
presisi utamanya muncul dengan floating point dan rentang proyeksi yang sesuai.
Membalik integer 24-bit saja hanya membalik kode. Pada OpenGL desktop, rentang
clip z [0,1] dapat dipilih dengan clip control; jangan mengasumsikan konfigurasi
itu tersedia identik pada WebGL. Karena itu reverse-Z bukan sekadar menukar dua
angka pada viewer ini.

### Empat konvensi yang dipatok

| Aspek | OpenGL/GLSL yang dipakai modul | Direct3D/DirectXMath pembanding |
|---|---|---|
| Handedness | View RH, depan kamera -z | Contoh historis LH, depan +z; modern D3D juga bisa RH |
| Matriks & operasi | Column vector, `P V M p`; upload GL column-major | DirectXMath row-major, row vector, `p M V P`; default packing HLSL column-major |
| NDC z | [-1,1] default; depth storage biasanya [0,1] | [0,1] |
| UV/texel origin | Konvensi image GL kiri bawah; NumPy membalik v sekali | Texel origin kiri atas (D3D10+) |

Handedness, urutan perkalian, dan layout memori adalah keputusan berbeda.
Dokumentasi [DirectXMath][dxmath] menjelaskan pilihan RH/LH dan packing HLSL;
[Microsoft D3D9][d3d9] memberi contoh konversi z dan winding;
[D3D10 coordinates][d3d10] mendefinisikan origin texel. UV pada berkas mesh
tetap sebuah parameterisasi; port perlu menyesuaikan orientasi data citra/sampler,
bukan membalik v secara acak di beberapa tempat.

Scene pengungkap bug: dua quad hampir coplanar jauh untuk hilangnya presisi;
geometri/plakat asimetris dan label F untuk mirror; titik dengan hasil `P V M`
yang sudah diketahui untuk kesalahan transpose/urutan; titik tepat pada near/far
untuk rentang NDC; texture bertanda TOP dan empat warna sudut untuk origin UV.
Pengujian ini sengaja memisahkan kesalahan yang bisa tampak serupa pada kubus simetris.

Perubahan otomatis yang dibuat di `run_lab1.py`: guard menerima M jika determinan
bagian 3x3 positif dan posisi landmark plakat cocok turunan skalar independen.
Kontrol negatif `M @ diag(-1,1,1,1)` wajib ditolak. Pada run ini pose benar diterima
dan mirror ditolak; titik hasil mirror akan mendarat di
{vec(mirror_screen[:2])}, sedangkan pose benar di {vec(scr[:2])}.
Ini bukti numerik terhadap mirror sumbu x yang disuntikkan, bukan klaim menangkap
semua kemungkinan bug konvensi; titik near/far dan texture berlabel tetap dibutuhkan.

## 7. Reproduksi dan hasil verifikasi

Jalankan dari direktori Module3 pada PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe run_lab1.py
.\\.venv\\Scripts\\python.exe worked_example.py --model data/mymodel.obj --vertex {vi}
```

Untuk versi dependensi yang sama persis, ganti file install dengan
`lab1/requirements.lock.txt`. Seluruh generator memakai path relatif terhadap
lokasi skrip, jadi `run_lab1.py` bisa dipanggil dari folder lain.

Hasil: **{len(checks)} pemeriksaan Lab 1 lulus**. Pratinjau memuat
{int(np.isfinite(depth).sum()):,} pixel terlihat. Matriks/hasil numerik dan setiap
status pemeriksaan disimpan di `results.json`; hash aset di `sha256.json`.
Ini verifikasi teknis Lab 1, bukan klaim Lab 2/3 atau penilaian dosen selesai.
Pemahaman, pengulangan hitung tangan, dan pembelaan lisan tetap dilakukan mahasiswa.

Pengujian regresi starter dijalankan melalui `python make_model.py` lalu
`python tests.py`: delapan pemeriksaan melaporkan OK, tanpa skip pada run itu.
Transkrip tersimpan di `starter_tests.txt`. `ref/temple_ref.ppm` dibuat oleh
rasteriser starter pada run pertama di komputer ini, sehingga pemeriksaan frame
starter merupakan konsistensi terhadap baseline lokal, bukan verifikasi visual
independen. `data/pusaka_model.obj` tetap merupakan fallback 180 segitiga dan
tidak menggantikan `data/mymodel.obj` 636 segitiga milik Lab 1 ini.

## Sumber bacaan

Semua sumber daring diakses 15 November 2026. Link menunjuk sumber yang dibaca;
shader memakai tag versi tetap, sementara dokumentasi API dapat diperbarui.

- [Reed, Depth Precision Visualized, 3 Juli 2015][reed].
- [NVIDIA, Visualizing Depth Precision, 21 Oktober 2021][nvidia].
- [Brano Kemen, Maximizing Depth Buffer Range and Precision, 28 November 2012][outerra].
- [Three.js, project_vertex.glsl.js, tag r160][shader].
- [Microsoft, Getting started with DirectXMath, bagian Matrix convention][dxmath].
- [Microsoft, Coordinate Systems (Direct3D 9), pembaruan 6 Januari 2021][d3d9].
- [Microsoft, Coordinate Systems (Direct3D 10), pembaruan 24 Mei 2021][d3d10].
- [Direktori Majapahit, Candi Bajang Ratu, 2021][bajang].

[reed]: https://www.reedbeta.com/blog/depth-precision-visualized/
[nvidia]: https://developer.nvidia.com/blog/visualizing-depth-precision/
[outerra]: https://outerra.blogspot.com/2012/11/maximizing-depth-buffer-range-and.html
[shader]: https://github.com/mrdoob/three.js/blob/r160/src/renderers/shaders/ShaderChunk/project_vertex.glsl.js
[dxmath]: https://learn.microsoft.com/en-us/windows/win32/dxmath/pg-xnamath-getting-started
[d3d9]: https://learn.microsoft.com/en-us/windows/win32/direct3d9/coordinate-systems
[d3d10]: https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates
[bajang]: https://www.direktorimajapahit.id/petabudaya/candi-bajang-ratu
'''
    (OUT/'Laporan_Lab1.md').write_text(report,encoding='utf-8')
    print(trace)
    print(f'\nLAB 1: {len(checks)} CHECKS PASSED')
    print(f'Model: {mesh.n_tris} triangles; UV minimum area: {uv_area.min():.6f}')
    print(f'Report: {OUT / "Laporan_Lab1.md"}')


if __name__ == '__main__':
    run()
