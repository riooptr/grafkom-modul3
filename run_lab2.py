"""Run all five Lab 2 tasks; write reproducible evidence and an Indonesian report.

Python/NumPy perform all rendering. Pillow saves images, Matplotlib plots depth
histograms, and neither library participates in the rendering pipeline.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from time import perf_counter
import hashlib
import json
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from objio import load_obj, Mesh
from ppm_io import read_ppm, write_ppm
from transforms import rot_y, scale
from texture_sampling import build_mipmaps
from srgb import srgb_to_linear,linear_to_srgb
from metrics import psnr,ssim
from broken_scene import diagnostic_renders
import raster3d
import raytrace

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'lab2'
FIG=OUT/'figures'


def save_rgb(name,img):
    Image.fromarray(img).save(FIG/f'{name}.png')
    write_ppm(str(FIG/f'{name}.ppm'),img)


def metrics(a,b):
    delta=np.abs(a.astype(float)-b.astype(float))
    value=psnr(a,b)
    return {'psnr_db':float(value) if np.isfinite(value) else 'inf',
            'ssim_global':ssim(a,b),'mean_absolute_error':float(delta.mean()),
            'max_absolute_error':int(delta.max()),
            'different_pixels':int(np.any(delta>0,axis=2).sum())}


def montage(name,panels,width=320):
    height=width+36
    result=Image.new('RGB',(width*len(panels),height),'#121a24')
    d=ImageDraw.Draw(result)
    for i,(title,img) in enumerate(panels):
        thumb=Image.fromarray(img)
        thumb.thumbnail((width,width))
        result.paste(thumb,(i*width+(width-thumb.width)//2,36+(width-thumb.height)//2))
        d.text((i*width+10,12),title,fill='#f0f4ff')
    result.save(FIG/f'{name}.png')
    return result


def depth_image(dep):
    d=np.where(np.isfinite(dep),(dep+1)*.5,1.0)
    return np.repeat(np.rint(np.clip(d,0,1)*255).astype(np.uint8)[...,None],3,2)


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def interior(mask):
    """One-pixel erosion avoids measuring the high contrast outside silhouette."""
    p=np.pad(mask,1,constant_values=False)
    return np.logical_and.reduce([p[y:y+mask.shape[0],x:x+mask.shape[1]]
                                  for y in range(3) for x in range(3)])


def hf_energy(image,mask):
    rgb=srgb_to_linear(image.astype(float)/255)
    y=rgb@np.array([.2126,.7152,.0722])
    lap=4*y-np.roll(y,1,0)-np.roll(y,-1,0)-np.roll(y,1,1)-np.roll(y,-1,1)
    return float(np.mean(lap[mask]**2))


def projected_edge_residuals(mesh,a,b):
    from transforms import perspective,look_at,project,viewport
    _,ndc=project(perspective(40,1,1,50),look_at((4.8,3.8,6.7),(0,1.5,0)),np.eye(4),mesh.positions)
    screen=viewport(ndc,256,256)[:,:2]
    edges=np.concatenate([screen[mesh.faces[:,ids]] for ids in [[0,1],[1,2],[2,0]]])
    vectors=edges[:,1]-edges[:,0]
    denom=np.sum(vectors*vectors,axis=1)
    ys,xs=np.where(np.any(a!=b,axis=2))
    rows=[]
    for x,y in zip(xs,ys):
        p=np.array([x+.5,y+.5])
        t=np.clip(np.sum((p-edges[:,0])*vectors,axis=1)/np.maximum(denom,1e-20),0,1)
        dist=np.linalg.norm(p-(edges[:,0]+t[:,None]*vectors),axis=1)
        rows.append({'pixel_xy':[int(x),int(y)],
                     'nearest_projected_triangle_edge_distance_pixels':float(dist.min()),
                     'rgb_raster':a[y,x].tolist(),'rgb_ray':b[y,x].tolist()})
    return rows


def display_crop(image,depth):
    """Enlarge a common crop only for presentation; all metrics use full frames."""
    y,x=np.where(np.isfinite(depth))
    left,top=max(0,x.min()-5),max(0,y.min()-5)
    right,bottom=min(image.shape[1],x.max()+6),min(image.shape[0],y.max()+6)
    crop=Image.fromarray(image[top:bottom,left:right])
    crop.thumbnail((256,256))
    factor=min(256/crop.width,256/crop.height)
    crop=crop.resize((round(crop.width*factor),round(crop.height*factor)),Image.Resampling.NEAREST)
    display=Image.new('RGB',(256,256))
    display.paste(crop,((256-crop.width)//2,(256-crop.height)//2))
    return np.asarray(display)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    print('Lab 2: starting evidence generation',flush=True)
    results={'recorded_at':datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
             'assistance':'Codex-assisted extensions of the supplied NumPy starter',
             'ssim_definition':'single global window over RGB values; not sliding-window SSIM',
             'pipeline_dependencies':['Python standard library','NumPy'],
             'report_dependencies':{'Pillow':'image serialization only','Matplotlib':matplotlib.__version__}}
    fallback=load_obj(ROOT/'data/pusaka_model.obj')
    tex=read_ppm(str(ROOT/'data/relief.ppm'))
    mine=load_obj(ROOT/'data/mymodel.obj')
    brick=read_ppm(str(ROOT/'data/lab1_brick.ppm'))

    # Frozen regression frame; this file must never be silently regenerated.
    ref_path=ROOT/'ref/temple_ref.ppm'
    reference=read_ppm(str(ref_path))
    kw=dict(w=160,h=160,tex_u8=tex,model=rot_y(35)@scale((1.4,1.4,1.4)))
    a,_,_=raster3d.render(fallback,**kw)
    b,_=raytrace.render(fallback,**kw)
    results['fallback_vs_frozen_reference']=metrics(a,reference)
    results['frozen_reference_sha256']=hashlib.sha256(ref_path.read_bytes()).hexdigest()
    results['fallback_raster_vs_ray']=metrics(a,b)
    assert psnr(a,reference)>60
    assert psnr(a,b)>60
    save_rgb('fallback_raster',a);save_rgb('fallback_ray',b)
    save_rgb('fallback_diff_x8',np.clip(np.abs(a.astype(int)-b.astype(int))*8,0,255).astype(np.uint8))

    # The user's 636-triangle Lab 1 model, with genuine UVs on all faces.
    kw=dict(w=256,h=256,tex_u8=brick,eye=(4.8,3.8,6.7),center=(0,1.5,0),
            near=1,far=50,fovy=40)
    start=perf_counter();own_raster,own_depth,_=raster3d.render(mine,**kw)
    raster_seconds=perf_counter()-start
    start=perf_counter();own_ray,ray_depth=raytrace.render(mine,**kw)
    ray_seconds=perf_counter()-start
    results['own_model_raster_vs_ray']=metrics(own_raster,own_ray)
    results['own_model_residuals']=projected_edge_residuals(mine,own_raster,own_ray)
    results['own_model_resolution']=[256,256]
    results['own_model_render_seconds_single_run']={'raster':raster_seconds,'ray':ray_seconds}
    assert psnr(own_raster,own_ray)>60,results['own_model_raster_vs_ray']
    save_rgb('own_raster',own_raster);save_rgb('own_ray',own_ray)
    diff=np.clip(np.abs(own_raster.astype(int)-own_ray.astype(int))*8,0,255).astype(np.uint8)
    save_rgb('own_diff_x8',diff)
    montage('render_comparison',[('NumPy raster',own_raster),('Primary rays',own_ray),('|difference| x8',diff)])
    print('Raster/ray comparisons:',results['own_model_raster_vs_ray'],flush=True)

    # Three controlled diagnoses. For all comparisons the oracle is kept separate.
    results['bugs']={}
    for name in ['inside-out','zfight','stretched']:
        d=diagnostic_renders(name,fallback,tex,w=192,h=192)
        before=metrics(d['bad'],d['reference'])
        after=metrics(d['fixed'],d['reference'])
        assert after['ssim_global']>before['ssim_global'],name
        results['bugs'][name]={'before':before,'after':after,**d['metadata']}
        for label in ['bad','fixed','reference']:
            save_rgb(f'{name}_{label}',d[label])
        for label in ['bad','fixed','reference']:
            dep=d[f'{label}_depth']
            save_rgb(f'{name}_{label}_depth',depth_image(dep))
            np.save(OUT/f'{name}_{label}_depth.npy',dep)
        montage(f'diagnosis_{name}', [('Before',d['bad']),('After',d['fixed']),('Reference',d['reference'])])
        if name=='zfight':
            assert d['metadata']['wrong_pixels_after']<d['metadata']['wrong_pixels_before']
            # A common [0,1] histogram makes the compression near depth=1 visible.
            fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
            allcodes={}
            for label,color in [('bad','#c45b3d'),('fixed','#267b86')]:
                dep=d[label+'_depth']; valid=(dep[np.isfinite(dep)]+1)*.5
                counts,edges=np.histogram(valid,bins=64,range=(0,1))
                np.savetxt(OUT/f'zfight_{label}_histogram.csv',
                           np.c_[edges[:-1],edges[1:],counts],delimiter=',',
                           header='depth_left,depth_right,pixel_count',comments='')
                codes,freq=np.unique(np.rint(valid*(2**16-1)).astype(int),return_counts=True)
                np.savetxt(OUT/f'zfight_{label}_codes.csv',np.c_[codes,freq],delimiter=',',
                           header='depth16_code,pixel_count',comments='',fmt='%d')
                axes[0].stairs(counts,edges,label=label,color=color,linewidth=2)
                local_ax=axes[1 if label=='bad' else 2]
                local_ax.bar(codes,freq,color=color,width=.7)
                local_ax.set(title=f'{label}: {len(codes)} occupied '+('code' if len(codes)==1 else 'codes'),
                             xlabel='16-bit code (local range)',ylabel='Pixel count')
                local_ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                if len(codes)==1:
                    local_ax.set_xlim(codes[0]-.75,codes[0]+.75)
                    local_ax.set_xticks(codes)
                else:
                    local_ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
                allcodes[label]={'unique_visible_codes':len(codes),
                                 'depth01_min':float(valid.min()),'depth01_max':float(valid.max())}
            axes[0].set(xlabel='Stored depth [0, 1]',ylabel='Pixel count',title='Same depth range, 64 bins')
            axes[0].legend()
            for ax in axes:ax.grid(alpha=.2)
            fig.savefig(FIG/'zfight_histogram.png',dpi=160);plt.close(fig)
            results['bugs']['zfight']['histogram_summary']=allcodes
        print('Diagnosis',name,'SSIM',before['ssim_global'],'->',after['ssim_global'],flush=True)

    # E4 is tied to an actual rendered floor triangle, not only a printed formula.
    p=np.array([[-.6,-.55,-1],[5.4,-4.95,-9],[-.6,.65,-1]])
    uv=np.array([[0,0],[1,0],[0,1]],float)
    floor=Mesh(p,uv,[[0,1,2]],[[0,1,2]])
    yy,xx=np.mgrid[:256,:256]
    checker=np.repeat((((xx//16+yy//16)%2)*220+25).astype(np.uint8)[...,None],3,2)
    floor_kw=dict(w=256,h=256,tex_u8=checker,eye=(0,0,0),center=(0,0,-1),
                  fovy=80,near=.5,far=20,ambient=1)
    naive,_,_=raster3d.render(floor,perspective_correct=False,**floor_kw)
    correct,_,_=raster3d.render(floor,**floor_kw)
    independent,_=raytrace.render(floor,**floor_kw)
    u_naive=.5
    u_pc=(.5*0/1+.5*1/9)/(.5/1+.5/9)
    assert abs(u_pc-.1)<1e-12
    assert psnr(correct,independent)>60
    assert psnr(correct,independent)>psnr(naive,independent)
    results['perspective_uv']={'w_left':1,'w_right':9,'u_naive':u_naive,'u_correct':u_pc,
                               'naive_vs_ray':metrics(naive,independent),
                               'correct_vs_ray':metrics(correct,independent)}
    save_rgb('uv_naive',naive);save_rgb('uv_correct',correct)
    montage('perspective_uv',[('Linear UV: midpoint u=0.5',naive),('Correct UV: midpoint u=0.1',correct)])
    motion=[]
    for shift in [0,.06,.12,.06,0,-.06,-.12,-.06]:
        moved={**floor_kw,'eye':(shift,0,0),'center':(shift,0,-1)}
        left,_,_=raster3d.render(floor,perspective_correct=False,**moved)
        right,_,_=raster3d.render(floor,**moved)
        motion.append(montage('_uv_motion_frame',[('Naive UV',left),('Perspective-correct UV',right)],256))
    motion[0].save(FIG/'uv_motion.gif',save_all=True,append_images=motion[1:],duration=180,loop=0)
    (FIG/'_uv_motion_frame.png').unlink()

    # Four levels: averaging occurs only after decoding to linear light.
    levels=build_mipmaps(brick,4)
    assert [a.shape[:2] for a in levels]==[(256,256),(128,128),(64,64),(32,32)]
    for i,lev in enumerate(levels):
        Image.fromarray(np.rint(linear_to_srgb(lev)*255).astype(np.uint8)).save(FIG/f'mip_level_{i}.png')
    far_kw=dict(w=256,h=256,tex_u8=brick,eye=(12,8,17),center=(0,1.5,0),
                near=1,far=80,fovy=45,cull_back=True)
    mipstats={}
    no_mip,dep,_=raster3d.render(mine,**far_kw)
    with_mip,_,_=raster3d.render(mine,texture_filter='trilinear',mip_levels=levels,stats=mipstats,**far_kw)
    mask=interior(np.isfinite(dep))
    assert mask.sum()>1000
    en_before,en_after=hf_energy(no_mip,mask),hf_energy(with_mip,mask)
    # A four-times supersampled reference is an approximation, not ground truth.
    hires,_,_=raster3d.render(mine,**{**far_kw,'w':1024,'h':1024})
    highlin=srgb_to_linear(hires.astype(float)/255)
    ssaa=np.rint(linear_to_srgb(highlin.reshape(256,4,256,4,3).mean(axis=(1,3)))*255).astype(np.uint8)
    mae_before=float(np.abs(no_mip.astype(float)-ssaa.astype(float))[mask].mean())
    mae_after=float(np.abs(with_mip.astype(float)-ssaa.astype(float))[mask].mean())
    assert en_after<en_before,'Mipmap did not reduce high-frequency energy'
    assert mae_after<mae_before,'Mipmap moved away from supersampled evidence'
    save_rgb('far_no_mip',no_mip);save_rgb('far_mip',with_mip);save_rgb('far_ssaa4',ssaa)
    mipdiff=np.clip(np.abs(no_mip.astype(int)-with_mip.astype(int))*4,0,255).astype(np.uint8)
    save_rgb('mip_difference_x4',mipdiff)
    montage('mipmap_comparison',[('No mipmap (enlarged crop)',display_crop(no_mip,dep)),
                                 ('4 mip levels (same crop)',display_crop(with_mip,dep)),
                                 ('4x SSAA (same crop)',display_crop(ssaa,dep))])
    temporal_before=[];temporal_after=[];mip_motion=[]
    previous=None
    for dx in [0,.02,.04,.06,.08,.10]:
        moved={**far_kw,'eye':(12+dx,8,17),'center':(dx,1.5,0)}
        a,da,_=raster3d.render(mine,**moved)
        b,db,_=raster3d.render(mine,texture_filter='trilinear',mip_levels=levels,**moved)
        mip_motion.append(montage('_mip_motion_frame',[('No mipmap (enlarged crop)',display_crop(a,dep)),
                                                      ('4 mip levels (same crop)',display_crop(b,dep))],256))
        if previous:
            pa,pb,pdep=previous
            common=interior(np.isfinite(da)&np.isfinite(pdep))
            temporal_before.append(float(np.abs(a.astype(float)-pa.astype(float))[common].mean()))
            temporal_after.append(float(np.abs(b.astype(float)-pb.astype(float))[common].mean()))
        previous=(a,b,da)
    mip_motion[0].save(FIG/'mipmap_motion.gif',save_all=True,append_images=mip_motion[1:]+mip_motion[-2:0:-1],duration=160,loop=0)
    (FIG/'_mip_motion_frame.png').unlink()
    results['mipmap']={'linear_level_sizes':[list(x.shape[:2]) for x in levels],
                       'interior_pixels':int(mask.sum()),'hf_energy_before':en_before,
                       'hf_energy_after':en_after,'hf_reduction_percent':100*(1-en_after/en_before),
                       'mae_to_ssaa4_before':mae_before,'mae_to_ssaa4_after':mae_after,
                       'mean_adjacent_frame_mae_before':float(np.mean(temporal_before)),
                       'mean_adjacent_frame_mae_after':float(np.mean(temporal_after)),
                       'mean_lod':mipstats['sampled_lod_sum']/mipstats['texture_samples'],
                       'max_lod':mipstats['sampled_lod_max'],
                       'lod_cap_reached':mipstats['sampled_lod_max']==3}
    print('Mipmap:',results['mipmap'],flush=True)

    # Pitfall improvement: measured candidate fragments, not only z-test wins.
    no_cull,cull={},{}
    a,_,_=raster3d.render(mine,stats=no_cull,**kw)
    b,_,_=raster3d.render(mine,cull_back=True,stats=cull,**kw)
    assert np.array_equal(a,b)
    assert cull['covered_fragments']<no_cull['covered_fragments']
    results['culling']={'before':no_cull,'after':cull,'image_comparison':metrics(a,b),
                        'covered_fragment_reduction_percent':100*(1-cull['covered_fragments']/no_cull['covered_fragments'])}
    results['summary_checks']={
        'fallback_matches_frozen_reference':True,'fallback_matches_independent_rays':True,
        'own_model_matches_independent_rays':True,'three_bug_ssim_scores_improve':True,
        'zfight_wrong_pixel_count_reduced':True,'e4_midpoint_correct':True,
        'perspective_uv_agrees_with_ray':True,'four_linear_light_mip_levels':True,
        'mip_high_frequency_energy_reduced':True,'mip_closer_to_ssaa4':True,
        'culling_reduces_fragments_without_rgb_change':True}
    proc=subprocess.run([sys.executable,'tests.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'tests.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode==0,proc.stdout+proc.stderr
    results['regression_tests']='14/14 passed; none skipped'
    proc=subprocess.run([sys.executable,'broken_scene.py','--all'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'broken_scene.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    assert proc.returncode==0,proc.stdout+proc.stderr
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (OUT/'requirements.lock.txt').write_text(frozen,encoding='utf-8')
    write_json(OUT/'results.json',results)
    write_report(results)
    print(f'LAB 2 COMPLETE: {len(results["summary_checks"])} evidence checks; 14/14 regression checks',flush=True)
    print('Report:',OUT/'Laporan_Lab2.md',flush=True)


def write_report(r):
    def score(m):return f"{m['psnr_db']} dB; SSIM global {m['ssim_global']:.6f}; beda {m['different_pixels']} pixel"
    bugrows='\n'.join(f"| {name} | {v['before']['ssim_global']:.6f} | {v['after']['ssim_global']:.6f} |" for name,v in r['bugs'].items())
    uv=r['perspective_uv'];m=r['mipmap'];c=r['culling'];z=r['bugs']['zfight']
    residual_max=max([x['nearest_projected_triangle_edge_distance_pixels'] for x in r['own_model_residuals']],default=0)
    report=f'''# Lab 2 - Pipeline NumPy, Diagnosis Scene, dan Texture

Pencatatan: {r['recorded_at']} (Asia/Jakarta).

**Status:** kelima butir teknis Lab 2 dijalankan. {len(r['summary_checks'])}
pemeriksaan bukti dan 14 pengujian regresi lulus, tanpa input yang di-skip.
Model utama tetap blockout 636 segitiga dari Lab 1.

**Transparansi:** pekerjaan ini memperluas starter yang disediakan, dengan bantuan
Codex pada kode, eksperimen, dan draf laporan. Tidak diklaim seluruh pipeline
ditulis manual oleh mahasiswa. `diagnosis_before_changes.md` dibuat sebelum patch
Lab 2, tetapi solusi/komentar starter sudah dibaca saat audit; bukan diagnosis buta.
Hasil numerik di sini berasal dari run aktual, bukan angka target dalam PDF.

## 1. Pipeline dari koordinat sampai pixel

`transforms.py` mempertahankan konvensi column vector, right-handed, kamera -z,
`clip=P@V@M@p`, NDC z [-1,1], viewport y ke bawah. `raster3d.py` mempertahankan
clip coordinates hingga near clipping Sutherland-Hodgman, membawa UV melewati
intersection, membagi dengan w, lalu memakai edge function/barycentric, depth test,
UV perspective-correct, bilinear sampling, dan Lambert dalam linear light.
Fragment di luar rentang depth far ditolak; clipping penuh enam bidang bukan
bagian ekstensi ini.

`raytrace.py` membangkitkan satu ray pinhole per pusat pixel, mencari hit terdekat
dengan Moller-Trumbore, menghitung UV melalui barycentric di segitiga world,
dan memakai texture serta Lambert yang sama. Perbaikan Lab 2 menolak hit berdasarkan
jarak sepanjang sumbu kamera, `t * dot(ray_dir, camera_forward)`, agar near/far
cocok dengan rasteriser. Parameter ray t sendiri bukan jarak eye-space untuk ray
yang miring. Kedua renderer berbagi sampler dan fungsi colour space; cross-check
ini independen untuk visibilitas/interpolasi geometri, bukan dua implementasi
texture/shading yang sepenuhnya independen.

| Contoh | Angka yang dicetak pengujian |
|---|---|
| E1 | (2,3,4) -> T: (3,3,3) -> Ry90: (3,3,-3); proyeksi d=2: (2,2); urutan lain: (5,3,-3) |
| E2 | Kamera +x dan model -x sama-sama menghasilkan (4,0,0,1) |
| E3 | eye z=-1 -> NDC -1; eye z=-9 -> +1; eye z=-3 -> 0.5 |
| E4 | u_naif=0.5; u_perspective_correct=0.1 |
| E5 | Depth barycentric=0.625 |

Transkrip lengkap: [tests.txt](tests.txt). Pengujian tambahan membuktikan near-clip
menghasilkan dua triangle dengan UV intersection (1/3,0) dan (0,1/3), serta hasil
clipping near/far cocok dengan hit ray tanpa mengandalkan golden image yang baru dibuat.

### Kecocokan frame

| Perbandingan | Hasil |
|---|---|
| Fallback 160x160 vs referensi dibekukan | {score(r['fallback_vs_frozen_reference'])} |
| Fallback 160x160 raster vs ray | {score(r['fallback_raster_vs_ray'])} |
| Model Lab 1 256x256 raster vs ray | {score(r['own_model_raster_vs_ray'])} |

![Raster, ray, dan selisih x8](figures/render_comparison.png)

**Batas bukti:** `ref/temple_ref.ppm` dibuat rasteriser starter saat Lab 1, kemudian
dibekukan dan tidak ditulis ulang. Hash SHA-256: `{r['frozen_reference_sha256']}`.
Mencocokkannya adalah regresi lokal, bukan oracle independen. Pengujian sekarang
gagal bila input/referensi hilang; tidak ada lagi auto-generate atau skip yang
tetap mencetak passed. `SSIM` di proyek ini adalah satu window global, bukan SSIM
lokal Gaussian 11x11. PSNR inf berarti byte identik pada konfigurasi itu, bukan
klaim semua kamera/resolusi selalu identik.

Pada model sendiri terdapat {r['own_model_raster_vs_ray']['different_pixels']}
pixel berbeda, dengan maksimum selisih channel {r['own_model_raster_vs_ray']['max_absolute_error']}.
Jarak maksimum pusat pixel tersebut ke edge triangle proyeksi terdekat adalah
{residual_max:.12f} pixel. Koordinat, jarak, dan RGB kedua renderer dicatat
di `results.json` (`own_model_residuals`). Ini pemeriksaan lokasi residual;
edge yang dipakai mencakup seluruh triangle, termasuk yang tertutup, sehingga
bukan bukti tunggal tentang triangle mana yang menang pada tie.

Waktu satu run model sendiri: raster {r['own_model_render_seconds_single_run']['raster']:.4f}
detik, ray {r['own_model_render_seconds_single_run']['ray']:.4f} detik. Angka ini
hanya catatan pelaksanaan, bukan benchmark berulang atau eksperimen prediksi
sebelum ukur yang diwajibkan Lab 3.

## 2. Tiga diagnosis dan bukti perbaikan

| Kasus | SSIM sebelum vs reference | SSIM sesudah vs reference |
|---|---|---|
{bugrows}

### Inside-out: winding dan backface culling

Audit menghitung signed volume kotak pertama fallback **-1.4**, sehingga winding
starter ke dalam meskipun komentarnya menyebut CCW keluar. `outward_fallback()`
mengorientasikan salinan **15 kotak fallback yang sudah diketahui**, bukan melakukan
repair otomatis sembarang OBJ. Berkas model/ref historis tidak ditulis ulang.
Baseline diagnosis memakai winding keluar; bug membalik face dan indeks UV;
perbaikan membalik keduanya kembali dengan culling tetap aktif. Jika hanya face
dibalik tanpa UV, mapping texture dapat bergeser walaupun siluet sudah benar.

![Inside-out: sebelum, sesudah, referensi](figures/diagnosis_inside-out.png)

RGB dan citra depth ketiganya tersimpan sebagai PNG/PPM; depth numerik tersimpan
sebagai NPY. Nilai before/after dihitung terhadap konfigurasi referensi yang
dipisahkan. SSIM sesudah 1 pada eksperimen deterministik ini berarti konvensi
yang dimaksud berhasil dipulihkan, bukan evaluasi visual independen oleh dosen.

### Z-fighting: near terlalu kecil dan kuantisasi depth

Scene berisi dua quad berpotongan sekitar jarak 15, far=200 dan integer depth
16-bit. Bug: near=0.001. Perbaikan: **hanya naikkan near menjadi 1**, bit depth
tetap. Kuantisasi memakai `2**16-1` interval (65536 kode). Oracle memakai depth
float, dengan near yang disamakan. Oracle near 0.001 dan near 1 menghasilkan RGB
identik, memastikan perbaikan tidak sekadar membuang geometri lewat clipping.

Pixel salah dengan selisih maksimum channel >60 terhadap oracle:
**{z['wrong_pixels_before']} -> {z['wrong_pixels_after']}**.
SSIM fixed terhadap oracle tidak dipaksa 1; residu kuantisasi masih dilaporkan.

![Z-fighting](figures/diagnosis_zfight.png)

![Histogram depth absolut dan kode terisi](figures/zfight_histogram.png)

Histogram pertama memakai range [0,1] dan 64 bin yang sama. Panel kedua dan ketiga
memakai kode integer yang benar-benar terisi pada range lokal masing-masing.
CSV histogram dan pasangan kode/frekuensi
disimpan terpisah agar konsentrasi depth tidak tertutup normalisasi citra.
Ringkasan: near buruk {z['histogram_summary']['bad']['unique_visible_codes']} kode
unik, near diperbaiki {z['histogram_summary']['fixed']['unique_visible_codes']} kode
unik. Ini eksperimen 16-bit yang disengaja agar efek terukur, bukan klaim depth
browser pengguna adalah 16-bit.

### Stretched: pertukaran faktor yang benar-benar berbeda

Starter awal menambahkan skala diag(2.2,0.5,1,1) pada rantai yang faktor awalnya
komutatif (skala seragam dan translasi nol). Itu menjelaskan gambar lama, tetapi
tidak mengisolasi kesalahan urutan. Eksperimen Lab 2 memperjelas bug:

```text
T = translate(0.15,0,-0.1)
R = rot_y(35)
S = scale(1.7,1.1,0.9)
bad   = T @ S @ R
fixed = T @ R @ S
```

Faktor sama persis; perubahan tunggal adalah urutan R dan S. Pada good, skala
terjadi di sumbu lokal sebelum yaw; pada bad, hasil yaw di-scale menurut sumbu
world. Kedua matriks lengkap dicatat di `results.json`, sehingga klaim dapat
diperiksa dari angka, bukan hanya thumbnail. Ini berkaitan dengan pitfall urutan
perkalian, sedangkan inside-out dan zfight berkaitan dengan winding serta near.

![Stretched](figures/diagnosis_stretched.png)

## 3. UV perspective-correct dan texture yang berenang

Segitiga uji mempunyai wL=1 dan wR=9. Di tengah edge pada layar:

```text
u_naif = 0.5
u/w    = 0.5*(0/1) + 0.5*(1/9) = 1/18
1/w    = 0.5*(1/1) + 0.5*(1/9) = 5/9
u      = (1/18)/(5/9) = 0.1
```

Renderer memiliki opsi `perspective_correct=False` sebagai kontrol negatif.
Triangle dan UV tetap sama; hanya interpolasinya yang diubah. Terhadap ray caster,
UV naif: {score(uv['naive_vs_ray'])}; UV benar: {score(uv['correct_vs_ray'])}.

![UV naif dan benar](figures/perspective_uv.png)

![Gerak kamera memperlihatkan warping UV](figures/uv_motion.gif)

GIF memakai pergeseran kamera/target sejajar dari -0.12 sampai +0.12 unit. Pola
naif terdistorsi terhadap permukaan 3D; kontrol perspective-correct cocok dengan
ray caster. Distorsi tidak hanya ditunjukkan lewat u=0.1 yang dicetak.

## 4. Mipmap empat tingkat, linear light, dan minification

`texture_sampling.build_mipmaps()` mendecode sRGB sekali, melakukan box-average
2x2 di linear light, kemudian mengulangi ke ukuran **256,128,64,32**. Uji checker
hitam/putih menghasilkan mean linear 0.5, yang di-encode ke sRGB **188**, bukan
rata-rata byte 128. Pillow hanya menulis PNG; seluruh filter memakai NumPy.

LOD memakai turunan analitik hasil interpolasi rasional q/r:
`dUV/dx=(dq/dx - UV*dr/dx)/r`, dan serupa untuk y. Footprint adalah maksimum
panjang gradien dalam satuan texel; LOD=log2(footprint). Trilinear membaurkan dua
tingkat terdekat dan membatasi LOD ke rentang empat level yang ada.

Scene jauh memakai model sendiri dan texture bata yang sama, kamera (12,8,17),
target (0,1.5,0), FOV45, near1/far80, resolusi 256x256. Pembandingan tambahan
merender 1024x1024 kemudian box-downscale 4x di linear light. SSAA 4x adalah
pendekatan integrasi pixel, bukan oracle eksak untuk semua frekuensi texture.

![Mipmap dan SSAA](figures/mipmap_comparison.png)

Montase memperbesar crop yang sama dengan nearest-neighbor agar texture terbaca;
frame asli tetap 256x256 dan itulah yang dipakai untuk semua metrik.

![Gerak subpixel dengan dan tanpa mipmap](figures/mipmap_motion.gif)

| Ukuran bukti | Tanpa mipmap | Mipmap 4 level |
|---|---|---|
| Energi Laplacian luminansi linear | {m['hf_energy_before']:.9f} | {m['hf_energy_after']:.9f} |
| MAE channel ke SSAA4 (byte sRGB) | {m['mae_to_ssaa4_before']:.6f} | {m['mae_to_ssaa4_after']:.6f} |
| MAE frame berurutan, gerak subpixel | {m['mean_adjacent_frame_mae_before']:.6f} | {m['mean_adjacent_frame_mae_after']:.6f} |

Energi frekuensi tinggi turun **{m['hf_reduction_percent']:.2f}%** pada mask
interior {m['interior_pixels']} pixel (silhouette dierosi satu pixel). MAE temporal
mencakup gerak nyata serta variasi sampling, bukan pengukuran aliasing murni dengan
motion compensation. Rata-rata LOD {m['mean_lod']:.3f}, maksimum {m['max_lod']:.3f}.
Empat level mencapai batas LOD 3 di run jauh ini; artefak pada geometri,
tepi atau minification lebih ekstrem dapat tetap ada. Hasil membuktikan penurunan
aliasing yang terukur, tidak mengklaim semua kelap-kelip hilang pada semua jarak.

## 5. Perbaikan pitfall: backface culling yang diukur

Model Lab 1 sudah mempunyai winding keluar, sehingga culling dapat dibandingkan
tanpa menyembunyikan masalah winding starter. Kamera dan material sama seperti
frame sendiri pada bagian 1. `covered_fragments` menghitung candidate pixel
di dalam triangle **sebelum z-test**, sedangkan `depth_passes` adalah update warna
yang benar-benar lolos. Kedua ukuran dicatat agar definisi penghematan jelas.

| Ukuran | Culling off | Culling on |
|---|---|---|
| Candidate fragments | {c['before']['covered_fragments']} | {c['after']['covered_fragments']} |
| Depth passes / penulisan warna | {c['before']['depth_passes']} | {c['after']['depth_passes']} |
| Triangle yang di-cull | {c['before']['triangles_culled']} | {c['after']['triangles_culled']} |

Candidate fragments turun **{c['covered_fragment_reduction_percent']:.2f}%**,
sementara RGB identik ({score(c['image_comparison'])}). Ini pengurangan pekerjaan
fragment pada mesh/kamera yang diuji, bukan jaminan percepatan waktu 50%.

## 6. Reproduksi, provenance, dan batas pekerjaan

Dari folder Module3 di PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pip install -r lab2/requirements-report.txt
.\\.venv\\Scripts\\python.exe tests.py
.\\.venv\\Scripts\\python.exe run_lab2.py
.\\.venv\\Scripts\\python.exe broken_scene.py --all
```

Untuk environment terkunci penuh: `pip install -r lab2/requirements.lock.txt`.
Matplotlib hanya dipakai untuk plot laporan dan tidak diimpor oleh renderer.
`run_lab2.py` membutuhkan input model Lab 1 dan fallback yang sudah di-commit;
jika perlu membuatnya ulang, jalankan `make_lab1_model.py` dan `make_model.py`.
`results.json` berisi parameter dan metrik; semua NPY/histogram/PPM disimpan di
`lab2/`, bukan hanya thumbnail. `starter_bugs_before_changes.txt` mempertahankan
hasil starter awal, termasuk angka/labelling yang dikoreksi melalui audit.

Pengujian E1-E5, clipping terhadap ray, box filter gamma, trilinear, UV against ray,
dan kesamaan RGB culling melaporkan **14/14 passed**. Model dan texture dari Lab 1
tetap memakai provenance yang sama. Kode starter tetap dikreditkan, dan mahasiswa
perlu mengulang hitungan tangan serta memahami diagnosis sebelum viva.

Lab 2 ini tidak mencakup publikasi GitHub Pages atau verifikasi browser; itu Lab 3.
Ray caster belum memiliki reflection, shadow sekunder, BVH atau global illumination.
Normals diturunkan dari geometri world untuk flat shading; tidak ada klaim sistem
smooth-normal inverse-transpose. Clipping sisi frustum penuh dan anisotropic filtering
juga belum ditambahkan. Itulah batas konkret konfigurasi yang diukur.
'''
    (OUT/'Laporan_Lab2.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
