"""Benchmark the same CPU scene three times and retain the actual 512x512 frames."""
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone,timedelta
import json
import platform
import numpy as np
from objio import load_obj
from ppm_io import read_ppm,write_ppm
from metrics import psnr,ssim
import raster3d
import raytrace

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'lab3'


def main():
    OUT.mkdir(exist_ok=True)
    prediction=OUT/'predictions_before_measurement.md'
    if not prediction.exists():raise RuntimeError('Write predictions before running the benchmark')
    mesh=load_obj(ROOT/'data/mymodel.obj');tex=read_ppm(str(ROOT/'data/lab1_brick.ppm'))
    params=dict(tex_u8=tex,eye=(4.8,3.8,6.7),center=(0,1.5,0),fovy=40,near=1,far=50)
    timings={}
    for name,module in [('raster',raster3d),('ray',raytrace)]:
        module.render(mesh,w=64,h=64,**params)
        timings[name]=[]
        for i in range(3):
            start=perf_counter();result=module.render(mesh,w=512,h=512,**params)
            duration=perf_counter()-start;timings[name].append(duration)
            print(f'{name} run {i+1}/3: {duration:.6f} seconds',flush=True)
        write_ppm(str(OUT/f'mymodel_{name}.ppm'),result[0])
    a=read_ppm(str(OUT/'mymodel_raster.ppm'));b=read_ppm(str(OUT/'mymodel_ray.ppm'))
    value=psnr(a,b)
    summary={'recorded_at':datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
             'python':platform.python_version(),'numpy':np.__version__,'resolution':[512,512],
             'triangles':mesh.n_tris,'samples_seconds':timings,
             'median_seconds':{k:float(np.median(v)) for k,v in timings.items()},
             'scene':{k:v for k,v in params.items() if k!='tex_u8'},
             'primary_only':True,'warmup_resolution':[64,64],
             'psnr_db':float(value) if np.isfinite(value) else 'inf',
             'ssim_global':ssim(a,b),'different_pixels':int(np.any(a!=b,axis=2).sum())}
    summary['ray_to_raster_ratio']=summary['median_seconds']['ray']/summary['median_seconds']['raster']
    (OUT/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
