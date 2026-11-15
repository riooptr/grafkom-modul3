"""Lab 2 checks against analytic results or independent primary ray visibility."""
import numpy as np

from objio import Mesh, load_obj
from ppm_io import read_ppm
from transforms import perspective
from texture_sampling import build_mipmaps, sample_trilinear
from srgb import linear_to_srgb
from metrics import psnr
import raster3d
import raytrace


def t_clip_intersections():
    # One vertex outside near=1: both intersections lie one third of the way
    # from that vertex towards either inside vertex (in view/world space).
    pos=np.array([[-.2,.1,-.5],[.8,-.4,-2],[-.6,.7,-2]])
    attrs=np.array([[0.,0.],[1.,0.],[0.,1.]])
    P=perspective(60,1,1,9)
    clip=(P@np.c_[pos,np.ones(3)].T).T
    pieces=raster3d.clip_triangle_near(clip,attrs)
    assert len(pieces)==2
    boundary=[]
    for coords,uv in pieces:
        assert np.all(coords[:,2]+coords[:,3]>=-1e-12)
        boundary.extend(uv[np.abs(coords[:,2]+coords[:,3])<1e-12].tolist())
    assert any(np.allclose(a,[1/3,0]) for a in boundary)
    assert any(np.allclose(a,[0,1/3]) for a in boundary)
    print('     near-clip: 1 outside -> 2 triangles; intersection UV=(1/3,0),(0,1/3)')


def t_near_far_ray_visibility():
    # Includes an oblique front triangle crossing near, a valid triangle, and
    # geometry beyond far. Primary ray t must be converted to axial distance.
    p=np.array([[-.7,-.5,-.5],[1.1,-.6,-2.3],[.1,1,-2.3],
                [-4,-4,-6],[4,-4,-6],[0,4,-6]])
    m=Mesh(p,[],[[0,1,2],[3,4,5]],[])
    kw=dict(w=80,h=80,near=1,far=4,eye=(0,0,0),center=(0,0,-1),fovy=75)
    a,da,_=raster3d.render(m,**kw)
    b,db=raytrace.render(m,**kw)
    assert np.array_equal(np.isfinite(da),np.isfinite(db)), 'Clipping visibility differs'
    assert psnr(a,b)>60,(psnr(a,b))
    print(f'     clip/ray: {np.isfinite(da).sum()} visible pixels; PSNR={psnr(a,b)} dB')


def t_gamma_correct_mipmap():
    checker=np.array([[[0,0,0],[255,255,255]],[[255,255,255],[0,0,0]]],np.uint8)
    levels=build_mipmaps(checker,4)
    assert len(levels)==2 and levels[1].shape==(1,1,3)
    assert np.allclose(levels[1],.5)
    encoded=np.rint(linear_to_srgb(levels[1])*255).astype(np.uint8)
    assert np.all(encoded==188),encoded
    print('     mip box: linear mean=0.5 -> sRGB 188; averaging sRGB bytes would give 128')


def t_trilinear_level_mix():
    levels=[np.zeros((4,4,3)),np.ones((2,2,3))]
    actual=sample_trilinear(levels,np.array([.1,.6]),np.array([.3,.8]),np.array([.25,.75]))
    assert np.allclose(actual,[[.25]*3,[.75]*3])
    assert np.allclose(sample_trilinear(levels,.4,.2,99),1)
    print('     trilinear: LOD 0.25/0.75 -> exact 0.25/0.75 linear RGB; max LOD clamped')


def t_perspective_floor_matches_ray():
    # Exactly wL=1,wR=9, with a non-flat UV field and nonconstant depth.
    p=np.array([[-.6,-.55,-1],[5.4,-4.95,-9],[-.6,.65,-1]])
    uv=np.array([[0,0],[1,0],[0,1]])
    m=Mesh(p,uv,[[0,1,2]],[[0,1,2]])
    y,x=np.mgrid[:64,:64]
    tex=np.repeat((((x//4+y//4)%2)*255).astype(np.uint8)[...,None],3,2)
    kw=dict(w=100,h=100,tex_u8=tex,eye=(0,0,0),center=(0,0,-1),fovy=80,near=.5,far=20,ambient=1)
    correct,_,_=raster3d.render(m,**kw)
    naive,_,_=raster3d.render(m,perspective_correct=False,**kw)
    independent,_=raytrace.render(m,**kw)
    pc,nv=psnr(correct,independent),psnr(naive,independent)
    assert pc>60 and nv<30,(pc,nv)
    print(f'     floor: perspective-correct vs ray={pc:.3f} dB; naive vs ray={nv:.3f} dB')


def t_culling_visible_surface_unchanged():
    m=load_obj('data/mymodel.obj')
    tex=read_ppm('data/lab1_brick.ppm')
    kw=dict(w=96,h=96,tex_u8=tex,eye=(4.8,3.8,6.7),center=(0,1.5,0),near=1,far=50,fovy=40)
    before,after={},{}
    a,_,_=raster3d.render(m,stats=before,**kw)
    b,_,_=raster3d.render(m,cull_back=True,stats=after,**kw)
    assert np.array_equal(a,b),'Culling changed visible surface'
    assert after['covered_fragments']<before['covered_fragments']
    print(f"     culling: candidate fragments {before['covered_fragments']} -> {after['covered_fragments']}; RGB unchanged")


CHECKS=[
    ('Lab2 near clipping interpolates intersection attributes',t_clip_intersections),
    ('Lab2 near/far visibility agrees with independent ray hits',t_near_far_ray_visibility),
    ('Lab2 mipmap averages linear light, not encoded sRGB',t_gamma_correct_mipmap),
    ('Lab2 trilinear blends adjacent mip levels',t_trilinear_level_mix),
    ('Lab2 perspective floor agrees with ray reference, naive UV fails',t_perspective_floor_matches_ray),
    ('Lab2 culling reduces candidate fragments without changing image',t_culling_visible_surface_unchanged),
]


if __name__=='__main__':
    for name,fn in CHECKS:
        fn()
        print('OK  '+name)
