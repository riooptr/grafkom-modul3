"""NumPy-only mipmap generation and texture sampling in linear light."""
from __future__ import annotations

import numpy as np
from srgb import srgb_to_linear


def build_mipmaps(tex_u8, levels=4):
    """Decode sRGB once, then box-average 2x2 blocks in linear light.

    An odd last row/column is edge-padded. Returns float RGB levels, including
    the unfiltered base. Small inputs stop at 1x1 instead of duplicating levels.
    """
    if levels < 1:
        raise ValueError('levels must be positive')
    tex = np.asarray(tex_u8)
    if tex.ndim != 3 or tex.shape[2] != 3 or min(tex.shape[:2]) < 1:
        raise ValueError('expected a nonempty HxWx3 RGB texture')
    result = [srgb_to_linear(tex.astype(np.float64) / 255.0)]
    while len(result) < levels and result[-1].shape[:2] != (1, 1):
        prev = result[-1]
        h, w = prev.shape[:2]
        padded = np.pad(prev, ((0,h%2), (0,w%2), (0,0)), mode='edge')
        ph,pw = padded.shape[:2]
        result.append(padded.reshape(ph//2,2,pw//2,2,3).mean(axis=(1,3)))
    return result


def sample_bilinear(tex_lin, u, v):
    """Legacy module repeat convention shared by forward and ray renderers."""
    h, w = tex_lin.shape[:2]
    u, v = np.broadcast_arrays(np.asarray(u), np.asarray(v))
    u = np.mod(u, 1.0) * (w - 1)
    v = np.mod(v, 1.0) * (h - 1)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (u-x0)[...,None]; fy = (v-y0)[...,None]
    top = tex_lin[y0,x0]*(1-fx) + tex_lin[y0,x1]*fx
    bot = tex_lin[y1,x0]*(1-fx) + tex_lin[y1,x1]*fx
    return top*(1-fy) + bot*fy


def sample_nearest(tex_lin, u, v):
    h,w = tex_lin.shape[:2]
    x = np.rint(np.mod(u,1.0)*(w-1)).astype(int)
    y = np.rint(np.mod(v,1.0)*(h-1)).astype(int)
    return tex_lin[y,x]


def sample_trilinear(levels, u, v, lod):
    u,v,lod = np.broadcast_arrays(u,v,lod)
    lod = np.clip(lod,0,len(levels)-1)
    lo = np.floor(lod).astype(int)
    hi = np.minimum(lo+1,len(levels)-1)
    fraction = lod-lo
    out = np.zeros((*u.shape,3),float)
    for i,tex in enumerate(levels):
        for selector,weight in [(lo,1-fraction),(hi,fraction)]:
            mask = selector == i
            if np.any(mask):
                out[mask] += sample_bilinear(tex,u[mask],v[mask])*weight[mask,None]
    return out
