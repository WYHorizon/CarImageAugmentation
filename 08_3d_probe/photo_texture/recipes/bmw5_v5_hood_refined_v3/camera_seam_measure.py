"""Screen-space contrast along specified screenshot features in unchanged comparison cameras."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
# Three different defects: a straight photo-switch step, a single-photo reflection cut,
# and bright flecks above it. Units of the first two metrics are u8 sRGB luma codes.
def luminance(path):
    array=np.array(Image.open(path).convert('RGB'),np.float32)
    if array.shape[:2]!=(564,628):
        raise ValueError('Expected 628 x 564 screenshot-camera image')
    return array@np.array([.2126,.7152,.0722],np.float32)

def line_contrast(luma,polyline):
    jumps=[]
    for first,second in zip(polyline,polyline[1:]):
        first,second=np.asarray(first,float),np.asarray(second,float)
        d=second-first
        normal=np.array([-d[1],d[0]])/np.linalg.norm(d)
        n=int(np.linalg.norm(d))
        steps=np.linspace(0,1,n)[3:-3,None]
        center=first+steps*d
        pair=[]
        for side in (-1,1):
            means=[]
            for offset in (2.,3.,4.):
                uv=(center+side*normal*offset).astype(np.float32)
                values=cv2.remap(luma,uv[:,0:1],uv[:,1:2],cv2.INTER_LINEAR)
                means.append(values)
            pair.append(np.mean(means,axis=0))
        jumps.extend(np.abs(pair[1]-pair[0]).ravel())
    return {'samples':len(jumps),'mean':round(float(np.mean(jumps)),2),'median':round(float(np.median(jumps)),2)}

def evaluate(path):
    y=luminance(path)
    # Exclude the windshield frame; this rectangle covers scattered bright fragments on hood paint.
    roi=y[48:106,348:574]
    return dict(left_patch_luma_jump=line_contrast(y,[(265,39),(285,105),(233,118)]),
                front_reflection_luma_jump=line_contrast(y,[(264,320),(310,337),(372,332),(425,324),(516,324)]),
                sharp_bright_fragment_pixels=int(np.sum(roi-cv2.medianBlur(roi,5)>18)))

if __name__=='__main__':
    result={p:evaluate(p) for p in sys.argv[1:]}
    print(json.dumps(result,indent=2))
