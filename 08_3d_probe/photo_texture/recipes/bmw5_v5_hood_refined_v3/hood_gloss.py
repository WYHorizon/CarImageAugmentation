"""Photo-based gloss alternative to the clean front_002 bonnet chart."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
import hood_refine as hood

OUT=Path(__file__).resolve().parent
ORIGINAL=hood.PHOTO/'outputs/bmw5_inputs_v2/front_002/source_rgb.png'
base_field=hood.donor_field

def gloss_donor():
    base=base_field()
    t,s=np.mgrid[0:1:384j,-1:1:512j]
    ax=np.abs(s)
    width=np.interp(t,[0,.56,1],[304,328,335])
    x=771+width*s
    upper=np.interp(ax,[0,.72,.92,1],[380,380.5,383,388])
    lower=np.interp(ax,[0,.16,.38,.65,.83,1],[465,465.5,465,477,470,457])
    y=upper+t*(lower-upper)
    def resample(path):
        return hood.sample(hood.rgb(path),x.ravel(),y.ravel()).reshape(*t.shape,3).astype('float32')
    orig=resample(ORIGINAL)
    clean=resample(hood.DONOR)
    residual=orig-clean
    # Studio blinds have clear limits in front_002. Exclude the entire strip,
    # including its projecting sharp lower lip, before interpolating the
    # underlying broad lighting between the unaffected upper and lower parts.
    start=np.interp(ax,[0,.45,.75,.93,1],[394,392,391,389,388])
    stop=np.interp(ax,[0,.42,.60,.80,.91,1],[429,428,429,438,441,446])
    bad=(y>=start)&(y<=stop)
    bad|=((x-771)/38)**2+((y-471)/22)**2<1
    # An isolated tinted display card in the raw reflection is also excluded.
    bad|=(orig[:,:,0]-orig[:,:,2]>9)&(orig[:,:,1]-orig[:,:,2]>6)
    field=np.zeros_like(residual)
    for i in range(s.shape[1]):
        reliable=np.flatnonzero(~bad[:,i])
        if not reliable.size:
            continue
        for c in range(3):
            field[:,i,c]=np.interp(np.arange(s.shape[0]),reliable,residual[reliable,i,c])
    field=cv2.GaussianBlur(field,(0,0),12, sigmaY=15)
    # Keep large real-photo contrast on the front curve, scaled smoothly down
    # towards the windshield frame. Avoid lifting the full bonnet equally.
    gain=.61*(.5+.5*hood.smooth(t,0,.35))
    return (base+field*gain[:,:,None]).astype('float32')

def main():
    with TemporaryDirectory(prefix='bmw5-hood-gloss-') as work:
        temp=Path(work)
        hood.donor_field=gloss_donor
        hood.OUT=temp
        hood.main()
        # Move named alternatives only; leave the default candidate intact.
        for source,target in [('hood_texture.png','hood_gloss_texture.png'),('hood_top_check.jpg','hood_gloss_check.jpg')]:
            (OUT/target).write_bytes((temp/source).read_bytes())
        method=json.loads((temp/'hood_method.json').read_text())
        method.update(raw_donor=str(ORIGINAL),raw_donor_sha256=hood.sha256(ORIGINAL),method='Slat and badge exclusion; interpolate true raw-minus-clean photo contrast across excluded band; blend in canonical chart before XZ transfer.')
        (OUT/'hood_gloss_method.json').write_text(json.dumps(method,indent=2))
if __name__=='__main__':
    main()
