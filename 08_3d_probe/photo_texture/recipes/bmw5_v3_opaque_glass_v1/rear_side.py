"""Apply rear and right-side smoked-glass fields using v3-specific orthographic outlines."""

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'bmw5_v3_glass_pilot_v1'))
from build_glass import chart, outline, project, render, sample


SIDE_CAMERA = chart(-90, 2, 800, 570, 340)
REAR_CAMERA = chart(180, 42, 800, 430, 130)
SIDE_SHAPE = (420, 1000)
REAR_SHAPE = (850, 860)

# Right-side B pillar, the near quarterlight divider and the exterior mirror
# lie between or outside these three outlines. The *far* quarterlight pillar
# visible through the source glass is naturally removed.
SIDE_PANES = [
    [(133,186),(142,181),(163,173),(190,165),(215,158),(233,154),
     (245,242),(204,240),(167,237),(140,234),(135,227),(131,220),
     (131,213)],
    [(246,152),(268,149),(301,147),(341,145),(383,143),(425,143),
     (455,144),(472,249),(427,249),(383,247),(322,246),(259,243)],
    [(496,144),(533,146),(571,148),(601,151),(633,156),(663,165),
     (697,180),(732,200),(752,212),(756,216),(741,218),(731,223),
     (724,231),(722,252),(721,260),(679,258),(636,257),(589,254),
     (540,252)]
]

# The original rear glass has a horizontal source-photo junction and two
# light rectangular fragments at its upper corners. A single continuous pane
# follows its external outline rather than those internal photograph edges.
REAR_PANE = [
    (253,241),(263,231),(277,230),(303,233),(342,237),(385,240),
    (430,241),(469,240),(513,237),(552,232),(579,230),(594,230),
    (606,238),(610,258),(613,295),(616,336),(618,389),(618,430),
    (616,459),(611,477),(602,483),(574,485),(531,487),(475,490),
    (426,491),(367,491),(314,487),(272,485),(252,482),(244,475),
    (243,449),(245,390),(246,343),(245,318),(247,286),(248,259)
]


def apply(base, vertices, faces, uv):
    from front_side import side_smoke

    height,width = base.shape[:2]
    atlas_vertices = np.column_stack((uv[:,0]*(width-1), (1-uv[:,1])*(height-1), np.ones(len(uv))))
    atlas_camera = dict(projection='orthographic', focal_px=1, cx=0, cy=0)
    _, triangle_ids, barycentric = render(atlas_vertices, faces, atlas_camera, (height,width))
    valid = triangle_ids >= 0
    indices = np.flatnonzero(valid)
    points = np.einsum('ij,ijk->ik', barycentric[valid], vertices[faces[triangle_ids[valid]]])
    del barycentric
    result = base.copy()
    ids = np.zeros((height,width),np.uint8)
    for label, camera, shape, panes in (
        (3,SIDE_CAMERA,SIDE_SHAPE,SIDE_PANES),
        (4,REAR_CAMERA,REAR_SHAPE,[REAR_PANE]),
    ):
        mask = outline(shape, panes, 2.25)
        x, y, depth, _ = project(points,camera)
        selected = np.flatnonzero((x>=0)&(x<shape[1]-1)&(y>=0)&(y<shape[0]-1))
        alpha = sample(mask,x[selected],y[selected])
        selected,alpha = selected[alpha>0],alpha[alpha>0]
        surface,_,_ = render(vertices,faces,camera,shape,False)
        observed = sample(np.nan_to_num(surface,posinf=5.),x[selected],y[selected])
        foreground = np.abs(observed-depth[selected])<.009
        selected,alpha = selected[foreground],alpha[foreground]
        xx,yy = x[selected],y[selected]
        if label == 3:
            s = np.clip((755-xx)/625,0,1)
            t = np.clip((yy-141)/113,0,1)
            colors = side_smoke(s,t)
        else:
            s = np.clip((xx-244)/374,0,1)
            t = np.clip((yy-230)/260,0,1)
            # Rear panel is more nearly horizontal than the side panes and
            # catches more diffuse overhead light. Broad, asymmetric light
            # falloff across the panel gives it curvature.
            upper = np.exp(-((t-.29-.10*s)/.37)**2)
            across = .73+.27*np.cos(np.pi*(s-.26))
            luminance = 24 + 12*(1-t) + 16*upper*across + 3*s
            colors = np.column_stack((luminance,luminance+6,luminance+10))
        dst = indices[selected]
        assert not np.any(ids.flat[dst])
        ids.flat[dst] = label
        pixels = result.reshape(-1,3)
        pixels[dst] = np.rint(pixels[dst] + alpha[:,None]*(colors.astype('float32')-pixels[dst])).astype('uint8')
    return result,ids
