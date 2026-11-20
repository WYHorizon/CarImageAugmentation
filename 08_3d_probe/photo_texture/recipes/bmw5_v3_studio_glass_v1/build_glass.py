"""Recolour the audited window footprints with an abstract diffuse sky reflection."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
PREVIOUS = OUTPUTS / "bmw5_v3_opaque_glass_v1"
V3 = OUTPUTS / "bmw5_v5_hood_refined_v3"
sys.path.insert(0, str(PREVIOUS))
from build_model import embedded_rgb, patch_glb, read_rgb, sha256
import front_side
import rear_side

pilot = front_side.pilot


def light_field(s, t, pane):
    # The upper part reflects diffuse overhead light; below the soft horizon
    # the colour deepens. A broad off-axis lobe wraps across the compound curve.
    # The field contains no image of a cabin, building or light fixture.
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    if pane == 1:
        low, mid, high = [62, 76, 79], [111, 131, 135], [171, 185, 187]
        horizon = .47 + .24 * s - .12 * (2*s-1)**2
        lobe = np.exp(-((s - .30 - .33*t) / .36)**2)
    elif pane == 4:
        low, mid, high = [55, 68, 70], [98, 117, 120], [154, 170, 172]
        horizon = .37 + .22 * s - .10 * (2*s-1)**2
        lobe = np.exp(-((s - .70 + .28*t) / .45)**2)
    else:
        low, mid, high = [46, 60, 62], [91, 111, 114], [152, 170, 171]
        horizon = .43 + .20 * np.cos(np.pi*(s-.08))
        lobe = np.exp(-((s - .22 - .32*t) / .44)**2)
    sky = .5 + .5 * np.tanh((horizon - t) / .20)
    color = np.array(low) + sky[:, None] * (np.array(mid)-low)
    sheen = (.26 + .74*lobe) * (.38 + .62*sky)
    color += sheen[:, None] * (np.array(high)-mid)
    # Gentle edge rolloff keeps highlights off the rubber seal.
    edges = 6*(2*s-1)**6 + 4*np.exp(-t/.045)
    return color - edges[:,None]


def pane_colors(label, xyz):
    if label < 3:
        name = "front" if label == 1 else "side"
        camera, shape = pilot.CHARTS[name], pilot.SHAPES[name]
        paths = [pilot.FRONT] if label == 1 else pilot.SIDE
    elif label == 3:
        camera, shape, paths = rear_side.SIDE_CAMERA, rear_side.SIDE_SHAPE, rear_side.SIDE_PANES
    else:
        camera, shape, paths = rear_side.REAR_CAMERA, rear_side.REAR_SHAPE, [rear_side.REAR_PANE]
    x, y, _, _ = pilot.project(xyz, camera)
    feather = {1:3, 2:2, 3:2.25, 4:2.25}[label]
    alpha = pilot.sample(pilot.outline(shape, paths, feather), x, y)
    if label == 1:
        u, v, s, t = front_side.sticker_coordinates(x, y)
        s = (s+1)/2
        alpha_sticker = np.array(Image.open(pilot.DONORS / "bmw5_front_glass_mask_v2/sticker_and_cover_protection.png").convert("L"))
        alpha_sticker[320:] = 0
        alpha_sticker[:,619:] = 0
        soft = cv2.GaussianBlur(alpha_sticker.astype(np.float32)/255., (3,3), .45)
        alpha *= 1 - pilot.sample(soft, u, v)
        old = front_side.windshield(s, t)
    elif label == 2:
        s = (x-254)/617
        roof = np.interp(x, [254,308,380,455,545,641,731,798,861], [211,177,152,144,141,141,147,159,179])
        sill = np.interp(x, [254,278,466,545,701,798,871], [259,258,248,248,244,239,230])
        t = (y-roof)/(sill-roof)
        divider = np.interp(y, [150,190,242], [761.8,756.6,749.8])
        alpha *= 1-np.clip(5.4-abs(x-divider),0,1)
        old = front_side.side_smoke(s,t)
    elif label == 3:
        s = (755-x)/625
        old = front_side.side_smoke(s, (y-141)/113)
        roof = np.interp(x,[131,190,246,301,383,455,533,633,697,756], [185,165,152,147,143,144,146,156,180,215])
        sill = np.interp(x,[131,190,259,383,472,540,679,756],[233,240,243,247,249,252,258,260])
        t = (y-roof)/(sill-roof)
    else:
        s, t = np.clip((x-244)/374,0,1), np.clip((y-230)/260,0,1)
        glow = np.exp(-((t-.29-.10*s)/.37)**2) * (.73+.27*np.cos(np.pi*(s-.26)))
        luma = 24 + 12*(1-t) + 16*glow + 3*s
        old = np.column_stack((luma,luma+6,luma+10))
    return alpha[:,None] * (light_field(s,t,label)-old)


def main():
    ids = np.array(Image.open(PREVIOUS / "window_ids.png"))
    previous_glb = (PREVIOUS / "textured_mesh.glb").read_bytes()
    previous = embedded_rgb(previous_glb)
    assert np.array_equal(previous, read_rgb(PREVIOUS / "texture.png"))
    v3 = embedded_rgb((V3 / "textured_mesh.glb").read_bytes())
    protected = np.array(Image.open(V3 / "hood_footprint.png"))>0
    protected |= np.any(read_rgb(OUTPUTS / "bmw5_v5_seam_refined_v2/texture.png") != v3, axis=2)
    assert not np.any((ids!=0) & protected)
    with np.load(pilot.BASE / "textured_mesh.npz") as mesh:
        verts, faces, uv = (mesh[k] for k in ("vertices","faces","uv"))
    h,w = ids.shape
    screen = np.column_stack((uv[:,0]*(w-1), (1-uv[:,1])*(h-1), np.ones(len(uv))))
    _, fid, bary = pilot.render(screen, faces, dict(projection="orthographic",focal_px=1,cx=0,cy=0), (h,w))
    result = previous.copy()
    summary = {}
    for label, name in enumerate(["front","left","right","rear"],1):
        region = ids==label
        assert np.all(fid[region]>=0)
        xyz = np.einsum("ij,ijk->ik",bary[region],verts[faces[fid[region]]])
        delta = pane_colors(label, xyz)
        result[region] = np.clip(np.rint(previous[region]+delta),0,255).astype("uint8")
        summary[name] = {key: np.median(image[region],axis=0).tolist() for key,image in [("previous_median_rgb",previous),("new_median_rgb",result)]}
    pad = ids==5
    _, nearest = distance_transform_edt(fid<0,return_indices=True)
    result[pad] = result[nearest[0][pad],nearest[1][pad]]
    assert np.array_equal(previous[ids==0], result[ids==0])
    assert np.array_equal(v3[protected], result[protected])
    Image.fromarray(result).save(HERE/"texture.png")
    packed,checks = patch_glb((V3/"textured_mesh.glb").read_bytes(), (HERE/"texture.png").read_bytes())
    assert checks["geometry_normals_uv_scene_material_metadata_exact"]
    assert np.array_equal(embedded_rgb(packed), result)
    temp = HERE/"textured_mesh.glb.tmp"
    temp.write_bytes(packed)
    temp.replace(HERE/"textured_mesh.glb")
    report = dict(glb_sha256=sha256(HERE/"textured_mesh.glb"), reference_glb_sha256=sha256(PREVIOUS/"textured_mesh.glb"),
                  original_v3_sha256=sha256(V3/"textured_mesh.glb"),
                  texture_sha256=sha256(HERE/"texture.png"), masks_sha256=sha256(PREVIOUS/"window_ids.png"),
                  pane_colors=summary, changed_outside_windows_and_gutters=0,
                  changed_on_v3_hood_and_fender_edits=0, geometry_normals_uv_materials_unchanged=True,
                  method="Baked object-free diffuse overhead light, zero transmission; original photographic stickers.")
    (HERE/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
