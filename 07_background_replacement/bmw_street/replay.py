"""Replay the released shopping-street GLB shot using its recorded local Qwen edit."""
import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENE = HERE.parent / "assets/backgrounds/04_neighborhood_shopping_street.png"
MODEL = ROOT / "08_3d_probe/bmw5/bmw5.glb"


def positions(glb):
    """Read POSITION from the first primitive of the bundled GLB."""
    raw = glb.read_bytes()
    size, kind = struct.unpack_from("<I4s", raw, 12)
    if raw[:4] != b"glTF" or kind != b"JSON":
        raise ValueError("Invalid glTF binary")
    tree = json.loads(raw[20:20+size])
    data_start = 28+size
    primitive = tree["meshes"][0]["primitives"][0]
    acc = tree["accessors"][primitive["attributes"]["POSITION"]]
    view = tree["bufferViews"][acc["bufferView"]]
    if acc["type"] != "VEC3" or acc["componentType"] != 5126 or view.get("byteStride",12) != 12:
        raise ValueError("Expected packed float32 XYZ positions in released GLB")
    offset = data_start + view.get("byteOffset",0) + acc.get("byteOffset",0)
    return np.frombuffer(raw, dtype="<f4", count=3*acc["count"], offset=offset).reshape(-1,3)


def ground_projection(points, camera, right, away):
    ground = camera["ground_y"]
    h = np.maximum(0, points[:, 1]-ground)
    yaw = np.deg2rad(camera["yaw_deg"])
    c,s = np.cos(yaw), np.sin(yaw)
    x = points[:,0]*c + points[:,2]*s + camera["car_origin_x"] + right*h
    z = -points[:,0]*s + points[:,2]*c - camera["camera_distance"] - away*h
    cx,cy = camera["vanishing_point"]
    return np.c_[cx + camera["lens"]*x/(-z), cy + camera["lens"]*(camera["camera_y"]-ground)/(-z)]


def soft_polygon(points, shape, blur):
    mask = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.asarray(points,dtype=np.float32))
    cv2.fillConvexPoly(mask, np.rint(hull*16).astype(np.int32), 255, cv2.LINE_AA, shift=4)
    return cv2.GaussianBlur(mask.astype(np.float32)/255, (0,0), blur)


def luminance(image):
    return image @ np.array([.2126,.7152,.0722], np.float32)


def smooth(image, radius):
    return cv2.GaussianBlur(image, (0,0), radius)


def align_car(edit, placed, matte):
    finder = cv2.SIFT_create(contrastThreshold=.018)
    mask = np.uint8(matte>.50)*255
    def features(image, region):
        gray = cv2.cvtColor(np.uint8(image),cv2.COLOR_RGB2GRAY)
        return finder.detectAndCompute(gray, region)
    dest, desc = features(placed, mask)
    src, source_desc = features(edit, cv2.dilate(mask,np.ones((11,11),np.uint8)))
    pairs = cv2.BFMatcher().knnMatch(source_desc,desc,k=2)
    good = [a for a,b in pairs if a.distance < .75*b.distance]
    a = np.float32([src[p.queryIdx].pt for p in good])
    b = np.float32([dest[p.trainIdx].pt for p in good])
    transform, inliers = cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5)
    assert inliers.sum() >= 25
    h,w = matte.shape
    aligned = cv2.warpAffine(edit,transform,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return aligned, transform, int(inliers.sum())


def transfer_shadow(street, local, matte, projected):
    h,w = matte.shape
    yy,xx = np.indices((h,w))
    street_y = smooth(luminance(street),2.5)
    edited_y = smooth(luminance(local),2.5)
    left_edge = 510-1.34*(yy-420)
    center_line = 510+1.19*(yy-420)
    roadway = (xx>left_edge+14)&(xx<center_line-15)&(yy>485)&(yy<790)
    border = cv2.dilate(np.uint8((projected>.01)|(matte>.02)),np.ones((75,75),np.uint8))
    ring = roadway & (border==0) & (street_y>60) & (street_y<210)
    # Estimate the generative image's unshadowed road brightness from the
    # surrounding road. Only the inferred attenuation is applied to the
    # original photograph, preserving its paving, markings and colour.
    features = np.stack([np.ones_like(street_y),street_y,(xx-440)/300,(yy-640)/200],axis=2)
    A = features[ring][::5]; t=edited_y[ring][::5]
    use = np.ones(len(t),bool)
    for _ in range(4):
        coeff = np.linalg.lstsq(A[use],t[use],rcond=None)[0]
        error = np.abs(A@coeff-t)
        use = error < np.percentile(error,80)
    expected = np.clip(features@coeff,45,235)
    observed = np.clip(1-edited_y/expected,0,.83)
    # The physically projected footprint anchors the effect. Include the
    # short soft margin measured in the model edit around this footprint.
    distance = cv2.distanceTransform(np.uint8(projected<.08),cv2.DIST_L2,5)
    falloff = np.clip(1-distance/21,0,1)
    falloff *= falloff*(3-2*falloff)
    density = .46*projected + .54*falloff*observed
    density[density<.0025]=0
    shaded = street*(1-density[...,None]*np.array([1.04,1.,.94],np.float32))
    return shaded,density


def polygon(points, shape, feather):
    mask = np.zeros(shape,np.uint8)
    cv2.fillPoly(mask,[np.array(points,np.int32)],255,lineType=cv2.LINE_AA)
    return smooth(mask.astype(np.float32)/255,feather)


def layout(background, render, camera):
    shape = background.shape[:2]
    vertices = positions(MODEL)
    cast = soft_polygon(ground_projection(vertices,camera,.30,.20), shape, 7.2)
    underbody = np.asarray(camera["footprint"])
    underbody = underbody.mean(0) + .89*(underbody - underbody.mean(0))
    core = soft_polygon(underbody, shape, 5)
    contact = np.zeros(shape,np.float32)
    for wheel in camera["wheels"].values():
        contact = np.maximum(contact, soft_polygon(wheel["quad"], shape, 2))
    density = 1-(1-.23*cast)*(1-.40*core)*(1-.53*contact)
    shaded = background*(1-density[...,None] * np.array([1.03,1.,.94]))
    alpha = np.asarray(render.getchannel("A")).astype(np.float32)/255
    # A small ambient lift couples the dark studio-based paint with the open
    # daylit street. The local image model adapts the broader lighting.
    car = np.asarray(render).astype(np.float32)[...,:3] * [1.028,1.035,1.045] + [1.,1.5,2.5]
    image = np.rint(car*alpha[...,None]+shaded*(1-alpha[...,None])).clip(0,255).astype(np.uint8)
    # Use the original PNG quantization on the geometric shadow.
    return image.astype(np.float32), alpha, np.rint(255*density).astype(np.float32)/255


def compose(output, render_path, edit_path, layout_output):
    street = np.array(Image.open(SCENE).convert("RGB"), np.float32)
    render = Image.open(render_path).convert("RGBA")
    height, width = street.shape[:2]
    if render.size == (2*width, 2*height):
        render = render.resize((width,height), Image.Resampling.LANCZOS)
    if render.size != (width,height):
        raise ValueError("Render must match the street or its 2x supersampled dimensions")
    placed, matte, shadow = layout(street, render, json.loads((HERE/"camera.json").read_text()))
    if layout_output:
        Image.fromarray(placed.astype(np.uint8)).save(layout_output)
    raw = np.array(Image.open(edit_path).convert("RGB"), np.float32)
    if raw.shape != street.shape:
        raise ValueError("Qwen edit must match the street dimensions")
    cv2.setRNGSeed(42)
    edited, _, _ = align_car(raw, placed, matte)
    shaded, density = transfer_shadow(street, edited, matte, shadow)
    # The previous shot's landmarks were projected into the new camera
    # using the depth of the photographed surface of the same GLB.
    plate = polygon([(333,627),(416,632),(416,665),(333,658)], matte.shape, 1)
    emblem = np.zeros_like(matte)
    cv2.ellipse(emblem,(374,571),(9,5),0,0,360,1,-1,cv2.LINE_AA)
    identity = np.maximum(plate,smooth(emblem,.9))
    chrome = polygon([(265,550),(289,564),(373,575),(462,570),(543,538),
                      (544,571),(470,606),(451,628),(311,624),(298,600),(262,585)],
                     matte.shape, 3)
    fine_difference = (placed-smooth(placed,1.6)) - (edited-smooth(edited,1.6))
    agreement = np.clip(1-np.abs(luminance(fine_difference))/34,0,1)
    paint = edited + fine_difference*(.49*chrome*agreement)[...,None]
    paint = paint*(1-identity[...,None]) + placed*identity[...,None]

    edge = cv2.distanceTransform(np.uint8(matte>.5),cv2.DIST_L2,5)
    material = np.clip((edge-.2)/2.8,0,1)[...,None]
    foreground = paint*material + np.asarray(render.convert("RGB"), np.float32)*(1-material)
    image = np.uint8(np.clip(np.rint(foreground*matte[...,None]+shaded*(1-matte[...,None])),0,255))
    outside = (matte == 0) & (density == 0)
    if not np.array_equal(image[outside], street[outside]):
        raise ValueError("Street outside car and shadow changed")
    Image.fromarray(image).save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", type=Path, default=HERE/"car_3d.png")
    parser.add_argument("--edit", type=Path, default=HERE/"qwen_raw.png")
    parser.add_argument("--save-layout", type=Path)
    args = parser.parse_args()
    compose(args.output, args.render, args.edit, args.save_layout)
