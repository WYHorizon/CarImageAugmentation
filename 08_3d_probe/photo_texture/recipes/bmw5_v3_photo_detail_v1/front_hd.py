"""Embed native-photo detail on the visible grille and headlight interiors of the merged BMW."""
import argparse
import copy
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from front_detail import (BASE, CAMS, GRILLE, HERE, LAMPS, OUTPUTS, coverage,
                          donor_coordinates, project, render, sample)
sys.path.insert(0, str(BASE))
from build_model import parse_glb, read_accessor, view_bytes, write_glb

SIZE = (1024, 640)
SCALE = 2
# Source crop in 1536x1024 camera coordinates, destination corner in the atlas.
CROPS = {
    "front_002": ((526, 459, 1004, 620), (16, 16)),
    "front_001": ((565, 437, 797, 537), (16, 384)),
    "front_003": ((731, 433, 964, 539), (528, 384)),
}


def original_photo_atlas():
    atlas = Image.new("RGB", SIZE)
    originals = OUTPUTS.parents[2] / "test_images" / "宝马5系" / "front"
    for view, (bounds, corner) in CROPS.items():
        index = view.split("_")[-1]
        left, top, right, bottom = bounds
        with Image.open(next(originals.glob(f"*_{index}.jpg"))) as original:
            xscale = original.width / 1536
            yscale = original.height / 1024
            high = original.resize(((right-left)*SCALE, (bottom-top)*SCALE),
                                   Image.Resampling.LANCZOS,
                                   box=(left*xscale, top*yscale, right*xscale, bottom*yscale, ))
            atlas.paste(high, corner)
    return atlas


def register_interiors(vertices, faces):
    texcoord = np.zeros((len(vertices), 2), "<f4")
    selected = np.zeros(len(faces), bool)
    assigned = np.zeros(len(vertices), bool)
    counts = {}
    for view, (bounds, corner) in CROPS.items():
        grille = view == "front_002"
        points = GRILLE if grille else LAMPS[view]["outline"]
        cam = CAMS[view]
        x, y, depth, _ = project(vertices, cam)
        dx, dy = donor_coordinates(view, x, y)
        field = coverage(points, inner=1, outer=0)
        inside = sample(field, dx, dy) > .9
        inside &= (vertices[:, 2] > (.84 if grille else .57))
        zbuffer, _, _ = render(vertices, faces, cam, (1024,1536), False)
        dz = depth - sample(np.nan_to_num(zbuffer, posinf=5), x, y)
        inside &= dz < (.050 if grille else .023)
        if not grille:
            inside &= vertices[:, 0] > 0 if view == "front_001" else vertices[:, 0] < 0
        # The chrome surrounds share some front-proxy vertices with the
        # adjacent lamp. Keep these faces on their existing source chart.
        pick = (inside & ~assigned)[faces].all(axis=1)
        centers = faces[pick]
        pick[pick] &= sample(field, dx[centers].mean(axis=1), dy[centers].mean(axis=1)) > .9
        unique = np.unique(faces[pick])
        assert len(unique) and not np.any(assigned[unique])
        texcoord[unique, 0] = (corner[0]+SCALE*(dx[unique]-bounds[0]+.5)) / SIZE[0]
        texcoord[unique, 1] = (corner[1]+SCALE*(dy[unique]-bounds[1]+.5)) / SIZE[1]
        assert np.min(texcoord[unique]) > 0 and np.max(texcoord[unique]) < 1
        assigned[unique] = True
        selected |= pick
        counts[view] = int(pick.sum())
    return selected, texcoord, counts


def apply_hd(data):
    """Return the merged GLB with its supported interior faces sampling a compact original-photo atlas."""
    original, binary = parse_glb(data)
    tree = copy.deepcopy(original)
    blocks = [view_bytes(tree, binary, i) for i in range(len(tree["bufferViews"]))]
    parts = tree["meshes"][0]["primitives"]
    body = parts[0]
    details = copy.deepcopy(body)
    v = read_accessor(tree, binary, body["attributes"]["POSITION"])
    faces = read_accessor(tree, binary, body["indices"]).reshape(-1,3)
    selection, uv, counts = register_interiors(v, faces)
    print("Native-photo triangles", counts)
    retained, high = faces[~selection], faces[selection]
    def buffer(data, target=None):
        spec = {"buffer": 0}
        if target is not None:
            spec["target"] = target
        tree["bufferViews"].append(spec)
        blocks.append(data)
        return len(blocks)-1
    def accessor(array, kind):
        coords = kind == "VEC2"
        array = array.astype("<f4" if coords else "<u4", copy=False)
        tree["accessors"].append({
            "type": kind, "bufferView": buffer(array.tobytes(),34962 if coords else 34963),
            "componentType": 5126 if coords else 5125, "count": len(array) if coords else array.size
        })
        return len(tree["accessors"])-1
    body["indices"] = accessor(retained, "SCALAR")
    details["indices"] = accessor(high, "SCALAR")
    details["attributes"]["TEXCOORD_0"] = accessor(uv, "VEC2")
    source = io.BytesIO()
    original_photo_atlas().save(source, format="PNG")
    image_index = len(tree["images"])
    tree["images"].append({"mimeType":"image/png", "bufferView":buffer(source.getvalue()),
                           "name":"Original photo grille and headlamp interiors"})
    material = copy.deepcopy(tree["materials"][body["material"]])
    color = material["pbrMetallicRoughness"]["baseColorTexture"]
    texture = copy.deepcopy(tree["textures"][color["index"]])
    texture["source"] = image_index
    color["index"] = len(tree["textures"])
    tree["textures"].append(texture)
    material["name"] = "Grille and headlamp details from original photographs"
    details["material"] = len(tree["materials"])
    tree["materials"].append(material)
    parts.append(details)
    output = write_glb(tree, blocks)
    checked, new_binary = parse_glb(output)
    assert checked["meshes"][0]["primitives"][1:-1] == original["meshes"][0]["primitives"][1:]
    for i in range(len(original["bufferViews"])):
        assert view_bytes(checked, new_binary, i) == view_bytes(original, binary, i)
    assert np.array_equal(read_accessor(checked,new_binary,body["indices"]).reshape(-1,3), retained)
    assert np.array_equal(read_accessor(checked,new_binary,details["indices"]).reshape(-1,3), high)
    reconstructed = np.empty_like(faces)
    reconstructed[~selection], reconstructed[selection] = retained, high
    assert np.array_equal(reconstructed, faces)
    return output


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("input", type=Path)
    args.add_argument("output", type=Path)
    options = args.parse_args()
    options.output.write_bytes(apply_hd(options.input.read_bytes()))
