"""Register the painted part of the image edit to the car's rear deck."""
import json

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from trunk_surface import CAMERA, HEIGHT, HERE, WIDTH, load, project, render, sample


# The edit is in the user's screenshot composition. Trace JUST its interior
# paint, several pixels away from glass, text, thin shutlines and the lower
# lip, so even a small local displacement cannot transfer those elements.
# Coordinates refer to the 686x366 screenshot pixel centres.
PAINT = [
    (27, 138), (43, 99), (63, 58), (82, 31), (106, 38), (152, 48),
    (205, 58), (252, 67), (304, 76), (351, 83), (405, 92), (454, 99),
    (506, 105), (554, 110), (598, 110), (592, 132), (578, 172),
    (562, 214), (550, 245), (513, 245), (466, 240), (409, 233),
    (361, 226), (305, 217), (250, 207), (201, 195), (153, 184),
    (106, 171), (61, 155), (35, 145),
]

# This is the paint immediately outboard of the right trunk shut line and
# above the tail lamp, including its continuous soft edge highlight.
SHOULDER = [
    (643, 8), (675, 8), (681, 115), (678, 182), (663, 204),
    (645, 218), (635, 230), (620, 249), (607, 261), (588, 269),
    (560, 275), (558, 255), (571, 216), (584, 179), (595, 148),
    (605, 117), (615, 99), (624, 77), (632, 41),
]

def paint_only(donor, outline, scales):
    polygon = np.rint((np.array(outline) + .5) * scales - .5).astype(np.int32)
    support = np.zeros(donor.shape[:2], np.uint8)
    cv2.fillPoly(support, [polygon], 255)
    # Extend only the narrow paint margin. No peripheral generated content
    # can enter our texture, including the edited image's replacement text.
    nearest = distance_transform_edt(~support.astype(bool), return_distances=False,
                                     return_indices=True)
    return donor[nearest[0], nearest[1]]

def registered_colour(donor, outline, xyz, camera, scales):
    donor = paint_only(donor, outline, scales)
    x, y, _, _ = project(xyz, camera)
    x = np.clip((x+.5)*scales[0]-.5, 0, donor.shape[1]-1)
    y = np.clip((y+.5)*scales[1]-.5, 0, donor.shape[0]-1)
    return sample(donor, x, y)


def main():
    folder = HERE / "source"
    with Image.open(folder / "ai_paint_reference.png") as image:
        donor = np.array(image.convert("RGB"))
    match = json.loads((folder / "screenshot_camera.json").read_text())
    orig_size = np.array([match["width"], match["height"]])
    scales = np.array([donor.shape[1], donor.shape[0]]) / orig_size
    _, _, _, vertices, faces, _, _ = load()
    _, face, bary = render(vertices, faces, CAMERA, (HEIGHT, WIDTH))
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    use = (face >= 0) & (yy >= 168) & (yy <= 414) & (xx >= 147) & (xx <= 871)
    xyz = np.einsum('ij,ijk->ik', bary[use], vertices[faces[face[use]]])
    chart = np.array(Image.open(HERE/"trunk_before.png").convert("RGB"))
    chart[use] = registered_colour(donor, PAINT, xyz, match["camera"], scales)
    # Use the visible shoulder's paint as the donor for the matching tiny
    # strip on the far side, which is partly outside the screenshot.
    shoulder_mask = np.array(Image.open(HERE/"trunk_shoulder_mask.png").convert("L"))
    # Include a few pixels around the support for bilinear interpolation.
    active = (cv2.dilate(shoulder_mask, np.ones((5,5), np.uint8)) > 0) & (face >= 0)
    xyz = np.einsum('ij,ijk->ik', bary[active], vertices[faces[face[active]]])
    xyz[:,0] = -np.abs(xyz[:,0])
    chart[active] = registered_colour(donor, SHOULDER, xyz, match["camera"], scales)
    Image.fromarray(chart).save(HERE/"trunk_ai_chart.png")


if __name__ == "__main__":
    main()
