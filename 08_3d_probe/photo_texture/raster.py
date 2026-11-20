"""CPU triangle rasterizer with a z-buffer and perspective-correct barycentrics.

No OpenGL, CUDA or display is required. NumPy and Numba are the only dependencies.
Run this file directly for small synthetic geometry checks.

    depth, face_id, bary = render(
        vertices, faces,
        camera={"R": R, "t": t, "focal_px": (fx, fy), "cx": cx, "cy": cy},
        size=(height, width),
    )

R and t transform world coordinates to camera coordinates: X_camera = R @ X + t.
Omit R and t when vertices are already in camera coordinates. Camera axes are
+X right, +Y down, +Z forward. Pixel centers are integer coordinates (u, v),
including (0, 0); smaller positive camera Z is nearer. Both triangle sides render.

``camera`` is a mapping; its optional keys are:
  projection: "perspective" (default) or "orthographic".
  focal_px: positive scalar or (fx, fy), default max(height, width). For
      orthographic projection this means pixels per camera-coordinate unit.
  cx, cy: principal point, default ((width - 1) / 2, (height - 1) / 2).
  R, t: world-to-camera rotation (3, 3) and translation (3,), defaults identity/0.
  near, far: camera-Z clipping planes, defaults 1e-4 and +infinity.

Depth is camera Z, NOT distance along the viewing ray. Background is depth=inf,
face_id=-1, bary=(0,0,0). Barycentrics refer to the ORIGINAL face vertex order,
including after clipping. Thus ``bary @ vertices[faces[face_id]]`` reconstructs
the visible surface point; interpolate colors/UVs with the same weights.
"""

from collections.abc import Mapping

import numpy as np
from numba import njit


@njit(cache=True)
def _clip_z(points, weights, count, plane, keep_greater, out_points, out_weights):
    """Sutherland-Hodgman clip, carrying original-face barycentric weights."""
    out_count = 0
    previous = count - 1
    for current in range(count):
        prev_z = points[previous, 2]
        curr_z = points[current, 2]
        prev_inside = prev_z >= plane if keep_greater else prev_z <= plane
        curr_inside = curr_z >= plane if keep_greater else curr_z <= plane
        if prev_inside != curr_inside:
            alpha = (plane - prev_z) / (curr_z - prev_z)
            for axis in range(3):
                out_points[out_count, axis] = (
                    points[previous, axis]
                    + alpha * (points[current, axis] - points[previous, axis])
                )
                out_weights[out_count, axis] = (
                    weights[previous, axis]
                    + alpha * (weights[current, axis] - weights[previous, axis])
                )
            # Pin the clipped coordinate to avoid division by an almost-zero Z.
            out_points[out_count, 2] = plane
            out_count += 1
        if curr_inside:
            for axis in range(3):
                out_points[out_count, axis] = points[current, axis]
                out_weights[out_count, axis] = weights[current, axis]
            out_count += 1
        previous = current
    return out_count


@njit(cache=True)
def _rasterize(vertices, faces, height, width, fx, fy, cx, cy,
               perspective, near, far, return_barycentric):
    # Keep depth comparisons in float64 even though the public result is float32.
    depth = np.full((height, width), np.inf, dtype=np.float64)
    face_id = np.full((height, width), -1, dtype=np.int32)
    if return_barycentric:
        bary = np.zeros((height, width, 3), dtype=np.float32)
    else:
        bary = np.empty((0, 0, 3), dtype=np.float32)

    # Two parallel clipping planes can produce at most five polygon vertices.
    points = np.empty((8, 3), dtype=np.float64)
    weights = np.empty((8, 3), dtype=np.float64)
    clipped = np.empty((8, 3), dtype=np.float64)
    clipped_weights = np.empty((8, 3), dtype=np.float64)
    for face in range(faces.shape[0]):
        min_z = np.inf
        max_z = -np.inf
        for corner in range(3):
            for axis in range(3):
                points[corner, axis] = vertices[faces[face, corner], axis]
                weights[corner, axis] = 1.0 if corner == axis else 0.0
            min_z = min(min_z, points[corner, 2])
            max_z = max(max_z, points[corner, 2])
        if max_z < near or min_z > far:
            continue
        count = 3
        if min_z < near:
            count = _clip_z(points, weights, count, near, True, clipped, clipped_weights)
            for corner in range(count):
                for axis in range(3):
                    points[corner, axis] = clipped[corner, axis]
                    weights[corner, axis] = clipped_weights[corner, axis]
        if max_z > far and count >= 3:
            count = _clip_z(points, weights, count, far, False, clipped, clipped_weights)
            for corner in range(count):
                for axis in range(3):
                    points[corner, axis] = clipped[corner, axis]
                    weights[corner, axis] = clipped_weights[corner, axis]
        if count < 3:
            continue

        # Triangulate the clipped convex polygon as a fan, retaining its face ID.
        for fan in range(1, count - 1):
            i0, i1, i2 = 0, fan, fan + 1
            z0, z1, z2 = points[i0, 2], points[i1, 2], points[i2, 2]
            if perspective:
                u0, v0 = fx * points[i0, 0] / z0 + cx, fy * points[i0, 1] / z0 + cy
                u1, v1 = fx * points[i1, 0] / z1 + cx, fy * points[i1, 1] / z1 + cy
                u2, v2 = fx * points[i2, 0] / z2 + cx, fy * points[i2, 1] / z2 + cy
            else:
                u0, v0 = fx * points[i0, 0] + cx, fy * points[i0, 1] + cy
                u1, v1 = fx * points[i1, 0] + cx, fy * points[i1, 1] + cy
                u2, v2 = fx * points[i2, 0] + cx, fy * points[i2, 1] + cy

            denominator = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denominator) < 1e-12:
                continue
            lo_u, hi_u = min(u0, u1, u2), max(u0, u1, u2)
            lo_v, hi_v = min(v0, v1, v2), max(v0, v1, v2)
            if hi_u < 0 or lo_u > width - 1 or hi_v < 0 or lo_v > height - 1:
                continue
            # Clamp before integer conversion: near-clipped projections can be large.
            x_start = int(np.ceil(max(0.0, lo_u)))
            x_stop = int(np.floor(min(float(width - 1), hi_u)))
            y_start = int(np.ceil(max(0.0, lo_v)))
            y_stop = int(np.floor(min(float(height - 1), hi_v)))
            inv_denominator = 1.0 / denominator
            for y in range(y_start, y_stop + 1):
                for x in range(x_start, x_stop + 1):
                    a = ((v1 - v2) * (x - u2) + (u2 - u1) * (y - v2)) * inv_denominator
                    b = ((v2 - v0) * (x - u2) + (u0 - u2) * (y - v2)) * inv_denominator
                    c = 1.0 - a - b
                    if a < -1e-10 or b < -1e-10 or c < -1e-10:
                        continue
                    if perspective:
                        a /= z0
                        b /= z1
                        c /= z2
                        z = 1.0 / (a + b + c)
                        a *= z
                        b *= z
                        c *= z
                    else:
                        z = a * z0 + b * z1 + c * z2
                    if z < depth[y, x]:
                        depth[y, x] = z
                        face_id[y, x] = face
                        if return_barycentric:
                            for axis in range(3):
                                bary[y, x, axis] = (
                                    a * weights[i0, axis]
                                    + b * weights[i1, axis]
                                    + c * weights[i2, axis]
                                )
    return depth, face_id, bary


def render(vertices, faces, camera=None, size=(512, 512), return_barycentric=True):
    """Return (depth[H,W], face_id[H,W], bary[H,W,3] or None).

    ``size`` is (HEIGHT, WIDTH), or an integer for a square image. ``vertices``
    can directly contain camera coordinates when camera has no R/t entries.
    See the module docstring for projection and background conventions.
    """
    if isinstance(size, (int, np.integer)):
        size = (size, size)
    if len(size) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in size):
        raise ValueError("size must be a positive integer or (height, width) integers")
    height, width = int(size[0]), int(size[1])
    if camera is None:
        camera = {}
    if not isinstance(camera, Mapping):
        raise TypeError("camera must be a mapping or None")
    known_keys = {"R", "t", "projection", "focal_px", "cx", "cy", "near", "far"}
    unknown = set(camera) - known_keys
    if unknown:
        raise ValueError(f"unknown camera keys: {sorted(unknown)}")

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (M, 3) array")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces contain out-of-range vertex indices")
    if len(faces) > np.iinfo(np.int32).max:
        raise ValueError("face count exceeds int32 face_id capacity")
    rotation = np.asarray(camera.get("R", np.eye(3)), dtype=np.float64)
    translation = np.asarray(camera.get("t", np.zeros(3)), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("R and t must have shapes (3, 3) and (3,)")
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("R and t must be finite")
    if "R" in camera or "t" in camera:
        vertices = vertices @ rotation.T + translation
    if not np.isfinite(vertices).all():
        raise ValueError("transformed vertices must be finite")

    projection = camera.get("projection", "perspective")
    if projection not in ("perspective", "orthographic"):
        raise ValueError("projection must be 'perspective' or 'orthographic'")
    focal = np.asarray(camera.get("focal_px", max(height, width)), dtype=np.float64)
    if focal.ndim == 0:
        focal = np.repeat(focal, 2)
    if focal.shape != (2,) or not np.isfinite(focal).all() or (focal <= 0).any():
        raise ValueError("focal_px must be a finite positive scalar or (fx, fy)")
    cx = float(camera.get("cx", (width - 1) / 2))
    cy = float(camera.get("cy", (height - 1) / 2))
    near, far = float(camera.get("near", 1e-4)), float(camera.get("far", np.inf))
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError("cx and cy must be finite")
    if not np.isfinite(near) or near <= 0 or np.isnan(far) or far <= near:
        raise ValueError("clipping planes must satisfy 0 < near < far <= infinity")

    depth, face_id, bary = _rasterize(
        np.ascontiguousarray(vertices), np.ascontiguousarray(faces, dtype=np.int64),
        height, width, float(focal[0]), float(focal[1]), cx, cy,
        projection == "perspective", near, far, bool(return_barycentric),
    )
    return depth.astype(np.float32), face_id, bary if return_barycentric else None


def _self_test():
    camera = {"projection": "orthographic", "focal_px": 1, "cx": 0, "cy": 0}
    front = np.array([[2., 2., 2.], [12., 2., 2.], [2., 12., 2.]])
    back = front.copy()
    back[:, 2] = 5
    vertices = np.concatenate((back, front))
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    for order in (faces, faces[::-1]):
        d, f, b = render(vertices, order, camera, size=(16, 16))
        assert d[5, 4] == 2
        np.testing.assert_allclose(b[5, 4], [.5, .2, .3], atol=1e-7)
        np.testing.assert_allclose(b[5, 4] @ vertices[order[f[5, 4]]], [4, 5, 2], atol=1e-6)
        assert np.isinf(d[0, 0]) and f[0, 0] == -1 and not b[0, 0].any()

    camera_p = {"focal_px": 1, "cx": 0, "cy": 0}
    z = np.array([2., 4., 8.])
    v = front.copy()
    v[:, :2] *= z[:, None]
    v[:, 2] = z
    tri = np.array([[0, 1, 2]])
    d, _, b = render(v, tri, camera_p, size=16)
    expected = np.array([.5, .2, .3]) / z
    expected /= expected.sum()
    np.testing.assert_allclose(b[5, 4], expected, atol=1e-7)
    np.testing.assert_allclose(d[5, 4], expected @ z, rtol=1e-6)

    # Near/far clipping must retain the visible part, not drop whole triangles.
    v = front.copy()
    v[:, 2] = [.5, 2, 2]
    clipped_camera = dict(camera, near=1., far=1.5)
    d, f, b = render(v, tri, clipped_camera, size=16)
    assert f[3, 3] == -1 and f[5, 5] == 0 and f[5, 7] == -1
    np.testing.assert_allclose(b[5, 5] @ v, [5, 5, 1.4], atol=1e-6)
    camera_p.update(near=1., far=1.5, focal_px=4.)
    d, f, b = render(v, tri, camera_p, size=64)
    ys, xs = np.nonzero(f >= 0)
    assert len(xs) > 0
    positions = b[ys, xs] @ v
    np.testing.assert_allclose(4 * positions[:, :2] / positions[:, 2, None],
                               np.column_stack((xs, ys)), atol=4e-6)
    assert d[ys, xs].min() >= 1 - 1e-6 and d[ys, xs].max() <= 1.5 + 1e-6

    R = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    t = np.array([1., 2., 3.])
    world = (front - t) @ R
    d, f, b = render(world, tri, dict(camera, R=R, t=t), size=16,
                     return_barycentric=False)
    assert d[5, 4] == 2 and f[5, 4] == 0 and b is None
    # Reversing winding must not cull a visible face.
    _, f, _ = render(front, tri[:, ::-1], camera, size=16)
    assert f[5, 4] == 0
    print("raster self-test passed: occlusion, barycentrics, perspective, clipping, R/t, winding")


if __name__ == "__main__":
    _self_test()
