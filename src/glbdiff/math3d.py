from __future__ import annotations

import math

from glbdiff.models import AABB, Vec3

IDENTITY_MAT4: tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def mat4_from_trs(
    t: tuple[float, float, float],
    r_xyzw: tuple[float, float, float, float],
    s: tuple[float, float, float],
) -> tuple[float, ...]:
    tx, ty, tz = t
    x, y, z, w = r_xyzw
    sx, sy, sz = s

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-12:
        inv = 1.0 / norm
        x *= inv
        y *= inv
        z *= inv
        w *= inv
    else:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        (1.0 - 2.0 * (yy + zz)) * sx,
        (2.0 * (xy + wz)) * sx,
        (2.0 * (xz - wy)) * sx,
        0.0,
        (2.0 * (xy - wz)) * sy,
        (1.0 - 2.0 * (xx + zz)) * sy,
        (2.0 * (yz + wx)) * sy,
        0.0,
        (2.0 * (xz + wy)) * sz,
        (2.0 * (yz - wx)) * sz,
        (1.0 - 2.0 * (xx + yy)) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    )


def mat4_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    out = [0.0] * 16
    for col in range(4):
        c4 = col * 4
        for row in range(4):
            out[c4 + row] = (
                a[row] * b[c4]
                + a[4 + row] * b[c4 + 1]
                + a[8 + row] * b[c4 + 2]
                + a[12 + row] * b[c4 + 3]
            )
    return tuple(out)


def mat4_transform_point(m: tuple[float, ...], p: Vec3) -> Vec3:
    px, py, pz = p
    return (
        m[0] * px + m[4] * py + m[8] * pz + m[12],
        m[1] * px + m[5] * py + m[9] * pz + m[13],
        m[2] * px + m[6] * py + m[10] * pz + m[14],
    )


def transform_aabb(m: tuple[float, ...], box: AABB) -> AABB:
    corners = [
        mat4_transform_point(m, (cx, cy, cz))
        for cx in (box.min_pt[0], box.max_pt[0])
        for cy in (box.min_pt[1], box.max_pt[1])
        for cz in (box.min_pt[2], box.max_pt[2])
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return AABB(
        min_pt=(min(xs), min(ys), min(zs)),
        max_pt=(max(xs), max(ys), max(zs)),
    )


def merge_aabbs(boxes: list[AABB]) -> AABB | None:
    if not boxes:
        return None
    min_x = min(b.min_pt[0] for b in boxes)
    min_y = min(b.min_pt[1] for b in boxes)
    min_z = min(b.min_pt[2] for b in boxes)
    max_x = max(b.max_pt[0] for b in boxes)
    max_y = max(b.max_pt[1] for b in boxes)
    max_z = max(b.max_pt[2] for b in boxes)
    return AABB(min_pt=(min_x, min_y, min_z), max_pt=(max_x, max_y, max_z))
