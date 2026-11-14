"""BVH broad phase and triangle SAT self-intersection checks."""

from dataclasses import dataclass
import numpy as np


@dataclass
class BVHNode:
    lo: np.ndarray
    hi: np.ndarray
    left: int = -1
    right: int = -1
    start: int = -1
    end: int = -1

    @property
    def leaf(self):
        return self.left < 0


class TriangleBVH:
    def __init__(self, vertices, faces, leaf_size=8):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        self.bounds_lo = self.triangles.min(axis=1)
        self.bounds_hi = self.triangles.max(axis=1)
        self.centroids = (self.bounds_lo + self.bounds_hi) / 2
        self.order = np.arange(len(self.faces), dtype=np.int64)
        self.leaf_size = int(leaf_size)
        self.nodes = []
        self.root = self._build(0, len(self.order))

    def _build(self, start, end):
        indices = self.order[start:end]
        lo = self.bounds_lo[indices].min(axis=0)
        hi = self.bounds_hi[indices].max(axis=0)
        node_index = len(self.nodes)
        self.nodes.append(BVHNode(lo, hi))
        if end - start <= self.leaf_size:
            self.nodes[node_index] = BVHNode(lo, hi, start=start, end=end)
            return node_index
        extent = self.centroids[indices].ptp(axis=0)
        axis = int(np.argmax(extent))
        middle = (start + end) // 2
        local_order = np.argsort(self.centroids[indices, axis], kind="mergesort")
        self.order[start:end] = indices[local_order]
        left = self._build(start, middle)
        right = self._build(middle, end)
        self.nodes[node_index] = BVHNode(lo, hi, left=left, right=right)
        return node_index

    def leaf_pairs(self):
        """Yield non-identical leaf pairs whose AABBs overlap."""
        stack = [(self.root, self.root)]
        while stack:
            left_index, right_index = stack.pop()
            left, right = self.nodes[left_index], self.nodes[right_index]
            if np.any(left.lo > right.hi) or np.any(right.lo > left.hi):
                continue
            if left_index == right_index:
                if left.leaf:
                    yield self.order[left.start:left.end], True
                else:
                    stack.append((left.left, left.left))
                    stack.append((left.left, left.right))
                    stack.append((left.right, left.right))
                continue
            if left.leaf and right.leaf:
                yield (self.order[left.start:left.end], self.order[right.start:right.end]), False
                continue
            left_volume = np.prod(np.maximum(left.hi - left.lo, 0))
            right_volume = np.prod(np.maximum(right.hi - right.lo, 0))
            if (not left.leaf) and (right.leaf or left_volume >= right_volume):
                stack.append((left.left, right_index))
                stack.append((left.right, right_index))
            else:
                stack.append((left_index, right.left))
                stack.append((left_index, right.right))


def _candidate_pairs(vertices, faces, leaf_size=8):
    bvh = TriangleBVH(vertices, faces, leaf_size=leaf_size)
    pairs = []
    for payload, same_leaf in bvh.leaf_pairs():
        if same_leaf:
            for i, left in enumerate(payload[:-1]):
                for right in payload[i + 1:]:
                    pairs.append((int(left), int(right)))
        else:
            left, right = payload
            pairs.extend((int(a), int(b)) for a in left for b in right)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64), bvh
    candidates = np.asarray(pairs, dtype=np.int64)
    left_lo, left_hi = bvh.bounds_lo[candidates[:, 0]], bvh.bounds_hi[candidates[:, 0]]
    right_lo, right_hi = bvh.bounds_lo[candidates[:, 1]], bvh.bounds_hi[candidates[:, 1]]
    face_overlap = np.all(left_lo <= right_hi, axis=1) & np.all(right_lo <= left_hi, axis=1)
    candidates = candidates[face_overlap]
    # Adjacent triangles are expected to touch at their shared edge/vertex and
    # are not self-intersections for an indexed surface mesh.
    common = np.zeros(len(candidates), dtype=bool)
    for left_column in range(3):
        for right_column in range(3):
            common |= faces[candidates[:, 0], left_column] == faces[candidates[:, 1], right_column]
    return candidates[~common], bvh


def _orient2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_on_segment(a, b, p, epsilon=1e-12):
    return abs(_orient2d(a, b, p)) <= epsilon and np.all(p >= np.minimum(a, b) - epsilon) and np.all(p <= np.maximum(a, b) + epsilon)


def _segments_intersect_2d(a, b, c, d, epsilon=1e-12):
    ab_c, ab_d = _orient2d(a, b, c), _orient2d(a, b, d)
    cd_a, cd_b = _orient2d(c, d, a), _orient2d(c, d, b)
    if ((ab_c > epsilon and ab_d < -epsilon) or (ab_c < -epsilon and ab_d > epsilon)) and ((cd_a > epsilon and cd_b < -epsilon) or (cd_a < -epsilon and cd_b > epsilon)):
        return True
    return (_point_on_segment(a, b, c, epsilon) or _point_on_segment(a, b, d, epsilon) or
            _point_on_segment(c, d, a, epsilon) or _point_on_segment(c, d, b, epsilon))


def _point_in_triangle_2d(p, triangle, epsilon=1e-12):
    signs = [_orient2d(triangle[i], triangle[(i + 1) % 3], p) for i in range(3)]
    return max(signs) >= -epsilon and min(signs) <= epsilon


def _coplanar_intersection(triangle_a, triangle_b):
    normal = np.cross(triangle_a[1] - triangle_a[0], triangle_a[2] - triangle_a[0])
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in range(3) if axis != drop]
    a, b = triangle_a[:, keep], triangle_b[:, keep]
    for i in range(3):
        for j in range(3):
            if _segments_intersect_2d(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]):
                return True
    return _point_in_triangle_2d(a[0], b) or _point_in_triangle_2d(b[0], a)


def _segment_triangle_hits(segments, triangles, normals, epsilon):
    distances = np.einsum("ni,nki->nk", normals, segments - triangles[:, :1])
    d0, d1 = distances, np.roll(distances, -1, axis=1)
    denominator = d0 - d1
    parameter = np.divide(d0, denominator, out=np.full_like(d0, np.nan), where=np.abs(denominator) > epsilon[:, None])
    points = segments + (np.roll(segments, -1, axis=1) - segments) * parameter[:, :, None]
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    vp = points - triangles[:, None, 0]
    dot00 = np.einsum("ni,ni->n", edge0, edge0)
    dot01 = np.einsum("ni,ni->n", edge0, edge1)
    dot11 = np.einsum("ni,ni->n", edge1, edge1)
    dot20 = np.einsum("nki,ni->nk", vp, edge0)
    dot21 = np.einsum("nki,ni->nk", vp, edge1)
    denominator_bary = dot00 * dot11 - dot01 * dot01
    u = np.divide(dot11[:, None] * dot20 - dot01[:, None] * dot21, denominator_bary[:, None],
                  out=np.full_like(dot20, np.nan), where=np.abs(denominator_bary[:, None]) > 1e-30)
    v = np.divide(dot00[:, None] * dot21 - dot01[:, None] * dot20, denominator_bary[:, None],
                  out=np.full_like(dot21, np.nan), where=np.abs(denominator_bary[:, None]) > 1e-30)
    parameter_valid = (parameter >= -1e-12) & (parameter <= 1 + 1e-12)
    plane_crossing = ~((d0 > epsilon[:, None]) & (d1 > epsilon[:, None]))
    plane_crossing &= ~((d0 < -epsilon[:, None]) & (d1 < -epsilon[:, None]))
    inside = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
    strict = (u > 1e-10) & (v > 1e-10) & (u + v < 1 - 1e-10)
    hit = parameter_valid & plane_crossing & inside
    return np.any(hit, axis=1), np.any(hit & strict, axis=1)


def _proper_non_coplanar_batch(triangle_a, triangle_b, relative_epsilon):
    normal_a = np.cross(triangle_a[:, 1] - triangle_a[:, 0], triangle_a[:, 2] - triangle_a[:, 0])
    normal_b = np.cross(triangle_b[:, 1] - triangle_b[:, 0], triangle_b[:, 2] - triangle_b[:, 0])
    scale = np.maximum(np.linalg.norm(normal_a, axis=1), np.linalg.norm(normal_b, axis=1))
    distance_epsilon = relative_epsilon * np.maximum(scale, 1e-30)
    _, strict_a = _segment_triangle_hits(triangle_a, triangle_b, normal_b, distance_epsilon)
    _, strict_b = _segment_triangle_hits(triangle_b, triangle_a, normal_a, distance_epsilon)
    return strict_a & strict_b


def triangle_sat_pairs(vertices, faces, pairs, contact_epsilon=0.0):
    """Classify candidate pairs with a vectorized 3D triangle SAT.

    Extra in-plane axes handle coplanar triangles; shared-index pairs should be
    removed before calling this function. The returned overlap is a projection
    depth in native units and is conservative for contact classification.
    """
    if len(pairs) == 0:
        return {"proper_crossing": np.empty((0, 2), dtype=np.int64),
                "coplanar_overlap": np.empty((0, 2), dtype=np.int64),
                "touch_only": np.empty((0, 2), dtype=np.int64),
                "separated": np.empty((0, 2), dtype=np.int64),
                "overlap_depth": np.empty(0, dtype=np.float64)}
    tri_a, tri_b = vertices[faces[pairs[:, 0]]], vertices[faces[pairs[:, 1]]]
    ea = np.stack((tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 1], tri_a[:, 0] - tri_a[:, 2]), axis=1)
    eb = np.stack((tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 1], tri_b[:, 0] - tri_b[:, 2]), axis=1)
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])
    axes = [na, nb]
    axes.extend(np.cross(na[:, None, :], ea, axis=2).transpose(1, 0, 2))
    axes.extend(np.cross(nb[:, None, :], eb, axis=2).transpose(1, 0, 2))
    axes.extend(np.cross(ea[:, :, None, :], eb[:, None, :, :], axis=3).reshape(len(pairs), 9, 3).transpose(1, 0, 2))
    axes = np.stack(axes, axis=1)
    axis_norm = np.linalg.norm(axes, axis=2)
    safe_axes = np.divide(axes, axis_norm[:, :, None], out=np.zeros_like(axes), where=axis_norm[:, :, None] > 1e-30)
    proj_a = np.einsum("mki,mai->mka", safe_axes, tri_a)
    proj_b = np.einsum("mki,mai->mka", safe_axes, tri_b)
    min_a, max_a = proj_a.min(axis=2), proj_a.max(axis=2)
    min_b, max_b = proj_b.min(axis=2), proj_b.max(axis=2)
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    overlap[axis_norm <= 1e-30] = np.inf
    depth = overlap.min(axis=1)
    separated = depth < -contact_epsilon
    # Coplanarity is checked against both triangle planes using a scale-aware
    # distance. A coplanar pair with positive in-plane overlap is not a proper
    # crossing, even if it is geometrically invalid due to overlapping faces.
    scale = np.maximum(np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1))
    coplanar = (np.linalg.norm(np.cross(na, nb), axis=1) <= 1e-10 * np.maximum(scale, 1e-30))
    coplanar &= np.abs(np.einsum("ij,ij->i", nb, tri_a[:, 0] - tri_b[:, 0])) <= 1e-10 * np.maximum(np.linalg.norm(nb, axis=1), 1e-30)
    coplanar_intersects = np.zeros(len(pairs), dtype=bool)
    for index in np.flatnonzero(coplanar & ~separated):
        coplanar_intersects[index] = _coplanar_intersection(tri_a[index], tri_b[index])
    coplanar_overlap = coplanar & coplanar_intersects & ~separated
    exact_proper = np.zeros(len(pairs), dtype=bool)
    exact_touch = np.zeros(len(pairs), dtype=bool)
    noncoplanar_indices = np.flatnonzero(~coplanar & ~separated)
    if len(noncoplanar_indices):
        exact_proper[noncoplanar_indices] = _proper_non_coplanar_batch(
            tri_a[noncoplanar_indices], tri_b[noncoplanar_indices], 1e-9)
        exact_touch[noncoplanar_indices] = ~exact_proper[noncoplanar_indices]
    proper = exact_proper
    # A non-coplanar exact point contact is retained as a proper candidate;
    # classifying it as harmless contact requires an additional feature-level
    # test and would risk hiding a zero-measure crossing.
    touch_only = (coplanar & ~coplanar_intersects & ~separated) | exact_touch
    return {"proper_crossing": pairs[proper], "coplanar_overlap": pairs[coplanar_overlap],
            "touch_only": pairs[touch_only], "separated": pairs[separated],
            "overlap_depth": depth}


def brute_force_intersections(vertices, faces, contact_epsilon=0.0):
    pairs = np.asarray([(i, j) for i in range(len(faces)) for j in range(i + 1, len(faces))], dtype=np.int64)
    if len(pairs):
        common = np.zeros(len(pairs), dtype=bool)
        for i in range(3):
            for j in range(3):
                common |= faces[pairs[:, 0], i] == faces[pairs[:, 1], j]
        pairs = pairs[~common]
    return triangle_sat_pairs(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), pairs, contact_epsilon)


def detect_self_intersections(vertices, faces, contact_epsilon=0.0, leaf_size=8):
    pairs, bvh = _candidate_pairs(vertices, faces, leaf_size=leaf_size)
    result = triangle_sat_pairs(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), pairs, contact_epsilon)
    result["candidate_pairs"] = int(len(pairs))
    result["bvh_nodes"] = int(len(bvh.nodes))
    result["bvh_leaf_size"] = int(leaf_size)
    return result
