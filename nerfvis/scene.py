"""
NeRF + Drawing
"""
import base64
import io
import os
import os.path as osp
import platform
import re
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from . import utils

_servers = []
_server_thds = []


def _f(name: str, field: str):
    return name + "__" + field


def _format_vec3(vec: np.ndarray):
    return f"[{vec[0]}, {vec[1]}, {vec[2]}]"


def _scipy_rotation_from_auto(rot: np.ndarray):
    if rot.shape[-1] == 3:
        q = utils.Rotation.from_rotvec(rot)
    elif rot.shape[-1] == 4:
        q = utils.Rotation.from_quat(rot)
    elif rot.shape[-1] == 9:
        q = utils.Rotation.from_matrix(rot.reshape(list(rot.shape[:-1]) + [3, 3]))
    else:
        raise NotImplementedError
    return q


def _angle_axis_rotate_vector_np(r: np.ndarray, v: np.ndarray):
    """
    Rotate each vector by corresponding axis-angle.
    The formula is from Ceres-solver.
    :param r: (B, 3) or (1, 3) axis-angle
    :param v: (B, 3) or (1, 3) vectors to rotate
    """
    if len(v.shape) == 1 or v.shape[0] == 1:
        v = np.broadcast_to(v, r.shape)
    theta = np.linalg.norm(r, axis=-1)
    good_mask = theta > 1e-15
    good_mask_v = np.broadcast_to(good_mask, v.shape[:1])
    bad_mask = ~good_mask
    bad_mask_v = np.broadcast_to(bad_mask, v.shape[:1])
    result = np.zeros_like(v)
    v_good, v_bad = v[good_mask_v], v[bad_mask_v]
    if v_good.size:
        theta_good = theta[good_mask][..., None]
        cos_theta, sin_theta = np.cos(theta_good), np.sin(theta_good)
        axis = r[good_mask] / theta_good
        perp = np.cross(np.broadcast_to(axis, v_good.shape), v_good)
        dot = np.sum(axis * v_good, axis=-1, keepdims=True)
        result[good_mask_v] = (
            v_good * cos_theta + perp * sin_theta + axis * (dot * (1 - cos_theta))
        )
    if v_bad.size:
        # From Ceres
        result[bad_mask_v] = v_bad + np.cross(
            np.broadcast_to(r[bad_mask], v_bad.shape), v_bad
        )
    return result


def _quaternion_rotate_vector_np(q: np.ndarray, pt: np.ndarray):
    """
    Rotate a point pt by a quaternion (xyzw, Hamilton) will be normalized
    Derived from cere::UnitQuaternionRotatePoint (rotation.h)
    :param q: (B, 4) xyzw, Hamilton convention quaternion
    :param pt: (B, 3) 3D points
    :return: (B, 3) R(q) pt
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    uv = np.cross(q[..., :-1], pt) * 2
    return pt + q[..., -1:] * uv + np.cross(q[..., :-1], uv)


def _rotate_vector_np(rot: np.ndarray, pt: np.ndarray):
    """
    Rotate a vector, using either an axis-angle, quaternion (xyzw), or flattened rotation matrix
    :param rot: (B, 3), (B, 4), or (B, 9), the rotations
    :param pt: (B, 3), the 3D points
    :return: (B, 3), rotated points
    """
    if rot.shape[-1] == 3:
        return _angle_axis_rotate_vector_np(rot, pt)
    elif rot.shape[-1] == 4:
        return _quaternion_rotate_vector_np(rot, pt)
    elif rot.shape[-1] == 9:
        return np.matmul(rot.reshape(list(rot.shape[:-1]) + [3, 3]), pt[..., None])[
            ..., 0
        ]
    else:
        raise NotImplementedError


def _to_np_array(obj) -> np.ndarray:
    """
    Convert list-like and np array-like and torch tensors to numpy array
    """
    if not isinstance(obj, np.ndarray):
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "detach"):
            obj = obj.detach()
        arr = np.array(obj)
    else:
        arr = obj
    if len(arr.strides) > 0 and min(arr.strides) < 0:
        # Negative stride or non-contiguous
        arr = arr.copy()
    return np.ascontiguousarray(arr)


def _standarize_rotation_to_rotvec(r: np.ndarray,
                                   ref_t: Optional[np.ndarray] = None,
                                   single_item: bool = False) -> np.ndarray:
    """
    Normalize all types of rotations to rotation vectors
    """
    r = _to_np_array(r)
    if ref_t is not None:
        ref_t = _to_np_array(ref_t)
    if r.ndim == 1 or (r.ndim == 2 and (single_item or
                      (ref_t is not None and ref_t.ndim == 1))):
        # Make single rotation into a vector
        # Special handlign for single (3, 3) ....
        r = r[None]
    if r.ndim == 3 and r.shape[1] == 3 and r.shape[2] == 3:
        # Matrix
        r = utils.Rotation.from_matrix(r).as_rotvec()
    elif r.ndim == 2 and r.shape[1] == 4 and ref_t is not None:
        # Quaternion
        r = utils.Rotation.from_quat(r).as_rotvec()
    r = r.astype(np.float32)
    return r


class Scene:
    def __init__(self, title: str = "Scene", default_opencv: bool = False):
        """
        Holds radiance field/volume/mesh/point cloud/lines objects for 3D visualization.
        Add objects using :code:`add_*` as seen below, then use
        :code:`export()`/:code:`display()`/:code:`embed()` to create a
        standalone web viewer you can open in a browser or embed in
        a notebook.

        - Single scene quick import:  `from nerfvis import scene`

        - Multiple scenes:  `from nerfvis import Scene` then `scene = Scene('title')`

        :param title: title to show when saving, default 'Scene'.
                      You can change it later by setting scene.title = '...'.
        """
        self.title = title
        self.fields: Dict["str", Any] = {}

        self.world_up = None
        self.cam_forward = None
        inf = float("inf")
        self.bb_min = np.array([inf, inf, inf])
        self.bb_max = np.array([-inf, -inf, -inf])
        if default_opencv:
            self.set_opencv()
        else:
            self.set_opengl()

    def _add_common(self, name, kwargs):
        assert isinstance(name, str), "Name must be a string"
        if "time" in kwargs:
            self.fields[_f(name, "time")] = _to_np_array(kwargs["time"]).astype(
                np.uint32
            )
            del kwargs["time"]
        if "color" in kwargs:
            self.fields[_f(name, "color")] = _to_np_array(kwargs["color"]).astype(
                np.float32
            )
            del kwargs["color"]
        scale = translation = None
        if "scale" in kwargs:
            self.fields[_f(name, "scale")] = np.float32(kwargs["scale"])
            scale = kwargs["scale"]
            del kwargs["scale"]
        if "rotation" in kwargs:
            self.fields[_f(name, "rotation")] = _standarize_rotation_to_rotvec(
                    kwargs["rotation"],
                    kwargs.get("translation"),
                    single_item=True)[0]
            del kwargs["rotation"]
        if "translation" in kwargs:
            self.fields[_f(name, "translation")] = np.array(
                kwargs["translation"]
            ).astype(np.float32)
            translation = kwargs["translation"]
            del kwargs["translation"]
        if "visible" in kwargs:
            self.fields[_f(name, "visible")] = int(kwargs["visible"])
            del kwargs["visible"]
        if "unlit" in kwargs:
            self.fields[_f(name, "unlit")] = int(kwargs["unlit"])
            del kwargs["unlit"]
        if "vert_color" in kwargs:
            vc = _to_np_array(kwargs["vert_color"])
            if vc.dtype != np.uint8:
                vc = (vc * 255).astype(np.uint8)
            self.fields[_f(name, "vert_color")] = vc
            del kwargs["vert_color"]
        return scale, translation

    def _check_args_used(self, name, kwargs):
        if len(kwargs):
            msg = (f"WARNING: Unused kwargs for {name}: {list(kwargs.keys())}, "
                          "might be a typo?")
            warnings.warn(msg)

    def _update_bb(self, points, scale = None, translation = None):
        # FIXME handle rotation

        if points.ndim == 2:
            min_xyz = np.min(points, axis=0)
            max_xyz = np.max(points, axis=0)
        else:
            min_xyz = max_xyz = points

        if scale is not None:
            scale = np.float32(scale)
            min_xyz = min_xyz * scale
            max_xyz = max_xyz * scale

        if translation is not None:
            transl = _to_np_array(translation).astype(np.float32)
            min_xyz = min_xyz + transl
            max_xyz = max_xyz + transl

        self.bb_min = np.minimum(min_xyz, self.bb_min)
        self.bb_max = np.maximum(max_xyz, self.bb_max)

    def add_cube(self, name: str = "cube", **kwargs):
        """
        Add a cube with side length 1 (verts {-0.5, 0.5}^3).

        :param name: an identifier for this object
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (36, 3) vertex color, optional advanced (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        _st = self._add_common(name, kwargs)
        self.fields[name] = "cube"
        p1 = np.array([-0.5, -0.5, -0.5])
        p2 = np.array([0.5, 0.5, 0.5])
        self._update_bb(p1, *_st)
        self._update_bb(p2, *_st)
        self._check_args_used(name, kwargs)

    def set_world_up(self, world_up: np.ndarray):
        """
        Set world up vector (3D)
        """
        self.world_up = world_up

    def set_opencv(self):
        """
        Use OpenCV world up ([0, -1, 0])
        """
        self.world_up = np.array([0.0, -1.0, 0.0])
        self.default_opencv = True

    def set_opengl(self):
        """
        Use OpenGL world up ([0, 1, 0])
        """
        self.world_up = np.array([0.0, 1.0, 0.0])
        self.default_opencv = False

    def set_title(self, title: str):
        """
        Set title of output page
        """
        self.title = title

    def add_wireframe_cube(self, name: str, **kwargs):
        """
        Add a wireframe cube with side length 1 (verts {-0.5, 0.5}^3).
        Uses add_lines.

        :param name: an identifier for this object
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (36, 3) vertex color, optional advanced (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit. Default True (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        verts, segs = [], []
        for i in range(8):
            x = ((i >> 2) & 1) * 2 - 1
            y = ((i >> 1) & 1) * 2 - 1
            z = (i & 1) * 2 - 1
            verts.append([x * 0.5, y * 0.5, z * 0.5])
            if x < 0:
                segs.append([i, i ^ 4])
            if y < 0:
                segs.append([i, i ^ 2])
            if z < 0:
                segs.append([i, i ^ 1])

        self.add_lines(
            name,
            _to_np_array(verts).astype(dtype=np.float32),
            segs=_to_np_array(segs),
            **kwargs,
        )

    def add_sphere(
        self,
        name: str = "sphere",
        rings: Optional[int] = None,
        sectors: Optional[int] = None,
        **kwargs,
    ):
        """
        Add a UV sphere with radius 1

        :param name: an identifier for this object
        :param rings: int, number of lateral rings in UV sphere generation, default 15
        :param sectors: int, number of sectors in UV sphere generation, default 30
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (rings*sectors, 3) vertex color, optional advanced (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        _st = self._add_common(name, kwargs)
        self.fields[name] = "sphere"
        if rings is not None:
            self.fields[_f(name, "rings")] = int(rings)
        if sectors is not None:
            self.fields[_f(name, "sectors")] = int(sectors)
        p1 = np.array([-1.0, -1.0, -1.0])
        p2 = np.array([1.0, 1.0, 1.0])
        self._update_bb(p1, *_st)
        self._update_bb(p2, *_st)
        self._check_args_used(name, kwargs)

    def add_line(self, name: str, a: np.ndarray, b: np.ndarray, **kwargs):
        """
        Add a single line segment from a to b

        :param name: an identifier for this object
        :param a: (3,), first point
        :param b: (3,), second point
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (2, 3) vertex color, optional (overrides color) (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit. Default true (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        _st = self._add_common(name, kwargs)
        self.fields[name] = "line"
        a = _to_np_array(a)
        b = _to_np_array(b)
        self.fields[_f(name, "a")] = a.astype(np.float32)
        self.fields[_f(name, "b")] = b.astype(np.float32)
        self._update_bb(a, *_st)
        self._update_bb(b, *_st)
        self._check_args_used(name, kwargs)

    def add_lines(
        self,
        name: str,
        points: np.ndarray,
        segs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Add a series of line segments (in browser, lines are always size 1 right now)

        :param name: an identifier for this object
        :param points: (N, 3) float, list of points
        :param segs: (N, 2) int, optionally, indices between points
            for which to draw the segments. If not given, draws 1-2-3-4...
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (N, 3) vertex color, optional (overrides color) (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit. Default true (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        _st = self._add_common(name, kwargs)
        points = _to_np_array(points)
        self.fields[name] = "lines"
        self.fields[_f(name, "points")] = points.astype(np.float32)
        if segs is not None:
            self.fields[_f(name, "segs")] = _to_np_array(segs).astype(np.int32)
        self._update_bb(points, *_st)
        self._check_args_used(name, kwargs)

    def add_points(
        self, name: str, points: np.ndarray, point_size: float = 1.0, **kwargs
    ):
        """
        Add a point cloud (in browser, points are always size 1 right now)

        :param name: an identifier for this object
        :param points: (N, 3) float, list of points
        :param point_size: float, point size
        :param color: (3,) color, default is orange (common param, can be 3 item list)
        :param vert_color: (N, 3) vertex color, optional (overrides color) (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit. Default true (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
        _st = self._add_common(name, kwargs)
        points = _to_np_array(points)
        self.fields[name] = "points"
        self.fields[_f(name, "points")] = points.astype(np.float32)
        if point_size != 1.0:
            self.fields[_f(name, "point_size")] = np.float32(point_size)
        self._update_bb(points, *_st)
        self._check_args_used(name, kwargs)

    def add_mesh(
        self,
        name: str,
        points: np.ndarray,
        faces: Optional[np.ndarray] = None,
        face_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Add a general mesh

        :param name: an identifier for this object
        :param points: (N, 3) float, list of points
        :param faces: (N, face_size) int, list of faces; if not given,
                faces will be :code:`0-1-2, 3-4-5, 6-7-8`, etc (glDrawArrays)
        :param face_size: int, one of 1,2,3. 3 means triangle mesh,
                1 means point cloud, and 2 means lines. By default is determined from faces
                (usually 3).
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (N, 3) vertex color, optional (overrides color) (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit.
                            Use this if you want to render vertex colors directly
                            without lighting. Default false. (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
        _st = self._add_common(name, kwargs)
        points = _to_np_array(points)
        self.fields[name] = "mesh"
        self.fields[_f(name, "points")] = points.astype(np.float32)
        if face_size is not None:
            self.fields[_f(name, "face_size")] = int(face_size)
            assert face_size >= 1 and face_size <= 3
        if faces is not None:
            assert faces.ndim == 2 and (
                face_size is None or faces.shape[1] == face_size
            ), f"faces must be (N, face_size={face_size if face_size is not None else -1})"
            self.fields[_f(name, "faces")] = _to_np_array(faces).astype(np.int32)
        self._update_bb(points, *_st)
        self._check_args_used(name, kwargs)

    def add_image(
        self,
        name: str,
        image: Union[str, np.ndarray],
        r: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        focal_length: float = 1111.11,
        z: float = 0.3,
        image_size: int = 512,
        jpeg_quality: int = 80,
        **kwargs,
    ):
        """
        Add an image (as a textured plane mesh, using JPEG compression)

        :param name: an identifier for this object
        :param path: path to the image
        :param rotation: (3,) or (4,) or (3, 3)
                  C2W rotations for each camera, either as axis-angle,
                  xyzw quaternion, or rotation matrix
                  (similar to "rotation")
        :param translation: (3,)
                  C2W translations for each camera applied after rotation
        :param r: alias for rotation (overrides, for similarity with add_camera_frustum)
        :param t: alias for translation (overrides, for similarity with add_camera_frustum)
        :param z: the depth at which to draw the frustum far points.
                  use negative values for OpenGL coordinates (original NeRF)
                  or positive values for OpenCV coordinates (NSVF).
                  If not given, tries to infer a good value. Else defaults to -0.3
                  NOTE: kind of weirdly (but to be consistent
                  with add_camera_frustum),for OpenGL, z needs to be negative,
                  while for OpenCV it should be positive.
        :param image_size: max size of image for display.
                           This is NOT the size of the
                           input image, but the size to be displayed!
                           NOTE: do not make this too large to save memory
        :param jpeg_quality: JPEG quality for compression (0-100)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
        """
        from PIL import Image  # pip install pillow
        if 'opengl' in kwargs:
            warnings.warn("opengl argument is no longer needed in add_image")

        if isinstance(image, str):
            im = Image.open(str(image))
        else:
            image = _to_np_array(image)
            if image.dtype != np.uint8:
                image = (image * 255).astype(dtype=np.uint8)
            im = Image.fromarray(image)
        if self.default_opencv != (z > 0):
            z = -z
        if z < 0.0:
            # Flip image
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        image_wh = im.size

        if hasattr(Image, "Resampling"):
            # Pillow 10
            BOX = Image.Resampling.BOX
        else:
            BOX = Image.BOX
        if image_size < image_wh[0]:
            focal_vis = image_size / image_wh[0] * focal_length
            vis_h = int(image_size * image_wh[1] / image_wh[0])
            im = im.resize((image_size, vis_h), resample=BOX)
        else:
            focal_vis = focal_length

        self.fields[name] = "image"
        if focal_length is not None:
            self.fields[_f(name, "focal_length")] = focal_vis
        if focal_length is not None:
            self.fields[_f(name, "z")] = z
        # For similarity to add_camera_frustum
        if r is not None:
            kwargs["rotation"] = r
        if t is not None:
            kwargs["translation"] = t

        # RGB
        im = im.convert("RGB")
        # Save as JPG in memory
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=jpeg_quality)
        im_bytes = bio.getvalue()
        # To numpy
        im_np = np.frombuffer(im_bytes, dtype=np.uint8)
        self.fields[_f(name, "data")] = im_np

        _st = self._add_common(name, kwargs)
        self._update_bb(kwargs.get("translation", np.zeros(3)), *_st)
        self._check_args_used(name, kwargs)


    def add_camera_frustum(
        self,
        name: str,
        focal_length: Optional[float] = None,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        z: float = 0.3,
        r: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        connect: bool = False,
        update_view: bool = True,
        **kwargs,
    ):
        """
        Add one or more ideal perspective camera frustums

        :param name: an identifier for this object
        :param focal_length: the focal length of the camera (unnormalized), default 1111
        :param image_width: the width of the image/sensor, default 800
        :param image_height: the height of the image/sensor, default 800
        :param z: the depth at which to draw the frustum far points.
                  use negative values for OpenGL coordinates (original NeRF)
                  or positive values for OpenCV coordinates (NSVF).
                  If not given, tries to infer a good value.
        :param r: (N, 3) or (N, 4) or (N, 3, 3) or None, optional
                  C2W rotations for each camera, either as axis-angle,
                  xyzw quaternion, or rotation matrix; if not given, only one camera
                  is added at identity.
        :param t: (N, 3) or None, optional
                  C2W translations for each camera applied after rotation;
                  if not given, only one camera is added at identity.
        :param connect: bool, if true then draws lines through the camera centers, default false
        :param color: (3,) color, default is orange (common param)
        :param vert_color: (N * 5, 3) vertex color, optional advanced (common param)
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true
                        (depends on GET parameter in web version) (common param)
        :param unlit: bool, whether mesh should be rendered unlit (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        :param update_view: bool, if true then updates the camera position, scene origin etc
                            using these cameras
        """
        _st = self._add_common(name, kwargs)
        self.fields[name] = "camerafrustum"
        if focal_length is not None:
            self.fields[_f(name, "focal_length")] = np.float32(focal_length)
        if image_width is not None:
            self.fields[_f(name, "image_width")] = np.float32(image_width)
        if image_height is not None:
            self.fields[_f(name, "image_height")] = np.float32(image_height)
        if connect:
            self.fields[_f(name, "connect")] = 1

        if r is not None:
            assert t is not None, "r,t should be both set or both unset"
            self.fields[_f(name, "r")] = _standarize_rotation_to_rotvec(r, ref_t=t)
        if t is not None:
            t = _to_np_array(t)
            assert r is not None, "r,t should be both set or both unset"
            assert r is not None
            if t.ndim == 1:
                t = t[None]
            self.fields[_f(name, "t")] = t.astype(np.float32)

        if update_view and r is not None:
            # Infer world up direction from GT cams
            ups = _rotate_vector_np(r, _to_np_array([0, -1.0, 0]))
            world_up = np.mean(ups, axis=0)
            world_up /= np.linalg.norm(world_up)

            # Camera forward vector
            forwards = _rotate_vector_np(r, _to_np_array([0, 0, 1.0]))
            cam_forward = np.mean(forwards, axis=0)
            cam_forward /= np.linalg.norm(cam_forward)

            # Set camera center of rotation (origin) for orbit
            self.origin = np.mean(t, axis=0)

            # Set camera position
            self.world_up = world_up
            self.cam_forward = cam_forward

        self.fields[_f(name, "z")] = np.float32(z)

        if self.default_opencv != (z > 0):
            z = -z

        elif t is not None and len(t) > 1:
            t = _to_np_array(t)
            alld = np.linalg.norm(t - t[0], axis=-1)
            mind = alld[alld > 0.0].min()
            self.fields[_f(name, "z")] = np.float32(mind * 0.6)

        if t is not None:
            self._update_bb(t, *_st)
        self._check_args_used(name, kwargs)

    def add_mesh_from_file(
        self, path: str, name_suffix: str = "", center: bool = False, **kwargs
    ):
        """
        Add a mesh from path using trimesh

        :param path: the path to the mesh file
        :param name_suffix: object name will be basename(path) + name_suffix
        :param center: if true, centers object to mean
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)

        Rest of keyword arguments passed to add_mesh
        """
        import trimesh  # pip install trimesh

        mesh: trimesh.Trimesh = trimesh.load(path, force="mesh")
        if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors):
            kwargs["vert_color"] = (
                mesh.visual.vertex_colors[..., :3].astype(np.float32) / 255.0
            )
        verts = _to_np_array(mesh.vertices)
        if center:
            verts = verts - np.mean(verts, axis=0)
        self.add_mesh(
            osp.basename(path).replace(".", "_") + name_suffix,
            points=verts,
            faces=mesh.faces,
            **kwargs,
        )

    def add_axes(self, name: str = "axes", length: float = 1.0, **kwargs):
        """
        Add RGB-XYZ axes at [0, 0, 0] (as hardcoded lines)

        :param name: identifier, default "axes"
        :param length: float, length of axes
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)

        Rest of keyword arguments passed to add_lines
        """
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [length, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, length, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, length],
            ],
            dtype=np.float32,
        )
        self.add_lines(
            name,
            points=points,
            segs=np.array(
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                ],
                dtype=np.int32,
            ),
            vert_color=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.int32,
            ),
            **kwargs,
        )

    def remove(self, name: str):
        """
        Remove an object with given name.
        Note: if you overwrite an object with the same name
        type and arguments, it will overwrite the
        object in the scene.

        :param name: the name given to add_*
        """
        self.remove_all([name])

    def remove_all(self, names: List[str]):
        """
        Remove object with given names

        :param names: list of names as given to add_*
        """
        to_delete = []
        for name in names:
            prefix = name + "__"
            for key in self.fields:
                if key.startswith(prefix) or key == name:
                    to_delete.append(key)
        for key in set(to_delete):
            del self.fields[key]

    def clear(self):
        """
        Clear all objects from the scene.
        """
        self.fields.clear()

    def add_volume(
        self,
        name: str,
        density: np.ndarray,
        colors: np.ndarray,
        radius: float = 1.0,
        density_threshold: float = 1.0,
        data_format: str = "RGBA",
        **kwargs,
    ):
        """
        Add a 3D volume using the PlenOctree renderer

        :param name: an identifier for this object
        :param density: (Dx, Dy, Dz); dimensions need not be powers
                                      of 2 nor equal
        :param colors: (Dx, Dy, Dz, (3, channel_size));
                                      color data,
                                      last dim is size 3 * channel_size
        :param radius: 1/2 side length of volume
        :param density_threshold: threshold below which
                                  density is ignored
        :param data_format: standard PlenOctree data format string,
                            one of :code:`RGBA | SH1 | SH4 | SH9 | SH16`.
                            The channel_size should be respectively
                            :code:`1 | 1 | 4 | 9 | 16```.
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true (depends on GET parameter in web version) (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        density = _to_np_array(density)
        colors = _to_np_array(colors)
        tree_data = utils.vol2plenoctree(
            density,
            colors,
            radius,
            density_threshold=density_threshold,
            data_format=data_format,
        )
        _st = self._add_common(name, kwargs)
        self.fields[name] = "volume"
        for k in tree_data:
            self.fields[_f(name, k)] = tree_data[k]
        self._update_bb(np.array([-radius, -radius, -radius]), *_st)
        self._update_bb(np.array([radius, radius, radius]), *_st)
        self._check_args_used(name, kwargs)

    def add_volume_from_npz(self, name: str, file: str, **kwargs):
        """
        Add a volume already saved as npz
        (for example, PlenOctree checkpoints downloaded from the website)

        :param name: an identifier for this object
        :param file: path to npz file
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param scale:  float, scale, default 1.0 (common param)
        :param visible: bool, whether mesh should be visible on init, default true (depends on GET parameter in web version) (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        tree_data = dict(np.load(file))
        _st = self._add_common(name, kwargs)
        self.fields[name] = "volume"
        for k in tree_data:
            self.fields[_f(name, k)] = tree_data[k]
        radius = 0.5 / tree_data.get("invradius3", tree_data.get("invradius", None))
        if isinstance(radius, float):
            radius = np.array([radius] * 3, dtype=np.float32)
        self._update_bb(-radius, *_st)
        self._update_bb(radius, *_st)
        self._check_args_used(name, kwargs)

    def set_nerf(self, *args, **kwargs):
        """
        Deprecated, please use add_nerf
        """
        warnings.warn("set_nerf is deprecated, please use add_nerf")
        self.add_nerf("nerf", *args, **kwargs)

    def add_nerf(
        self,
        name: str,
        eval_fn: Callable[..., Tuple[Any, Any]],
        center: Union[
            Tuple[float, float, float], List[float], float, np.ndarray, None
        ] = None,
        radius: Union[
            Tuple[float, float, float], List[float], float, np.ndarray, None
        ] = None,
        scale: float = 1.0,
        reso: int = 256,
        use_dirs: bool = False,
        sh_deg: int = 1,
        sh_proj_sample_count: int = 15,
        sh_proj_use_sparse: bool = True,
        sigma_thresh: float = 3.0,
        weight_thresh: float = 0.001,
        r: Optional[Any] = None,
        t: Optional[Any] = None,
        focal_length: Optional[Union[float, Tuple[float, float]]] = None,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        sigma_multiplier: float = 1.0,
        chunk: int = 720720,
        device: str = "cuda:0",
        **kwargs,
    ):
        """
        Discretize and display a NeRF (low quality, for visualization purposes only).
        Currently only supports PyTorch NeRFs.
        Requires tqdm, torch, svox, scipy

        :param eval_fn:
                        - If :code:`use_dirs=False`: NeRF function taking a batch of points :code:`(B, 3)` and returning :code:`(rgb (B, 3), sigma (B, 1))` after activation applied.

                        - If :code:`use_dirs=True` then this function should take points :code:`(B, 1, 3)` and kwarg 'dirs' :code:`(1, sh_proj_sample_count, 3)`; it should return :code:`(rgb (B, sh_proj_sample_count, 3), sigma (B, sh_proj_sample_count, 1))` sigma activation
                            should be applied but rgb must NOT have activation applied for SH projection to work correctly.

        :param center: float or (3,), xyz center of volume to discretize
                       (will try to infer from cameras from add_camera_frustum if not given)
        :param radius: float or (3,), xyz half edge length of volume to discretize
                       (will try to infer from cameras from add_camera_frustum if not given)
        :param scale: float, multiples radius by this before using it (this is provided
                      for convenience, for manually scaling the scene boundaries which
                      is often needed to get the right bounds if using automatic
                      radius from camera frustums i.e. not specifying center and radius)
        :param reso: int, resolution of tree in all dimensions (must be power of 2)
        :param use_dirs: bool, if true, assumes normal NeRF with viewdirs; uses SH projection
                         to recover SH at each point with degree sh_deg.
                         Will view directions as kwarg 'dirs' of eval_fn and expect
                         pre-activation RGB output in addition to density.
                         See the description for the eval_fn param above for more info on
                         the function spec.
        :param sh_deg: int, SH degree if use_dirs, must be between 0-4
        :param sh_proj_sample_count: SH projection samples if use_dirs
        :param sh_proj_use_sparse: Use sparse SH projection via least-squares rather than
                                   monte carlo inner product
        :param sigma_thresh: float, simple density threshold (used if r, t not given)
        :param weight_thresh: float, weight threshold as in PlenOctrees (used if r, t given)
        :param r: (N, 3) or (N, 4) or (N, 3, 3) or None, optional
                  C2W rotations for each camera, either as axis-angle,
                  xyzw quaternion, or rotation matrix; if not given, only one camera
                  is added at identity.
        :param t: (N, 3) or None, optional
                  C2W translations for each camera applied after rotation;
                  if not given, only one camera is added at identity.
        :param focal_length: float or Tuple (fx, fy), optional, focal length for weight thresholding
        :param image_width: float, optional, image width for weight thresholding
        :param image_height: float, optional, image height for weight thresholding
        :param translation: (3,), model translation (common param)
        :param rotation: (3,), model rotation in axis-angle (common param)
        :param nerf_scale:  float, scale, default 1.0;
                            note this has a different name due to legacy
                            conflict (common param)
        :param visible: bool, whether mesh should be visible on init, default true (depends on GET parameter in web version) (common param)
        :param time: int, time at which the mesh should be displayed; -1=always display (default)
                    (common param)
        """
        import torch

        if center is None and not np.isinf(self.bb_min).any():
            center = (self.bb_min + self.bb_max) * 0.5
        if radius is None and not np.isinf(self.bb_min).any():
            radius = (self.bb_max - self.bb_min) * 0.5

        if isinstance(center, list) or isinstance(center, tuple):
            center = np.array(center)
        if isinstance(radius, list) or isinstance(radius, tuple):
            radius = np.array(radius)
        radius *= scale
        print("* Discretizing NeRF (requires torch, tqdm, svox, scipy)")

        if r is not None and t is not None:
            c2w = np.eye(4, dtype=np.float32)[None].repeat(r.shape[0], axis=0)
            c2w[:, :3, 3] = t
            if r.ndim == 3 and r.shape[-1] == 3 and r.shape[-2] == 3:
                # No conversion needed
                c2w[:, :3, :3] = r
            else:
                c2w[:, :3, :3] = _scipy_rotation_from_auto(_to_np_array(r)).as_matrix()
            c2w = torch.from_numpy(c2w).to(device=device)
        else:
            c2w = None

        import torch
        from tqdm import tqdm

        from svox import N3Tree
        from svox.helpers import _get_c_extension

        from .utils.sh import project_function, project_function_sparse

        project_fun = (
            project_function_sparse if sh_proj_use_sparse else project_function
        )

        _C = _get_c_extension()
        with torch.no_grad():
            # Hardcoded for now
            sh_dim = (sh_deg + 1) ** 2
            data_format = f"SH{sh_dim}" if use_dirs else "RGBA"
            init_grid_depth = reso.bit_length() - 2
            assert 2 ** (init_grid_depth + 1) == reso, "Grid size must be a power of 2"
            tree = N3Tree(
                N=2,
                init_refine=0,
                init_reserve=500000,
                geom_resize_fact=1.0,
                depth_limit=init_grid_depth,
                radius=radius,
                center=center,
                data_format=data_format,
                device=device,
            )

            offset = tree.offset.cpu()
            scale = tree.invradius.cpu()

            arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
            xx = (arr - offset[0]) / scale[0]
            yy = (arr - offset[1]) / scale[1]
            zz = (arr - offset[2]) / scale[2]
            grid = (
                torch.stack(torch.meshgrid(xx, yy, zz, indexing="ij")).reshape(3, -1).T
            )

            print("  Evaluating NeRF on a grid")
            out_chunks = []
            if use_dirs:
                print("   Note: using SH projection")
                # Adjust chunk size according to sample count to avoid OOM
                chunk = max(chunk // sh_proj_sample_count, 1)
            for i in tqdm(range(0, grid.shape[0], chunk)):
                grid_chunk = grid[i : i + chunk].cuda()
                # TODO: support mip-NeRF
                if use_dirs:

                    def _spherical_func(viewdirs):
                        raw_rgb, sigma = eval_fn(grid_chunk[:, None], dirs=viewdirs)
                        return raw_rgb, sigma

                    rgb, sigma = project_fun(
                        order=sh_deg,
                        spherical_func=_spherical_func,
                        sample_count=sh_proj_sample_count,
                        device=grid_chunk.device,
                    )
                else:
                    rgb, sigma = eval_fn(grid_chunk)
                    if rgb.shape[-1] == 1:
                        rgb = rgb.expand(-1, 3)  # Grayscale
                    elif rgb.shape[-1] != 3 and str(tree.data_format) == "RGBA":
                        tree.expand(f"SH{rgb.shape[-1] // 3}")

                rgb_sigma = torch.cat([rgb, sigma], dim=-1)
                del grid_chunk, rgb, sigma
                out_chunks.append(rgb_sigma.squeeze(-1))
            rgb_sigma = torch.cat(out_chunks, 0)
            del out_chunks

            def _calculate_grid_weights(
                sigmas, c2w, focal_length, image_width, image_height
            ):
                print("  Performing weight thresholding ")
                # Weight thresholding impl
                opts = _C.RenderOptions()
                opts.step_size = 1e-5
                opts.sigma_thresh = 0.0
                opts.ndc_width = -1

                cam = _C.CameraSpec()
                if isinstance(focal_length, float):
                    focal_length = (focal_length, focal_length)
                cam.fx = focal_length[0]
                cam.fy = focal_length[1]
                cam.width = image_width
                cam.height = image_height

                grid_data = sigmas.reshape((reso, reso, reso)).contiguous()
                maximum_weight = torch.zeros_like(grid_data)
                camspace_trans = torch.diag(
                    torch.tensor(
                        [1, -1, -1, 1],
                        dtype=sigmas.dtype,
                        device=sigmas.device,
                    )
                )
                for idx in tqdm(range(c2w.shape[0])):
                    cam.c2w = c2w[idx]
                    cam.c2w = cam.c2w @ camspace_trans
                    grid_weight, _ = _C.grid_weight_render(
                        grid_data,
                        cam,
                        opts,
                        tree.offset,
                        tree.invradius,
                    )
                    maximum_weight = torch.max(maximum_weight, grid_weight)
                return maximum_weight

            if c2w is None:
                # Sigma thresh
                mask = rgb_sigma[..., -1] >= sigma_thresh
            else:
                # Weight thresh
                assert (
                    focal_length is not None
                    and image_height is not None
                    and image_width is not None
                ), (
                    "All of r, t, focal_length, image_width, "
                    "image_height should be provided to set_nerf to use weight thresholding"
                )
                grid_weights = _calculate_grid_weights(
                    rgb_sigma[..., -1:],
                    c2w.float(),
                    focal_length,
                    image_width,
                    image_height,
                )
                mask = grid_weights.reshape(-1) >= weight_thresh
            grid = grid[mask.cpu()]
            rgb_sigma = rgb_sigma[mask]
            del mask
            assert (
                grid.shape[0] > 0
            ), "This NeRF is completely empty! Make sure you set the bounds reasonably"
            print(
                "  Grid shape =",
                grid.shape,
                "min =",
                grid.min(dim=0).values,
                " max =",
                grid.max(dim=0).values,
            )
            grid = grid.cuda()

            torch.cuda.empty_cache()

            print("  Building octree structure")
            for i in range(init_grid_depth):
                tree[grid].refine()
            print("  tree:", tree)

            if sigma_multiplier != 1.0:
                rgb_sigma[..., -1] *= sigma_multiplier
            tree[grid] = rgb_sigma

            # Just a sanity check, if it failed maybe all points got filtered out
            assert tree.max_depth == init_grid_depth
            print(" Finishing up")

            tree.shrink_to_fit()
            tree_data = {
                "data_dim": tree.data_dim,
                "data_format": repr(tree.data_format),
                "child": tree.child.cpu(),
                "invradius3": tree.invradius.cpu(),
                "offset": tree.offset.cpu(),
                "data": tree.data.data.half().cpu().numpy(),
            }
            if "nerf_scale" in kwargs:
                kwargs["scale"] = kwargs["nerf_scale"]
                del kwargs["nerf_scale"]
            _st = self._add_common(name, kwargs)
            self._update_bb(center - radius, *_st)
            self._update_bb(center + radius, *_st)
            self.fields[name] = "volume"
            for k in tree_data:
                self.fields[_f(name, k)] = tree_data[k]
            self._check_args_used(name, kwargs)

    def write(self, path: str, compress: bool = True):
        """
        Write to drawlist npz which you can open with volrend
        (:code:`volrend --draw <output.npz>`;
        nerfvis_base branch recommended for more up-to-date experience)
        as well as in the web viewer. Usually, it's easier to use one of
        :code:`Scene.export()`,
        :code:`Scene.display()`, or
        :code:`Scene.emebd()`

        :param path: output npz path
        """
        if not path.endswith(".draw.npz"):
            warnings.warn(
                "The filename does not end in .draw.npz, "
                "this will not work in web viewer"
            )
        if self.fields:
            fields = self.fields
        else:
            # Make sure it is nonempty
            fields = {"0": 0}
        if compress:
            np.savez_compressed(path, **fields)
        else:
            np.savez(path, **fields)

    def export(
        self,
        dirname: Optional[str] = None,
        display: bool = False,
        world_up: Optional[np.ndarray] = None,
        cam_center: Optional[np.ndarray] = None,
        cam_forward: Optional[np.ndarray] = None,
        cam_origin: Optional[np.ndarray] = None,
        compress: bool = True,
        instructions: List[str] = [],
        css: str = "",
        url: str = "localhost",
        port: int = 8888,
        open_browser: bool = False,
        output_html_name: str = "index.html",
        embed_output: bool = False,
        serve_nonblocking: bool = False,
    ) -> Tuple[str, str]:
        """
        Write to a standalone web viewer

        :param dirname: output folder path, if not given then uses :code:`./nerfvis_scenes/(0-9a-zA-Z_ from self.title)`
        :param display: if true, serves the output using http.server (default false)
        :param world_up: (3,), optionally, world up unit vector for mouse orbiting
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_center: (3,), optionally, camera center point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_forward: (3,), optionally, camera forward-pointing vector
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_origin: (3,), optionally, camera center of rotation point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param compress: whether to compress the output npz file (slower but smaller)
        :param instructions: list of additional javascript instructions to execute (advanced)
        :param css: additional CSS to inject
        :param url: str, URL for server (if display=True) default localhost
        :param port: int, port for server (if display=True) default 8888
                        (if not available, tries next up to 32)
        :param open_browser: bool, if true then opens the web browser, if possible (default False)
        :param embed_output: bool, if true, embeds the output in the html as self-loading
                                    base64/blob url instead of saving as NPZ.
                                    This may cause the page to load very slowly initially,
                                    but combines all data into a single html file.
        :param serve_nonblocking: bool, if true, and display=True, open a server in another
                                  threading and continues execution

        :return: dirname, if it was not provided, returns the generated folder name;
                 url
        """
        if dirname is None:
            dirname = osp.join(
                "nerfvis_scenes", re.sub("[^0-9a-zA-Z_]", "", self.title)
            )
        os.makedirs(dirname, exist_ok=True)

        if world_up is not None:
            world_up = _to_np_array(world_up)
        elif self.world_up is not None:
            world_up = _to_np_array(self.world_up)
        bb_available = not np.isinf(self.bb_min).any()
        if cam_origin is None:
            cam_origin = np.zeros(3)

        if cam_forward is None:
            if self.cam_forward is not None:
                cam_forward = self.cam_forward
            elif cam_center is not None and cam_origin is not None:
                cam_forward = cam_origin - cam_center
                cam_forward /= np.linalg.norm(cam_forward)
            else:
                cam_forward = np.array([0.7071068, 0.0, -0.7071068])

        if cam_center is None:
            if bb_available:
                bbox_middle = (self.bb_max + self.bb_min) * 0.5
                radius = ((self.bb_max - self.bb_min) * 0.7).max()
                cam_center = bbox_middle - cam_forward * radius * 3.0
            elif cam_forward is not None and cam_origin is not None:
                cam_center = cam_origin - cam_forward

        all_instructions = []
        all_instructions.extend(instructions)
        # Use lower PlenOctree render quality
        # (ugly hack, executes javascript in the browser like this)
        all_instructions.extend(
            [
                "let opt = Volrend.get_options()",
                "opt.step_size = 2e-3",
                "opt.stop_thresh = 1e-1",
                "opt.sigma_thresh = 1e-1",
                "Volrend.set_options(opt)",
            ]
        )
        out_npz_fname = f"volrend.draw.npz"
        all_instructions.append(f'Volrend.set_title("{self.title}")')
        if world_up is not None:
            all_instructions.append(
                "Volrend.set_world_up(" + _format_vec3(world_up) + ")"
            )
        if cam_center is not None:
            all_instructions.append(
                "Volrend.set_cam_center(" + _format_vec3(cam_center) + ")"
            )
        if cam_forward is not None:
            cam_forward = _to_np_array(cam_forward)
            all_instructions.append(
                "Volrend.set_cam_back(" + _format_vec3(-cam_forward) + ")"
            )
        if cam_origin is not None:
            all_instructions.append(
                "Volrend.set_cam_origin(" + _format_vec3(cam_origin) + ")"
            )
        index_html_src_path = osp.join(
            osp.dirname(osp.realpath(__file__)), "index.html"
        )
        index_html_path = osp.join(dirname, output_html_name)
        if osp.isfile(index_html_path):
            os.unlink(index_html_path)

        out_npz_path = osp.join(dirname, out_npz_fname)
        self.write(out_npz_path, compress=compress)

        if embed_output:
            # Embed the output as a blob URL
            with open(out_npz_path, "rb") as f:
                npz_bytes = f.read()
                base64_npz = base64.b64encode(npz_bytes).decode("utf-8")
                all_instructions.append(
                    "let blob_url = await fetch(\"data: 'application/octet-stream';base64,"
                    + base64_npz
                    + '").then(res => res.blob()).then(URL.createObjectURL);'
                )
                all_instructions.append("Volrend.load_remote(blob_url)")
            os.unlink(out_npz_path)
        else:
            all_instructions.append(f'Volrend.load_remote("{out_npz_fname}")')

        # Inject JS into HTML at </body>
        with open(index_html_src_path, "r") as f:
            html = f.read()
        bodyspl = html.split("</body>")
        assert len(bodyspl) == 2, "Malformed html"

        JS_INJECT = """
<script>
window.addEventListener("volrend_ready", async function() {
    {{instructions}};
});
</script>
""".replace(
            "{{instructions}}", ";\n".join(all_instructions)
        )

        html = bodyspl[0] + JS_INJECT + "</body>" + bodyspl[1]

        if css:
            stylespl = html.split("</style>")
            assert len(bodyspl) >= 2, "Malformed html"
            html = (
                stylespl[0] + "\n" + css + "\n</style>" + "</style>".join(stylespl[1:])
            )
        with open(index_html_path, "w") as f:
            f.write(html)

        final_url = ""
        if display:
            from http.server import HTTPServer, SimpleHTTPRequestHandler

            class LocalHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    if tuple(map(int, platform.python_version_tuple())) < (3, 7, 0):
                        # Python < 3.7 legacy
                        assert dirname is not None
                        os.chdir(dirname)
                        super().__init__(*args, **kwargs)
                    else:
                        super().__init__(*args, directory=dirname, **kwargs)

            global _server_thds
            global _servers
            for s in _servers:
                s.shutdown()
                s.server_close()
            for thd in _server_thds:
                thd.join()
            _servers = []
            _server_thds = []

            server = None
            for port_i in range(port, port + 32):
                try:
                    server = HTTPServer(("", port_i), LocalHandler)
                    print(f"Serving {url}:{port_i}")
                    port = port_i
                    break
                except OSError:
                    server = None
                    pass
            assert server is not None, f"Could not find open port near {port}"

            final_url = f"http://{url}:{port}"
            if open_browser:
                import webbrowser

                def open_webbrowser():
                    if not webbrowser.open_new(final_url):
                        print(
                            "Could not open web browser",
                            "(note: server still launched, "
                            "please just open given port manually, using port forarding)",
                        )

                t = threading.Thread(target=open_webbrowser)
                t.start()
            if serve_nonblocking:
                t_serve = threading.Thread(target=server.serve_forever)
                t_serve.start()
                _servers.append(server)
                _server_thds.append(t_serve)
            else:
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    print("nervfvis-server interrupted")
        return dirname, final_url

    def display(self, *args, **kwargs) -> Tuple[str, str]:
        """
        Alias of :code:`Scene.export` with :code:`display=True` (serve using http.server)

        :param dirname: output folder path, if not given then uses :code:`./nerfvis_scenes/(0-9a-zA-Z_ from self.title)`
        :param world_up: (3,), optionally, world up unit vector for mouse orbiting
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_center: (3,), optionally, camera center point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_forward: (3,), optionally, camera forward-pointing vector
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_origin: (3,), optionally, camera center of rotation point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param compress: whether to compress the output npz file (slower but smaller)
        :param instructions: list of additional javascript instructions to execute (advanced)
        :param url: str, URL for server (if display=True) default localhost
        :param port: int, port for server (if display=True) default 8888
                        (if not available, tries next up to 32)
        :param open_browser: bool, if true then opens the web browser, if possible (default False)

        :return: dirname, if it was not provided, returns the generated folder name;
                 url
        """
        return self.export(*args, display=True, **kwargs)

    def embed(self, width: int = 1100, height: int = 600, *args, **kwargs):
        """
        Calls :code:`Scene.export` but in addition embeds inside a notebook.

        NOTE 1: if the notebook is opened from a different notebook root folder
        in the future it will not work.

        NOTE 2: still writes files to ./nerfvis_scenes folder, and it may get large
        overtime. You may want to clean it up manually.

        :param dirname: output folder path, if not given then uses :code:`./nerfvis_scenes/(0-9a-zA-Z_ from self.title)`
        :param world_up: (3,), optionally, world up unit vector for mouse orbiting
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_center: (3,), optionally, camera center point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_forward: (3,), optionally, camera forward-pointing vector
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param cam_origin: (3,), optionally, camera center of rotation point
                               (will try to infer from cameras from add_camera_frustum if not given)
        :param compress: whether to compress the output npz file (slower but smaller)
        :param instructions: list of additional javascript instructions to execute (advanced)
        :param url: str, URL for server (if display=True) default localhost
        :param port: int, port for server (if display=True) default 8888
                        (if not available, tries next up to 32)
        :param open_browser: bool, if true then opens the web browser, if possible (default False)

        :return: dirname, if it was not provided, returns the generated folder name;
                 url
        """
        import random
        import string

        from IPython.display import HTML, IFrame, display  # Requires IPython

        JUPYTER_ROOT = os.readlink("/proc/%s/cwd" % os.environ["JPY_PARENT_PID"])
        dirname_rel = osp.join(
            "nerfvis_ipy_scenes", re.sub("[^0-9a-zA-Z_]", "", self.title)
        )
        dirname = osp.join(JUPYTER_ROOT, dirname_rel)
        randstr = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
        )
        embed_name = "ipython_embed_" + randstr + ".html"
        #  css_inject = "#main-wrapper {max-height:101vh}"
        self.export(
            dirname,
            *args,
            display=False,
            open_browser=False,
            #  css=css_inject,
            serve_nonblocking=False,
            embed_output=True,
            output_html_name=embed_name,
            **kwargs,
        )
        html_file = os.path.join(dirname_rel, embed_name)
        display(IFrame(html_file, width, height))
