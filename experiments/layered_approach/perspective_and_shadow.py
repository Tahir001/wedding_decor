#!/usr/bin/env python3
"""Perspective warping and shadow generation for layered item compositing."""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def perspective_warp(
    img: Image.Image,
    camera_angle_deg: float = 45.0,
    target_width: Optional[int] = None,
) -> Image.Image:
    if target_width and img.width != target_width:
        aspect = img.height / img.width
        img = img.resize((target_width, int(target_width * aspect)), Image.LANCZOS)
    angle_rad = math.radians(camera_angle_deg)
    v_scale = math.cos(angle_rad)
    w, h = img.size
    new_h = max(1, int(h * v_scale))
    arr = np.array(img)
    src_pts = np.float32([[0, 0], [w, 0], [0, h]])
    y_offset = (h - new_h) / 2
    dst_pts = np.float32([[0, y_offset], [w, y_offset], [0, y_offset + new_h]])
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(warped)


def generate_shadow(
    img: Image.Image,
    offset_x: int = 0,
    offset_y: int = 4,
    blur_radius: int = 12,
    opacity: float = 0.35,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    alpha = np.array(img.split()[3]).astype(np.float32) / 255.0
    h, w = alpha.shape
    shadow_alpha = np.zeros((h, w), dtype=np.float32)
    y_start = max(0, offset_y)
    y_end = min(h, h + offset_y)
    x_start = max(0, offset_x)
    x_end = min(w, w + offset_x)
    src_y_start = max(0, -offset_y)
    src_y_end = src_y_start + (y_end - y_start)
    src_x_start = max(0, -offset_x)
    src_x_end = src_x_start + (x_end - x_start)
    shadow_alpha[y_start:y_end, x_start:x_end] = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
    if blur_radius > 0:
        ksize = blur_radius * 6 + 1
        if ksize % 2 == 0:
            ksize += 1
        shadow_alpha = cv2.GaussianBlur(shadow_alpha, (ksize, ksize), blur_radius)
    shadow_alpha = np.clip(shadow_alpha * opacity, 0.0, 1.0)
    shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    shadow_rgba[:, :, 0] = color[0]
    shadow_rgba[:, :, 1] = color[1]
    shadow_rgba[:, :, 2] = color[2]
    shadow_rgba[:, :, 3] = (shadow_alpha * 255).astype(np.uint8)
    return Image.fromarray(shadow_rgba)


def rotate_item(img: Image.Image, angle_deg: float) -> Image.Image:
    if angle_deg == 0:
        return img
    return img.rotate(-angle_deg, expand=True, resample=Image.BICUBIC,
                      fillcolor=(0, 0, 0, 0))


def prepare_item(
    img: Image.Image,
    canvas_width: int,
    canvas_height: int,
    scale: float,
    camera_angle_deg: float = 45.0,
    rotation_deg: float = 0.0,
    should_rotate: bool = True,
    shadow_config: Optional[dict] = None,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    target_w = max(1, int(canvas_width * scale))
    aspect = img.height / img.width
    target_h = max(1, int(target_w * aspect))
    item = img.resize((target_w, target_h), Image.LANCZOS)
    item = perspective_warp(item, camera_angle_deg)
    if should_rotate and rotation_deg != 0:
        item = rotate_item(item, rotation_deg)
    shadow = None
    if shadow_config and shadow_config.get("enabled", True):
        shadow = generate_shadow(
            item,
            offset_x=shadow_config.get("offset_x", 0),
            offset_y=shadow_config.get("offset_y", 4),
            blur_radius=shadow_config.get("blur_radius", 12),
            opacity=shadow_config.get("opacity", 0.35),
            color=tuple(shadow_config.get("color", [0, 0, 0])),
        )
    return item, shadow