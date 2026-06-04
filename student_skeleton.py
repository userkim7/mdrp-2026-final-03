#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

try:
    from impl import detect_monitor as _impl_detect_monitor
except Exception:
    _impl_detect_monitor = None


RECTIFIED_WIDTH = 480
RECTIFIED_HEIGHT = 270
MONITOR_DETECT_MAX_WIDTH = 640
MONITOR_REFRESH_INTERVAL = 12

_monitor_frame_index = 0
_cached_monitor = None
_prev_monitor_thumb = None

_prev_line_thumb = None
_last_line = None
_last_line_color = None
_last_line_quality = -1.0
_line_failures = 0


def _order_points(pts):
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return (
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    )


def _valid_points(points) -> bool:
    if any(p is None for p in points):
        return False
    try:
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    except Exception:
        return False
    return bool(np.all(np.isfinite(pts)) and abs(cv2.contourArea(pts)) > 20)


def _normalize_line(line):
    if line is None:
        return None
    try:
        arr = np.asarray(line).reshape(-1)
    except Exception:
        return None
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return None
    return np.asarray(arr[:4], dtype=np.float32)


def _fallback_monitor(image):
    h, w = image.shape[:2]
    return (
        np.array([0, 0], dtype=np.float32),
        np.array([w - 1, 0], dtype=np.float32),
        np.array([w - 1, h - 1], dtype=np.float32),
        np.array([0, h - 1], dtype=np.float32),
    )


def _scene_changed(image, previous, size=(64, 36), threshold=30.0):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    except Exception:
        return False, previous
    if previous is None:
        return True, thumb
    diff = float(np.mean(cv2.absdiff(thumb, previous)))
    return diff > threshold, thumb


def _scale_points(points, scale):
    if scale == 1.0:
        return tuple(np.asarray(p, dtype=np.float32) for p in points)
    return tuple(np.asarray(p, dtype=np.float32) / scale for p in points)


def detect_monitor(image):
    """
    Detect the monitor corners in the input BGR image.

    Return:
      top_left, top_right, bottom_right, bottom_left

    Each point should be an (x, y) pair in the original image coordinate system.
    Return a full-frame fallback only when detection has no usable candidate.
    """
    global _monitor_frame_index, _cached_monitor, _prev_monitor_thumb

    if image is None or image.size == 0:
        return None, None, None, None

    _monitor_frame_index += 1
    _, _prev_monitor_thumb = _scene_changed(
        image, _prev_monitor_thumb, threshold=52.0
    )

    if _cached_monitor is not None and _monitor_frame_index % MONITOR_REFRESH_INTERVAL != 1:
        return _cached_monitor

    detected = None
    if _impl_detect_monitor is not None:
        h, w = image.shape[:2]
        if w > MONITOR_DETECT_MAX_WIDTH:
            scale = MONITOR_DETECT_MAX_WIDTH / float(w)
            small = cv2.resize(
                image,
                (MONITOR_DETECT_MAX_WIDTH, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale = 1.0
            small = image

        try:
            points = _impl_detect_monitor(small)
            if _valid_points(points):
                detected = _scale_points(points, scale)
        except Exception:
            detected = None

    if detected is not None:
        _cached_monitor = tuple(np.asarray(p, dtype=np.float32) for p in detected)
        return _cached_monitor

    if _cached_monitor is not None:
        return _cached_monitor

    _cached_monitor = _fallback_monitor(image)
    return _cached_monitor


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    Perspective-transform the detected monitor into a front-facing 16:9 view.

    If the detected monitor is portrait-like, rotate it into landscape before
    angle measurement. This follows the final rule clarification.
    """
    points = (top_left, top_right, bottom_right, bottom_left)
    if image is None or image.size == 0 or not _valid_points(points):
        return None

    tl, tr, br, bl = _order_points(points)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if width < 5 or height < 5:
        return None

    if height > width * 1.08:
        src = np.array([tr, br, bl, tl], dtype=np.float32)
    else:
        src = np.array([tl, tr, br, bl], dtype=np.float32)

    dst = np.array(
        [
            [0, 0],
            [RECTIFIED_WIDTH - 1, 0],
            [RECTIFIED_WIDTH - 1, RECTIFIED_HEIGHT - 1],
            [0, RECTIFIED_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    try:
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            image,
            transform,
            (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
            flags=cv2.INTER_LINEAR,
        )
    except Exception:
        return None


def _trim_monitor_frame(image):
    if image is None or image.size == 0:
        return image, 0, 0

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 55
    max_x = max(2, int(w * 0.075))
    max_y = max(2, int(h * 0.10))

    left = 0
    while left < max_x:
        col = gray[:, left]
        if float(np.mean(dark[:, left])) < 0.34 and float(np.mean(col)) > 70:
            break
        left += 1

    right = w - 1
    while right > w - 1 - max_x:
        col = gray[:, right]
        if float(np.mean(dark[:, right])) < 0.34 and float(np.mean(col)) > 70:
            break
        right -= 1

    top = 0
    while top < max_y:
        row = gray[top, :]
        if float(np.mean(dark[top, :])) < 0.34 and float(np.mean(row)) > 70:
            break
        top += 1

    bottom = h - 1
    while bottom > h - 1 - max_y:
        row = gray[bottom, :]
        if float(np.mean(dark[bottom, :])) < 0.34 and float(np.mean(row)) > 70:
            break
        bottom -= 1

    if right - left < w * 0.55 or bottom - top < h * 0.55:
        return image, 0, 0

    return image[top : bottom + 1, left : right + 1], left, top


def _line_quality(image, line):
    line = _normalize_line(line)
    if image is None or line is None:
        return -1.0

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in line]
    length = math.hypot(x2 - x1, y2 - y1)
    if length < max(18, min(h, w) * 0.10):
        return -1.0

    samples = int(max(16, min(160, round(length))))
    xs = np.linspace(x1, x2, samples)
    ys = np.linspace(y1, y2, samples)
    xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)

    colors = image[yi, xi].astype(np.float32)
    color_std = float(np.mean(np.std(colors, axis=0)))
    if color_std > 70.0:
        uniform_factor = 0.34
    elif color_std > 48.0:
        uniform_factor = 0.58
    else:
        uniform_factor = 0.88 + max(0.0, min(0.45, (38.0 - color_std) / 75.0))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.count_nonzero(gray[yi, xi] < 90)) / max(samples, 1)
    color_ratio = float(
        np.count_nonzero((sat[yi, xi] > 42) & (val[yi, xi] > 55))
    ) / max(samples, 1)
    support_ratio = max(dark_ratio, color_ratio)
    support_bonus = 0.55 + 1.10 * support_ratio

    dx = x2 - x1
    dy = y2 - y1
    norm = max(1.0, math.hypot(dx, dy))
    nx = -dy / norm
    ny = dx / norm
    off = max(3.0, min(h, w) * 0.018)
    xpa = np.clip(np.round(xs + nx * off).astype(np.int32), 0, w - 1)
    ypa = np.clip(np.round(ys + ny * off).astype(np.int32), 0, h - 1)
    xpb = np.clip(np.round(xs - nx * off).astype(np.int32), 0, w - 1)
    ypb = np.clip(np.round(ys - ny * off).astype(np.int32), 0, h - 1)
    side_colors = (
        image[ypa, xpa].astype(np.float32) + image[ypb, xpb].astype(np.float32)
    ) * 0.5
    contrast = float(np.mean(np.linalg.norm(colors - side_colors, axis=1)))
    contrast_factor = 0.62 + min(0.80, contrast / 70.0)

    border = max(6, int(min(h, w) * 0.050))
    edge_dist = np.minimum.reduce([xi, w - 1 - xi, yi, h - 1 - yi])
    near_edge_ratio = float(np.count_nonzero(edge_dist < border)) / max(samples, 1)
    border_penalty = 1.0
    if near_edge_ratio > 0.55:
        border_penalty = 0.04
    elif near_edge_ratio > 0.30:
        border_penalty = 0.16

    return length * uniform_factor * support_bonus * contrast_factor * border_penalty


def _component_line_from_mask(image, mask):
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0
    min_len = max(20.0, min(h, w) * 0.15)

    for cnt in contours:
        if cv2.contourArea(cnt) < 8:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_long = max(cw, ch)
        box_short = max(1, min(cw, ch))
        if box_long < min_len or box_long / box_short < 3.6:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            continue

        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01
        ).reshape(-1)
        unit = np.array([float(vx), float(vy)], dtype=np.float32)
        normal = np.array([-float(vy), float(vx)], dtype=np.float32)
        origin = np.array([float(x0), float(y0)], dtype=np.float32)
        ts = (pts - origin) @ unit
        ds = np.abs((pts - origin) @ normal)
        p1 = origin + unit * float(ts.min())
        p2 = origin + unit * float(ts.max())
        line = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32)
        line_len = float(ts.max() - ts.min())
        thickness = float(np.percentile(ds, 90) * 2.0 + 1.0)

        score = _line_quality(image, line)
        if line_len < min_len:
            continue
        if thickness > max(18.0, min(h, w) * 0.075):
            score *= 0.35
        elif thickness > max(10.0, min(h, w) * 0.045):
            score *= 0.72
        score *= 1.0 + min(1.2, box_long / box_short / 10.0)
        if score > best_score:
            best_score = score
            best = line

    return None if best is None else np.asarray(best, dtype=np.int32)


def _fast_hough_line(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    median = float(np.median(blur))
    dark_limit = min(95.0, max(45.0, median - 22.0))
    dark = ((blur < dark_limit) | (blur < 55)).astype(np.uint8) * 255
    colored = ((sat > 42) & (val > 55)).astype(np.uint8) * 255
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)
    local_dark = (blackhat > 16).astype(np.uint8) * 255
    local_edges = cv2.Canny(blackhat, 18, 70)

    border = max(6, int(min(h, w) * 0.050))
    for mask in (dark, colored, local_dark, local_edges):
        mask[:border, :] = 0
        mask[h - border :, :] = 0
        mask[:, :border] = 0
        mask[:, w - border :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, kernel, iterations=1)
    local_dark = cv2.morphologyEx(local_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    local_dark = cv2.morphologyEx(local_dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = cv2.bitwise_or(cv2.bitwise_or(dark, colored), local_dark)
    edges = cv2.bitwise_or(cv2.Canny(blur, 35, 110), cv2.Canny(mask, 35, 110))
    edges = cv2.bitwise_or(edges, local_edges)

    best = _component_line_from_mask(image, dark)
    best_score = _line_quality(image, best)

    color_line = _component_line_from_mask(image, colored)
    color_score = _line_quality(image, color_line)
    if color_score > best_score:
        best = color_line
        best_score = color_score

    local_line = _component_line_from_mask(image, local_dark)
    local_score = _line_quality(image, local_line)
    if local_score > best_score:
        best = local_line
        best_score = local_score

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(min(h, w) * 0.045)),
        minLineLength=max(22, int(min(h, w) * 0.18)),
        maxLineGap=max(8, int(min(h, w) * 0.05)),
    )

    if lines is not None:
        for raw in lines[:, 0, :]:
            score = _line_quality(image, raw)
            if score > best_score:
                best = raw
                best_score = score

    return None if best is None else np.asarray(best, dtype=np.int32)


def _line_angle(line) -> Optional[float]:
    line = _normalize_line(line)
    if line is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dy > 0 or (dy == 0 and dx > 0):
        dx = -dx
        dy = -dy
    return float(math.degrees(math.atan2(-dx, -dy)))


def _sample_line_color(image, line):
    line = _normalize_line(line)
    if image is None or image.size == 0 or line is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in line]
    length = math.hypot(x2 - x1, y2 - y1)
    samples = int(max(8, min(80, round(length))))
    xs = np.clip(np.round(np.linspace(x1, x2, samples)).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(np.linspace(y1, y2, samples)).astype(np.int32), 0, h - 1)
    return np.mean(image[ys, xs].astype(np.float32), axis=0)


def _accept_temporal_line(image, line, scene_changed):
    global _last_line, _last_line_color, _last_line_quality, _line_failures

    current = _normalize_line(line)
    if current is None:
        _line_failures += 1
        return None
    current_quality = _line_quality(image, current)

    if scene_changed or _last_line is None or _line_failures >= 10:
        _last_line = current
        _last_line_color = _sample_line_color(image, current)
        _last_line_quality = current_quality
        _line_failures = 0
        return current

    previous = _normalize_line(_last_line)
    if previous is None:
        _last_line = current
        _last_line_color = _sample_line_color(image, current)
        _last_line_quality = current_quality
        _line_failures = 0
        return current

    h, w = image.shape[:2]
    diag = math.hypot(float(w), float(h))
    cur_len = float(np.linalg.norm(current[2:4] - current[:2]))
    prev_len = float(np.linalg.norm(previous[2:4] - previous[:2]))
    if cur_len < 1 or prev_len < 1:
        return previous

    cur_center = (current[:2] + current[2:4]) / 2.0
    prev_center = (previous[:2] + previous[2:4]) / 2.0
    center_shift = float(np.linalg.norm(cur_center - prev_center))
    length_ratio = cur_len / max(prev_len, 1.0)

    cur_angle = _line_angle(current)
    prev_angle = _line_angle(previous)
    if cur_angle is None or prev_angle is None:
        return previous
    angle_diff = abs(cur_angle - prev_angle) % 180.0
    angle_diff = min(angle_diff, 180.0 - angle_diff)

    color_ok = True
    cur_color = _sample_line_color(image, current)
    if cur_color is not None and _last_line_color is not None:
        color_ok = float(np.linalg.norm(cur_color - _last_line_color)) < 105.0

    accept = (
        angle_diff < 14.0
        and 0.55 <= length_ratio <= 1.85
        and center_shift < diag * 0.20
    ) or (
        color_ok
        and angle_diff < 18.0
        and 0.45 <= length_ratio <= 2.20
        and center_shift < diag * 0.26
    )
    accept = accept or (
        current_quality > max(35.0, _last_line_quality * 1.35)
        and angle_diff < 32.0
        and 0.35 <= length_ratio <= 2.80
        and center_shift < diag * 0.36
    )

    if accept:
        _last_line = current
        _last_line_color = cur_color
        _last_line_quality = current_quality
        _line_failures = 0
        return current

    _line_failures += 1
    return previous


def detect_line(rectified):
    """
    Detect the longest line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return a fallback line if detection fails so /student/angle keeps 15Hz.
    """
    global _prev_line_thumb, _last_line, _last_line_color, _last_line_quality

    if rectified is None or rectified.size == 0:
        return None

    changed, _prev_line_thumb = _scene_changed(
        rectified, _prev_line_thumb, threshold=18.0
    )
    if changed:
        _last_line = None
        _last_line_color = None
        _last_line_quality = -1.0

    h, w = rectified.shape[:2]
    margin_x = max(5, int(w * 0.042))
    margin_y = max(5, int(h * 0.052))
    inner = rectified[margin_y : h - margin_y, margin_x : w - margin_x]
    inner, trim_x, trim_y = _trim_monitor_frame(inner)
    margin_x += trim_x
    margin_y += trim_y
    if inner.size == 0:
        inner = rectified
        margin_x = 0
        margin_y = 0

    line = _fast_hough_line(inner)
    if line is None:
        if _last_line is not None:
            return np.asarray(_last_line, dtype=np.int32)
        return np.array([w // 2, h - 2, w // 2, 2], dtype=np.int32)

    x1, y1, x2, y2 = [int(round(float(v))) for v in line]
    shifted = np.array(
        [x1 + margin_x, y1 + margin_y, x2 + margin_x, y2 + margin_y],
        dtype=np.float32,
    )

    accepted = _accept_temporal_line(rectified, shifted, changed)
    if accepted is None:
        return np.array([w // 2, h - 2, w // 2, 2], dtype=np.int32)
    return np.asarray(accepted, dtype=np.int32)


def calculate_angle(line) -> Optional[float]:
    """
    Calculate the line angle in degrees.

    The angle is expressed in the rectified monitor coordinate system.
    0 degrees is upward vertical, right-up is negative, and left-up is positive.
    """
    angle = _line_angle(line)
    if angle is None or not np.isfinite(angle):
        return 0.0

    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return float(angle)


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )
        self._log_frame_index = 0

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            self.get_logger().warning("Monitor not detected.")
            return

        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            self.get_logger().warning("Monitor not rectified.")
            return

        line = detect_line(rectified)
        if line is None:
            self.get_logger().warning("Line not detected.")
            return
        self._debug_line(msg, rectified, line)

        angle = calculate_angle(line)
        if angle is None:
            self.get_logger().warning("Angle not calculated.")
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self._log_frame_index += 1
        if self._log_frame_index % 30 == 0:
            self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        if self.line_pub.get_subscription_count() == 0:
            return

        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
