#!/usr/bin/env python3

from __future__ import annotations

import math
from collections import deque
from threading import Lock
from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

try:
    from impl import calculate_angle as _impl_calculate_angle
    from impl import detect_line as _impl_detect_line
    from impl import detect_monitor as _impl_detect_monitor
except Exception:
    _impl_calculate_angle = None
    _impl_detect_line = None
    _impl_detect_monitor = None


RECTIFIED_WIDTH = 480
RECTIFIED_HEIGHT = 270
MONITOR_DETECT_MAX_WIDTH = 840


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


def _scale_points(points, scale):
    if scale == 1.0:
        return points
    return tuple(np.asarray(p, dtype=np.float32) / scale for p in points)


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


def detect_monitor(image):
    """
    Detect the monitor corners in the input BGR image.

    Return:
      top_left, top_right, bottom_right, bottom_left

    Each point is an (x, y) pair in the original image coordinate system.
    Return (None, None, None, None) if detection fails.
    """
    if image is None or image.size == 0 or _impl_detect_monitor is None:
        return None, None, None, None

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
    except Exception:
        return None, None, None, None

    if not _valid_points(points):
        return None, None, None, None

    points = _scale_points(points, scale)
    return tuple(np.asarray(p, dtype=np.float32) for p in points)


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    """
    Perspective-transform the detected monitor into a fixed 16:9 front view.

    If the detected monitor is portrait-like, rotate it into landscape before
    measuring angles, following the final rule clarification.
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
    uniform_bonus = 1.0 + max(0.0, min(0.45, (34.0 - color_std) / 80.0))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.count_nonzero(gray[yi, xi] < 90)) / max(samples, 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_ratio = float(
        np.count_nonzero((sat[yi, xi] > 45) & (val[yi, xi] > 55))
    ) / max(samples, 1)
    support_bonus = 0.75 + max(dark_ratio, color_ratio)

    border = max(4, int(min(h, w) * 0.025))
    touches_border = (
        min(x1, x2) < border
        or max(x1, x2) > w - 1 - border
        or min(y1, y2) < border
        or max(y1, y2) > h - 1 - border
    )
    border_penalty = 0.35 if touches_border and length > max(w, h) * 0.55 else 1.0

    return length * uniform_bonus * support_bonus * border_penalty


def _fast_hough_line(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark = (gray < 80).astype(np.uint8) * 255
    color = ((sat > 40) & (val > 55)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(dark, color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    edges = cv2.bitwise_or(cv2.Canny(blur, 35, 110), cv2.Canny(mask, 35, 110))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(min(h, w) * 0.045)),
        minLineLength=max(22, int(min(h, w) * 0.18)),
        maxLineGap=max(8, int(min(h, w) * 0.05)),
    )
    if lines is None:
        return None

    best = None
    best_score = -1.0
    for raw in lines[:, 0, :]:
        score = _line_quality(image, raw)
        if score > best_score:
            best_score = score
            best = raw

    if best is None:
        return None
    return np.asarray(best, dtype=np.int32)


def detect_line(rectified):
    """
    Detect the longest line inside the rectified monitor image.

    Return:
      (x1, y1, x2, y2)

    Return None if line detection fails.
    """
    if rectified is None or rectified.size == 0:
        return None

    h, w = rectified.shape[:2]
    margin_x = max(4, int(w * 0.035))
    margin_y = max(4, int(h * 0.045))
    inner = rectified[margin_y : h - margin_y, margin_x : w - margin_x]
    if inner.size == 0:
        inner = rectified
        margin_x = 0
        margin_y = 0

    impl_line = None
    if _impl_detect_line is not None:
        try:
            impl_line = _impl_detect_line(inner)
        except Exception:
            impl_line = None

    fast_line = _fast_hough_line(inner)
    impl_line = _normalize_line(impl_line)
    impl_score = _line_quality(inner, impl_line)
    fast_score = _line_quality(inner, fast_line)

    line = impl_line
    if fast_line is not None and fast_score > impl_score * 1.18:
        line = fast_line
    if line is None:
        return None

    x1, y1, x2, y2 = [int(round(float(v))) for v in line]
    return np.array(
        [x1 + margin_x, y1 + margin_y, x2 + margin_x, y2 + margin_y],
        dtype=np.int32,
    )


def calculate_angle(line) -> Optional[float]:
    """
    Calculate the line angle in degrees in rectified monitor coordinates.
    0 degrees is vertical, right-up is negative, left-up is positive.
    """
    if line is None:
        return None

    if _impl_calculate_angle is not None:
        try:
            angle = _impl_calculate_angle(line)
            if angle is not None and np.isfinite(angle):
                return float(angle)
        except Exception:
            pass

    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dy > 0 or (dy == 0 and dx > 0):
        dx = -dx
        dy = -dy
    return float(math.degrees(math.atan2(-dx, -dy)))


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")
        self.declare_parameter("monitor_refresh_interval", 10)
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("debug_every", 5)
        self.declare_parameter("publish_hz", 15.0)
        self.declare_parameter("process_every", 2)

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)
        self.monitor_refresh_interval = max(
            1, int(self.get_parameter("monitor_refresh_interval").value)
        )
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.debug_every = max(1, int(self.get_parameter("debug_every").value))
        self.publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.process_every = max(1, int(self.get_parameter("process_every").value))

        self.bridge = CvBridge()
        self.frame_index = 0
        self.cached_corners = None
        self.last_angle = 0.0
        self.angle_lock = Lock()
        self.angle_history = deque(maxlen=3)
        self.line_failures = 0
        self.prev_scene_thumb = None

        self.image_group = MutuallyExclusiveCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            1,
            callback_group=self.image_group,
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

        self.publish_timer = self.create_timer(
            1.0 / self.publish_hz,
            self._timer_publish_angle,
            callback_group=self.timer_group,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        self.frame_index += 1
        if self.frame_index % self.process_every != 0:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            if self.frame_index % 90 == 0:
                self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        scene_changed = self._scene_changed(image)
        if scene_changed:
            self.angle_history.clear()

        refreshed = False
        need_refresh = (
            self.cached_corners is None
            or self.frame_index % self.monitor_refresh_interval == 1
            or self.line_failures >= 4
            or scene_changed
        )

        if need_refresh:
            corners = detect_monitor(image)
            refreshed = True
            if _valid_points(corners) and self._accept_monitor_candidate(
                image.shape, corners, scene_changed
            ):
                self.cached_corners = corners
            elif scene_changed:
                self.cached_corners = None

        if self.cached_corners is None:
            if self.frame_index % 90 == 0:
                self.get_logger().warning("Monitor not detected; publishing fallback.")
            return

        rectified, line = self._process_with_corners(image, self.cached_corners)

        if line is None and not refreshed:
            corners = detect_monitor(image)
            if _valid_points(corners) and self._accept_monitor_candidate(
                image.shape, corners, scene_changed
            ):
                self.cached_corners = corners
                rectified, line = self._process_with_corners(image, self.cached_corners)

        if line is not None:
            angle = calculate_angle(line)
            if angle is not None and np.isfinite(angle):
                raw_angle = float(angle)
                if self._accept_angle(raw_angle, scene_changed):
                    angle = self._smooth_angle(raw_angle)
                    with self.angle_lock:
                        self.last_angle = angle
                    self.line_failures = 0
                    if (
                        self.publish_debug
                        and rectified is not None
                        and self.frame_index % self.debug_every == 0
                    ):
                        self._debug_line(msg, rectified, line)
                else:
                    self.line_failures += 1
            else:
                self.line_failures += 1
        else:
            self.line_failures += 1

        if self.frame_index % 90 == 0:
            with self.angle_lock:
                angle = self.last_angle
            self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _scene_changed(self, image) -> bool:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        except Exception:
            return False

        if self.prev_scene_thumb is None:
            self.prev_scene_thumb = thumb
            return False

        diff = float(np.mean(cv2.absdiff(thumb, self.prev_scene_thumb)))
        self.prev_scene_thumb = thumb
        return diff > 18.0

    def _accept_monitor_candidate(self, image_shape, candidate, scene_changed: bool) -> bool:
        if self.cached_corners is None or scene_changed or self.line_failures >= 4:
            return True

        current = np.asarray(self.cached_corners, dtype=np.float32).reshape(4, 2)
        new = np.asarray(candidate, dtype=np.float32).reshape(4, 2)
        cur_area = abs(cv2.contourArea(current))
        new_area = abs(cv2.contourArea(new))
        if cur_area < 1 or new_area < 1:
            return True

        h, w = image_shape[:2]
        diag = math.hypot(float(w), float(h))
        center_shift = float(np.linalg.norm(np.mean(current, axis=0) - np.mean(new, axis=0)))
        area_ratio = new_area / cur_area

        if center_shift > diag * 0.18 and not (0.65 <= area_ratio <= 1.55):
            return False
        if area_ratio < 0.35 or area_ratio > 2.8:
            return False
        return True

    def _accept_angle(self, angle: float, scene_changed: bool) -> bool:
        if scene_changed or len(self.angle_history) < 2:
            return True

        prev = float(np.median(np.asarray(self.angle_history, dtype=np.float32)))
        diff = abs(angle - prev) % 180.0
        diff = min(diff, 180.0 - diff)
        return diff < 55.0 or self.line_failures >= 2

    def _process_with_corners(self, image, corners):
        rectified = rectify_monitor(image, *corners)
        if rectified is None:
            self.cached_corners = None
            return None, None
        line = detect_line(rectified)
        return rectified, line

    def _smooth_angle(self, angle: float) -> float:
        while angle > 90.0:
            angle -= 180.0
        while angle <= -90.0:
            angle += 180.0

        self.angle_history.append(angle)
        if len(self.angle_history) < 3:
            return angle
        return float(np.median(np.asarray(self.angle_history, dtype=np.float32)))

    def _publish_angle(self, angle: float) -> None:
        angle_msg = Float32()
        if angle is None or not np.isfinite(angle):
            angle = 0.0
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

    def _timer_publish_angle(self) -> None:
        with self.angle_lock:
            angle = self.last_angle
        self._publish_angle(angle)

    def _debug_line(self, msg, rectified, line) -> None:
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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
