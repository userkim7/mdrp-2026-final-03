import math
import cv2
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - visualization is optional in ROS runs.
    plt = None

try:
    from utils import plt_show  # only for Elice
except Exception:  # pragma: no cover - final ROS environment may not ship utils.py.
    def plt_show():
        pass


ASPECT_W = 16.0
ASPECT_H = 9.0
ASPECT_RATIO = ASPECT_W / ASPECT_H

_line_for_angle = None
_rectified_to_original = None
_monitor_top_angle = None
_bright_sign_mode = None
_photo_sign_mode = False


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


def _quad_from_points(pts):
    pts = np.asarray(pts, dtype="float32").reshape(-1, 2)
    if len(pts) == 4:
        return np.array(_order_points(pts), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    quad = np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype="float32",
    )

    # If the extreme-point heuristic duplicates a corner, fall back to a
    # low-weight rectangle instead of crashing. This should be rare.
    unique = np.unique(np.round(quad).astype(np.int32), axis=0)
    if len(unique) < 4:
        rect = cv2.minAreaRect(pts)
        quad = cv2.boxPoints(rect).astype("float32")

    return np.array(_order_points(quad), dtype="float32")


def _angle_to_vertical(line):
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    dy = y2 - y1

    # A line has no direction. Use the endpoint ordering that points upward
    # in the rectified monitor coordinate system.
    if dy > 0 or (dy == 0 and dx > 0):
        dx = -dx
        dy = -dy

    # 0 degree is the upward vertical direction. Right-up is negative,
    # left-up is positive.
    return math.degrees(math.atan2(-dx, -dy))


def detect_monitor(image):
    global _line_for_angle, _monitor_top_angle, _bright_sign_mode, _photo_sign_mode

    _line_for_angle = None
    _monitor_top_angle = None
    _bright_sign_mode = False
    _photo_sign_mode = False

    orig_h, orig_w = image.shape[:2]
    coord_scale = 1.0
    # Detection does not need full-resolution coordinates.  Keeping the long
    # side near 1000px cuts contour/Hough cost a lot while preserving corner
    # geometry well enough for the final full-resolution warp.
    max_detection_side = 1200.0
    max_side = float(max(orig_h, orig_w))
    if max_side > max_detection_side:
        coord_scale = max_side / max_detection_side
        resized_w = max(1, int(round(orig_w / coord_scale)))
        resized_h = max(1, int(round(orig_h / coord_scale)))
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(raw_gray, (5, 5), 0)
    h, w = gray.shape[:2]

    def quad_score(quad):
        area = abs(cv2.contourArea(quad.astype("float32")))
        if area < h * w * 0.005:
            return -1
        if area > h * w * 0.85:
            return -1

        top_left, top_right, bottom_right, bottom_left = _order_points(quad)
        width = (
            np.linalg.norm(top_right - top_left)
            + np.linalg.norm(bottom_right - bottom_left)
        ) / 2
        height = (
            np.linalg.norm(bottom_left - top_left)
            + np.linalg.norm(bottom_right - top_right)
        ) / 2

        if width < 30 or height < 30:
            return -1

        ratio = width / max(height, 1)
        if ratio < 0.35 or ratio > 6.0:
            return -1

        return area

    border_metric_cache = {}

    def black_border_score(quad):
        quad = np.array(_order_points(quad), dtype=np.float32)
        cache_key = tuple(np.round(quad.reshape(-1), 1))
        if cache_key in border_metric_cache:
            return border_metric_cache[cache_key]
        side_scores = []

        for idx in range(4):
            p1 = quad[idx]
            p2 = quad[(idx + 1) % 4]
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 12:
                continue

            samples = max(12, int(length))
            xs = np.linspace(p1[0], p2[0], samples)
            ys = np.linspace(p1[1], p2[1], samples)
            norm = max(length, 1.0)
            nx = -vec[1] / norm
            ny = vec[0] / norm
            best_side_score = 0.0

            for offset in (-5, -3, -1, 0, 1, 3, 5):
                xx = np.round(xs + nx * offset).astype(np.int32)
                yy = np.round(ys + ny * offset).astype(np.int32)

                valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                if not np.any(valid):
                    continue

                values = gray[yy[valid], xx[valid]]
                side_score = np.count_nonzero(values < 95) / max(int(np.count_nonzero(valid)), 1)
                best_side_score = max(best_side_score, float(side_score))

            side_scores.append(best_side_score)

        if len(side_scores) < 4:
            border_metric_cache[cache_key] = 0.0
            return 0.0

        side_scores = np.array(side_scores, dtype=np.float32)
        avg_score = float(np.mean(side_scores))
        min_score = float(np.min(side_scores))
        supported_sides = float(np.count_nonzero(side_scores > 0.16)) / 4.0

        # A real monitor frame should show evidence on all four sides.
        # Large accidental quadrilaterals often have one dark side and one
        # scene edge, so make the weakest side matter without completely
        # rejecting a monitor whose one side is antialiased or glare-covered.
        result = avg_score * (0.25 + 0.75 * supported_sides) * (0.40 + 0.60 * min_score)
        border_metric_cache[cache_key] = result
        return result

    frame_metric_cache = {}
    detail_metric_cache = {}

    def solid_black_frame_metrics(quad):
        quad = np.array(_order_points(quad), dtype=np.float32)
        cache_key = tuple(np.round(quad.reshape(-1), 1))
        if cache_key in frame_metric_cache:
            return frame_metric_cache[cache_key]

        side_scores = []
        band = int(max(3, min(8, round(min(h, w) * 0.006))))
        offsets = range(-band, band + 1)

        def longest_run(flags):
            best = 0
            cur = 0
            for flag in flags:
                if flag:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        for idx in range(4):
            p1 = quad[idx]
            p2 = quad[(idx + 1) % 4]
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 18:
                continue

            samples = max(24, min(220, int(round(length))))
            xs = np.linspace(p1[0], p2[0], samples)
            ys = np.linspace(p1[1], p2[1], samples)
            nx = -vec[1] / max(length, 1.0)
            ny = vec[0] / max(length, 1.0)

            valid_any = np.zeros(samples, dtype=bool)
            dark_any = np.zeros(samples, dtype=bool)
            strict_any = np.zeros(samples, dtype=bool)

            for offset in offsets:
                xx = np.round(xs + nx * offset).astype(np.int32)
                yy = np.round(ys + ny * offset).astype(np.int32)
                valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
                if not np.any(valid):
                    continue

                values = raw_gray[yy[valid], xx[valid]]
                valid_any[valid] = True
                valid_idx = np.where(valid)[0]
                dark_any[valid_idx] |= values < 58
                strict_any[valid_idx] |= values < 34

            denom = max(int(np.count_nonzero(valid_any)), 1)
            side_flags = dark_any & valid_any
            dark_frac = float(np.count_nonzero(side_flags) / denom)
            strict_frac = float(np.count_nonzero(strict_any & valid_any) / denom)
            run_frac = float(longest_run(side_flags) / denom)
            side_scores.append(0.45 * dark_frac + 0.35 * run_frac + 0.20 * strict_frac)

        if len(side_scores) < 4:
            return 0.0, 0.0, 0, 0.0

        side_scores = np.array(side_scores, dtype=np.float32)
        avg_score = float(np.mean(side_scores))
        min_score = float(np.min(side_scores))
        supported = int(np.count_nonzero(side_scores > 0.24))
        score = avg_score * (0.25 + 0.75 * supported / 4.0) * (0.35 + 0.65 * min_score)
        result = (score, min_score, supported, avg_score)
        frame_metric_cache[cache_key] = result
        return result

    def quad_internal_detail_score(quad):
        quad = np.array(_order_points(quad), dtype=np.float32)
        cache_key = tuple(np.round(quad.reshape(-1), 1))
        if cache_key in detail_metric_cache:
            return detail_metric_cache[cache_key]

        tl, tr, br, bl = _order_points(quad)
        avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        if avg_h > avg_w:
            src_quad = np.ascontiguousarray(np.array([bl, tl, tr, br], dtype=np.float32))
        else:
            src_quad = np.ascontiguousarray(np.array([tl, tr, br, bl], dtype=np.float32))

        test_w, test_h = 128, 72
        dst_quad = np.ascontiguousarray(
            np.array(
                [[0, 0], [test_w - 1, 0], [test_w - 1, test_h - 1], [0, test_h - 1]],
                dtype=np.float32,
            )
        )
        try:
            transform = cv2.getPerspectiveTransform(src_quad, dst_quad)
            warped = cv2.warpPerspective(image, transform, (test_w, test_h))
        except cv2.error:
            detail_metric_cache[cache_key] = 0.0
            return 0.0

        gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        inner = gray_warp[5:test_h - 5, 7:test_w - 7]
        if inner.size == 0:
            detail_metric_cache[cache_key] = 0.0
            return 0.0

        edges_inner = cv2.Canny(inner, 24, 88)
        result = cv2.countNonZero(edges_inner) / max(float(inner.size), 1.0)
        detail_metric_cache[cache_key] = result
        return result

    def strong_frame_rank(item):
        base_score, quad = item
        quad = np.asarray(quad, dtype=np.float32)
        frame, min_side, supported, avg_frame = solid_black_frame_metrics(quad)
        area = abs(cv2.contourArea(quad))
        area_frac = area / max(h * w, 1)

        tl, tr, br, bl = _order_points(quad)
        avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        ratio = avg_w / max(avg_h, 1.0)
        target_ratio = ASPECT_RATIO if ratio >= 1.0 else 1.0 / ASPECT_RATIO
        ratio_penalty = 1.0 / (1.0 + 0.55 * abs(ratio - target_ratio))

        border = black_border_score(quad)
        frame_strength = max(frame, border * 0.70)
        score = area * (0.25 + 5.20 * frame_strength) * ratio_penalty
        if min_side < 0.10:
            score *= 0.30
        if avg_frame < 0.30:
            score *= 0.35
        if area_frac > 0.45 and avg_frame < 0.48:
            score *= 0.12
        if area_frac > 0.65:
            score *= 0.18
        return score + base_score * 0.01

    def is_strong_frame_candidate(item):
        _, quad = item
        frame, min_side, supported, avg_frame = solid_black_frame_metrics(quad)
        if supported >= 4 and frame >= 0.115 and min_side >= 0.065 and avg_frame >= 0.24:
            return True
        if supported >= 4 and frame >= 0.095 and min_side >= 0.10 and avg_frame >= 0.34:
            return True
        return False

    def candidate_rank(item):
        base_score, quad = item
        quad = np.asarray(quad, dtype=np.float32)
        border = black_border_score(quad)
        frame, min_side, supported, avg_frame = solid_black_frame_metrics(quad)
        area = abs(cv2.contourArea(quad))
        area_frac = area / max(h * w, 1)

        candidate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(candidate_mask, quad.astype(np.int32), 255)
        if np.count_nonzero(candidate_mask) > 0:
            mean_sat = cv2.mean(hsv_for_monitor[:, :, 1], mask=candidate_mask)[0]
            mean_val = cv2.mean(hsv_for_monitor[:, :, 2], mask=candidate_mask)[0]
        else:
            mean_sat = 255
            mean_val = 0

        tl, tr, br, bl = _order_points(quad)
        avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        shape_ratio = avg_w / max(avg_h, 1.0)

        # The rule defines the monitor by a black border and largest interior.
        # In busy photos, though, the true monitor can be a small bright,
        # low-saturation sign whose fitted quad lies on the inner screen rather
        # than exactly on the black frame. Protect those candidates, while
        # heavily penalizing large colorful scene quads with weak frame support.
        bright_screen_like = (
            area_frac < 0.35
            and mean_val > 105
            and mean_sat < 145
            and (border > 0.045 or frame > 0.08)
            and 0.35 <= shape_ratio <= 4.4
        )

        if bright_screen_like:
            return base_score * (0.75 + 1.45 * min(max(border, frame) / 0.24, 1.0))

        frame_bonus = 0.70 + 3.20 * min(frame, 1.0)
        border_factor = (0.22 + 1.20 * min(border, 1.0)) * frame_bonus
        if border < 0.030 and frame < 0.060:
            border_factor *= 0.015
        elif border < 0.055 and frame < 0.090:
            border_factor *= 0.045
        elif border < 0.090 and frame < 0.120:
            border_factor *= 0.16
        elif border < 0.130 and frame < 0.160:
            border_factor *= 0.42

        rank = base_score * border_factor
        if area_frac > 0.30 and mean_sat > 45 and max(border, frame) < 0.18:
            rank *= 0.08
        if area_frac > 0.55 and mean_sat > 35 and supported < 4:
            rank *= 0.20
        if area_frac > 0.60 and avg_frame < 0.30:
            rank *= 0.12
        if area_frac > 0.45 and supported < 4:
            rank *= 0.18
        if area_frac > 0.65:
            rank *= 0.08
        if max(border, frame) < 0.11:
            detail_score = quad_internal_detail_score(quad)
            if detail_score < 0.012:
                rank *= 0.08
            elif detail_score < 0.024:
                rank *= 0.35
            else:
                rank *= 1.0 + min(0.25, detail_score * 2.5)

        return rank

    candidates = []

    edges = cv2.Canny(gray, 30, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)

    dark = (gray < 90).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(list(dark_contours))

    solid_black = (raw_gray < 58).astype(np.uint8) * 255
    solid_black = cv2.morphologyEx(
        solid_black,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    black_contours, _ = cv2.findContours(solid_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(list(black_contours))

    hsv_for_monitor = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_for_monitor = hsv_for_monitor[:, :, 1]
    val_for_monitor = hsv_for_monitor[:, :, 2]
    bright = ((val_for_monitor > 145) & (sat_for_monitor < 95)).astype(np.uint8) * 255
    for close_size in (9, 31):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        bright_closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        bright_closed = cv2.morphologyEx(bright_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        bright_contours, _ = cv2.findContours(bright_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bright_contours = list(bright_contours)
        contours.extend(bright_contours)

        for bright_cnt in bright_contours:
            if cv2.contourArea(bright_cnt) < h * w * 0.003:
                continue

            added_bright_quad = False
            peri = cv2.arcLength(bright_cnt, True)
            for eps in (0.01, 0.018, 0.028, 0.04):
                approx = cv2.approxPolyDP(bright_cnt, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                quad = approx.reshape(4, 2).astype("float32")
                score = quad_score(quad)
                if score <= 0:
                    continue

                tl, tr, br, bl = _order_points(quad)
                bright_width = (
                    np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
                ) / 2
                bright_height = (
                    np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)
                ) / 2
                bright_ratio = bright_width / max(bright_height, 1)
                if bright_ratio < 1.2 or bright_ratio > 6.5:
                    continue

                xs = quad[:, 0]
                ys = quad[:, 1]
                if xs.min() < 3 or ys.min() < 3 or xs.max() > w - 4 or ys.max() > h - 4:
                    continue

                candidates.append((score * 12.0, quad))
                added_bright_quad = True
                break

            if not added_bright_quad:
                hull = cv2.convexHull(bright_cnt).reshape(-1, 2).astype("float32")
                if len(hull) >= 4:
                    quad = _quad_from_points(hull)
                    score = quad_score(quad)
                    if score > 0:
                        tl, tr, br, bl = _order_points(quad)
                        bright_width = (
                            np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
                        ) / 2
                        bright_height = (
                            np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)
                        ) / 2
                        bright_ratio = bright_width / max(bright_height, 1)

                        xs = quad[:, 0]
                        ys = quad[:, 1]
                        touches_border = (
                            xs.min() < 3 or ys.min() < 3 or xs.max() > w - 4 or ys.max() > h - 4
                        )
                        if 1.2 <= bright_ratio <= 6.5 and not touches_border:
                            candidates.append((score * 10.0, quad))
                            added_bright_quad = True

            # Do not use minAreaRect here: it forces a perspective-skewed
            # screen into a rotated rectangle and can include background.

    for close_size in (19, 43):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        closed_dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        closed_contours, _ = cv2.findContours(closed_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(list(closed_contours))

    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, th = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        more_contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(list(more_contours))

    def angle_diff(a, b):
        diff = abs(a - b) % 180
        return min(diff, 180 - diff)

    def line_from_segment(seg):
        x1, y1, x2, y2 = seg.astype(float)
        theta = math.atan2(y2 - y1, x2 - x1)
        nx = -math.sin(theta)
        ny = math.cos(theta)
        rho = nx * ((x1 + x2) / 2) + ny * ((y1 + y2) / 2)
        if rho < 0:
            rho = -rho
            nx = -nx
            ny = -ny
        return np.array([nx, ny, rho], dtype=np.float32)

    def intersect(line1, line2):
        a = np.array([[line1[0], line1[1]], [line2[0], line2[1]]], dtype=np.float32)
        b = np.array([line1[2], line2[2]], dtype=np.float32)
        det = np.linalg.det(a)
        if abs(det) < 1e-4:
            return None
        return np.linalg.solve(a, b)

    def refine_screen_quad(top_left, top_right, bottom_right, bottom_left):
        src = np.ascontiguousarray(
            np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        )
        src_tl, src_tr, src_br, src_bl = _order_points(src)
        src_w = (
            np.linalg.norm(src_tr - src_tl)
            + np.linalg.norm(src_br - src_bl)
        ) / 2.0
        src_h = (
            np.linalg.norm(src_bl - src_tl)
            + np.linalg.norm(src_br - src_tr)
        ) / 2.0
        warp_src = src
        if src_h > src_w:
            # Portrait monitors are still evaluated in a landscape 16:9
            # coordinate system: rotate the long side onto the x-axis.
            warp_src = np.ascontiguousarray(
                np.array([src_bl, src_tl, src_tr, src_br], dtype=np.float32)
            )
            test_w, test_h = 481, 271
        else:
            test_w, test_h = 481, 271
        dst = np.ascontiguousarray(
            np.array(
                [[0, 0], [test_w - 1, 0], [test_w - 1, test_h - 1], [0, test_h - 1]],
                dtype=np.float32,
            )
        )
        if abs(cv2.contourArea(src)) < 10:
            return top_left, top_right, bottom_right, bottom_left

        try:
            transform = cv2.getPerspectiveTransform(warp_src, dst)
            warped = cv2.warpPerspective(image, transform, (test_w, test_h))
        except cv2.error:
            return top_left, top_right, bottom_right, bottom_left

        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sat = warped_hsv[:, :, 1]
        val = warped_hsv[:, :, 2]

        # Screen backgrounds can be white, light gray, or mid gray. Use low
        # saturation plus a moderate value threshold instead of only looking
        # for very bright white regions.
        screen_mask = (
            ((val > 118) & (sat < 145))
            | ((val > 82) & (sat < 55))
        ).astype(np.uint8) * 255
        screen_mask = cv2.morphologyEx(
            screen_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
            iterations=2,
        )
        screen_mask = cv2.morphologyEx(
            screen_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )

        screen_contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_screen = None
        best_screen_score = -1.0
        for cnt in screen_contours:
            area = cv2.contourArea(cnt)
            if area < test_w * test_h * 0.10:
                continue

            rect = cv2.boundingRect(cnt)
            bx, by, bw, bh = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            touches_every_side = (
                bx <= 2
                and by <= 2
                and bx + bw >= test_w - 3
                and by + bh >= test_h - 3
            )
            if touches_every_side:
                continue
            if bw < test_w * 0.35 or bh < test_h * 0.25:
                continue
            if by > test_h * 0.35:
                continue

            center_penalty = abs((bx + bw / 2.0) - test_w / 2.0) / max(test_w / 2.0, 1)
            score = area * (1.0 - 0.15 * center_penalty)
            if score > best_screen_score:
                best_screen_score = score
                best_screen = cnt

        if best_screen is not None:
            peri = cv2.arcLength(best_screen, True)
            screen_quad = None
            for eps in (0.018, 0.025, 0.035, 0.05, 0.075):
                approx = cv2.approxPolyDP(best_screen, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    screen_quad = approx.reshape(4, 2).astype(np.float32)
                    break

            if screen_quad is None:
                hull = cv2.convexHull(best_screen).reshape(-1, 2).astype(np.float32)
                if len(hull) >= 4:
                    screen_quad = _quad_from_points(hull)

            if screen_quad is not None:
                screen_quad = np.ascontiguousarray(np.array(_order_points(screen_quad), dtype=np.float32))
            if screen_quad is not None and abs(cv2.contourArea(screen_quad)) > test_w * test_h * 0.12:
                screen_area = abs(cv2.contourArea(screen_quad))
                if screen_area < test_w * test_h * 0.72:
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    detail_edges = cv2.Canny(warped_gray, 24, 88)
                    detail_mask = np.zeros((test_h, test_w), dtype=np.uint8)
                    cv2.fillConvexPoly(detail_mask, screen_quad.astype(np.int32), 255)
                    inside_pixels = max(int(np.count_nonzero(detail_mask)), 1)
                    outside_pixels = max(test_w * test_h - inside_pixels, 1)
                    inside_detail = cv2.countNonZero(cv2.bitwise_and(detail_edges, detail_mask)) / inside_pixels
                    outside_detail = cv2.countNonZero(
                        cv2.bitwise_and(detail_edges, cv2.bitwise_not(detail_mask))
                    ) / outside_pixels
                    if inside_detail < 0.010 and outside_detail > max(0.006, inside_detail * 1.35):
                        screen_quad = None

            if screen_quad is not None and abs(cv2.contourArea(screen_quad)) > test_w * test_h * 0.12:
                center = screen_quad.mean(axis=0)
                screen_quad = (screen_quad - center) * 1.035 + center
                screen_quad[:, 0] = np.clip(screen_quad[:, 0], 0, test_w - 1)
                screen_quad[:, 1] = np.clip(screen_quad[:, 1], 0, test_h - 1)
                screen_quad = np.ascontiguousarray(screen_quad.astype(np.float32))

                try:
                    inverse = cv2.getPerspectiveTransform(dst, warp_src)
                    mapped = cv2.perspectiveTransform(screen_quad.reshape(1, 4, 2), inverse)[0]
                except cv2.error:
                    return top_left, top_right, bottom_right, bottom_left
                ntl, ntr, nbr, nbl = _order_points(mapped)

                old_area = abs(cv2.contourArea(src))
                new_area = abs(cv2.contourArea(np.array([ntl, ntr, nbr, nbl], dtype=np.float32)))
                if old_area * 0.25 <= new_area <= old_area * 1.08:
                    return ntl, ntr, nbr, nbl

        screen_like = ((val > 145) & (sat < 120)).astype(np.float32)
        colorful = ((sat > 80) & (val > 70)).astype(np.float32)
        row_score = screen_like.mean(axis=1) - 0.35 * colorful.mean(axis=1)
        smooth = np.convolve(row_score, np.ones(9, dtype=np.float32) / 9.0, mode="same")

        best_y = None
        best_drop = 0.0
        for y in range(int(test_h * 0.48), int(test_h * 0.90)):
            above = float(np.mean(smooth[max(0, y - 18):y]))
            below = float(np.mean(smooth[y:min(test_h, y + 14)]))
            drop = above - below
            if above > 0.30 and below < 0.32 and drop > best_drop:
                best_drop = drop
                best_y = y

        if best_y is None or best_drop < 0.12:
            return top_left, top_right, bottom_right, bottom_left

        y_cut = min(test_h - 2, best_y + 4)
        if y_cut > test_h * 0.86 or y_cut < test_h * 0.55:
            return top_left, top_right, bottom_right, bottom_left

        bottom_probe = np.array(
            [[[0, y_cut], [test_w - 1, y_cut]]],
            dtype=np.float32,
        )
        try:
            inverse = cv2.getPerspectiveTransform(dst, warp_src)
            new_bottom = cv2.perspectiveTransform(bottom_probe, inverse)[0]
        except cv2.error:
            return top_left, top_right, bottom_right, bottom_left
        new_bottom_left = new_bottom[0]
        new_bottom_right = new_bottom[1]

        old_height = (
            np.linalg.norm(bottom_left - top_left)
            + np.linalg.norm(bottom_right - top_right)
        ) / 2.0
        new_height = (
            np.linalg.norm(new_bottom_left - top_left)
            + np.linalg.norm(new_bottom_right - top_right)
        ) / 2.0
        if new_height < old_height * 0.55:
            return top_left, top_right, bottom_right, bottom_left

        return top_left, top_right, new_bottom_right, new_bottom_left

    hough_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, int(min(h, w) * 0.06)),
        minLineLength=max(60, int(min(h, w) * 0.18)),
        maxLineGap=max(15, int(min(h, w) * 0.04)),
    )

    if hough_lines is not None and len(hough_lines) >= 4:
        segs = []
        angle_weights = np.zeros(180, dtype=np.float32)

        for raw in hough_lines[:, 0, :]:
            x1, y1, x2, y2 = raw.astype(float)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < max(60, min(h, w) * 0.14):
                continue

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            segs.append((raw.astype(np.float32), angle, length))
            angle_weights[int(round(angle)) % 180] += length

        if len(segs) >= 4:
            angle1 = int(np.argmax(angle_weights))
            angle2 = None
            best_weight = -1
            for a in range(180):
                diff = angle_diff(a, angle1)
                if 45 <= diff <= 135 and angle_weights[a] > best_weight:
                    best_weight = angle_weights[a]
                    angle2 = a

            if angle2 is not None:
                for tol in (8, 12, 16):
                    group1 = [s for s in segs if angle_diff(s[1], angle1) <= tol]
                    group2 = [s for s in segs if angle_diff(s[1], angle2) <= tol]
                    if len(group1) < 2 or len(group2) < 2:
                        continue

                    lines1 = [(line_from_segment(s[0]), s[2]) for s in group1]
                    lines2 = [(line_from_segment(s[0]), s[2]) for s in group2]
                    lines1.sort(key=lambda item: item[0][2])
                    lines2.sort(key=lambda item: item[0][2])

                    line1_low = lines1[0][0]
                    line1_high = lines1[-1][0]
                    line2_low = lines2[0][0]
                    line2_high = lines2[-1][0]

                    pts = [
                        intersect(line1_low, line2_low),
                        intersect(line1_low, line2_high),
                        intersect(line1_high, line2_high),
                        intersect(line1_high, line2_low),
                    ]

                    if any(pt is None for pt in pts):
                        continue

                    quad = np.array(pts, dtype=np.float32)
                    if np.any(quad[:, 0] < -w * 0.15) or np.any(quad[:, 0] > w * 1.15):
                        continue
                    if np.any(quad[:, 1] < -h * 0.15) or np.any(quad[:, 1] > h * 1.15):
                        continue

                    score = quad_score(quad)
                    if score > 0:
                        candidates.append((score * 0.75, quad))
                        break

    if len(contours) > 520:
        contours.sort(key=cv2.contourArea, reverse=True)
        contours = contours[:520]

    for cnt in contours:
        if cv2.contourArea(cnt) < h * w * 0.02:
            continue

        added_contour_quad = False
        peri = cv2.arcLength(cnt, True)
        for eps in (0.015, 0.025, 0.04, 0.06):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype("float32")
                score = quad_score(quad)
                if score > 0:
                    xs = quad[:, 0]
                    ys = quad[:, 1]
                    bbox_ratio = (xs.max() - xs.min()) / max(ys.max() - ys.min(), 1)
                    if score > h * w * 0.18 and bbox_ratio < 1.25:
                        score *= 0.15
                    candidates.append((score, quad))
                    added_contour_quad = True
                break

        if not added_contour_quad:
            hull = cv2.convexHull(cnt).reshape(-1, 2).astype("float32")
            if len(hull) >= 4:
                quad = _quad_from_points(hull)
                score = quad_score(quad)
                if score > 0:
                    xs = quad[:, 0]
                    ys = quad[:, 1]
                    bbox_ratio = (xs.max() - xs.min()) / max(ys.max() - ys.min(), 1)
                    if score > h * w * 0.18 and bbox_ratio < 1.25:
                        score *= 0.15
                    candidates.append((score * 0.9, quad))
                    added_contour_quad = True

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype("float32")
        score = quad_score(box)
        if score > 0:
            xs = box[:, 0]
            ys = box[:, 1]
            bbox_ratio = (xs.max() - xs.min()) / max(ys.max() - ys.min(), 1)
            if score > h * w * 0.18 and bbox_ratio < 1.25:
                score *= 0.15
            rect_weight = 0.25 if added_contour_quad else 0.55
            candidates.append((score * rect_weight, box))

    if len(candidates) == 0:
        top_left = np.array([0, 0], dtype="float32")
        top_right = np.array([orig_w - 1, 0], dtype="float32")
        bottom_right = np.array([orig_w - 1, orig_h - 1], dtype="float32")
        bottom_left = np.array([0, orig_h - 1], dtype="float32")
        return top_left, top_right, bottom_right, bottom_left

    if len(candidates) > 360:
        def quick_keep_rank(item):
            base_score, quad = item
            quad = np.asarray(quad, dtype=np.float32)
            border = black_border_score(quad)
            area = abs(cv2.contourArea(quad))
            area_frac = area / max(h * w, 1)
            rank = base_score * (0.10 + 2.40 * min(border, 1.0))
            if border > 0.06:
                rank *= 2.5
            if area_frac > 0.55 and border < 0.08:
                rank *= 0.10
            return rank

        candidates.sort(key=quick_keep_rank, reverse=True)
        candidates = candidates[:360]

    strong_frame_candidates = [item for item in candidates if is_strong_frame_candidate(item)]
    if strong_frame_candidates:
        _, best_quad = max(strong_frame_candidates, key=strong_frame_rank)
    else:
        _, best_quad = max(candidates, key=candidate_rank)
    top_left, top_right, bottom_right, bottom_left = _order_points(best_quad)

    _line_for_angle = None
    selected_area = abs(cv2.contourArea(np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)))
    selected_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(
        selected_mask,
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32),
        255,
    )
    selected_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if np.count_nonzero(selected_mask) > 0:
        mean_sat = cv2.mean(selected_hsv[:, :, 1], mask=selected_mask)[0]
        mean_val = cv2.mean(selected_hsv[:, :, 2], mask=selected_mask)[0]
    else:
        mean_sat = 255
        mean_val = 0
    _bright_sign_mode = (selected_area < h * w * 0.25 and mean_val > 125 and mean_sat < 120)
    _photo_sign_mode = _bright_sign_mode and cv2.mean(selected_hsv[:, :, 1])[0] > 25

    should_refine_quad = (
        selected_area < h * w * 0.70
        and mean_val > 70
    )
    if should_refine_quad:
        top_left, top_right, bottom_right, bottom_left = refine_screen_quad(
            top_left, top_right, bottom_right, bottom_left
        )
        selected_area = abs(cv2.contourArea(np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)))
        selected_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(
            selected_mask,
            np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32),
            255,
        )
        if np.count_nonzero(selected_mask) > 0:
            mean_sat = cv2.mean(selected_hsv[:, :, 1], mask=selected_mask)[0]
            mean_val = cv2.mean(selected_hsv[:, :, 2], mask=selected_mask)[0]
        _bright_sign_mode = (selected_area < h * w * 0.25 and mean_val > 125 and mean_sat < 120)
        _photo_sign_mode = _bright_sign_mode and cv2.mean(selected_hsv[:, :, 1])[0] > 25

    top_angle = math.degrees(
        math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
    )
    bottom_angle = math.degrees(
        math.atan2(bottom_right[1] - bottom_left[1], bottom_right[0] - bottom_left[0])
    )
    if top_angle > 90:
        top_angle -= 180
    if top_angle <= -90:
        top_angle += 180
    if bottom_angle > 90:
        bottom_angle -= 180
    if bottom_angle <= -90:
        bottom_angle += 180
    _monitor_top_angle = (top_angle + bottom_angle) / 2

    if coord_scale != 1.0:
        top_left = (top_left * coord_scale).astype("float32")
        top_right = (top_right * coord_scale).astype("float32")
        bottom_right = (bottom_right * coord_scale).astype("float32")
        bottom_left = (bottom_left * coord_scale).astype("float32")

    return top_left, top_right, bottom_right, bottom_left


def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    global _rectified_to_original

    if top_left is None or top_right is None or bottom_right is None or bottom_left is None:
        _rectified_to_original = None
        return image

    def refine_rectified_frame(rectified):
        rh, rw = rectified.shape[:2]
        rgray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        dark = (rgray < 85).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        dark = cv2.dilate(
            dark,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        best_quad = None
        best_score = -1.0

        def fit_side_line(points):
            if len(points) < max(25, int(min(rh, rw) * 0.05)):
                return None

            try:
                vx, vy, x0, y0 = cv2.fitLine(
                    points.astype(np.float32),
                    cv2.DIST_HUBER,
                    0,
                    0.01,
                    0.01,
                ).reshape(-1)
            except cv2.error:
                return None

            vx = float(vx)
            vy = float(vy)
            norm = math.hypot(vx, vy)
            if norm < 1e-6:
                return None

            vx /= norm
            vy /= norm
            nx = -vy
            ny = vx
            rho = nx * float(x0) + ny * float(y0)
            if rho < 0:
                nx = -nx
                ny = -ny
                rho = -rho

            return np.array([nx, ny, rho], dtype=np.float32)

        def intersect_fit_lines(line1, line2):
            a = np.array(
                [[line1[0], line1[1]], [line2[0], line2[1]]],
                dtype=np.float32,
            )
            b = np.array([line1[2], line2[2]], dtype=np.float32)
            det = np.linalg.det(a)
            if abs(det) < 1e-4:
                return None
            return np.linalg.solve(a, b)

        def refine_quad_edges(quad):
            edge_map = cv2.Canny(rgray, 35, 125)
            edge_map = cv2.bitwise_and(
                edge_map,
                cv2.dilate(
                    dark,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1,
                ),
            )
            ys, xs = np.where(edge_map > 0)
            if len(xs) < 80:
                return None

            edge_pts = np.column_stack((xs, ys)).astype(np.float32)
            quad = np.array(_order_points(quad), dtype=np.float32)
            side_pairs = (
                (quad[0], quad[1]),
                (quad[1], quad[2]),
                (quad[3], quad[2]),
                (quad[0], quad[3]),
            )

            fitted = []
            band = max(4.0, min(rh, rw) * 0.014)
            pad = max(5.0, min(rh, rw) * 0.016)

            for p1, p2 in side_pairs:
                vec = p2 - p1
                side_len = float(np.linalg.norm(vec))
                if side_len < 20:
                    return None

                unit = vec / side_len
                normal = np.array([-unit[1], unit[0]], dtype=np.float32)
                rel = edge_pts - p1
                ts = rel @ unit
                ds = np.abs(rel @ normal)
                near = (ds <= band) & (ts >= -pad) & (ts <= side_len + pad)
                line = fit_side_line(edge_pts[near])
                if line is None:
                    return None
                fitted.append(line)

            top_line, right_line, bottom_line, left_line = fitted
            refined_pts = [
                intersect_fit_lines(top_line, left_line),
                intersect_fit_lines(top_line, right_line),
                intersect_fit_lines(bottom_line, right_line),
                intersect_fit_lines(bottom_line, left_line),
            ]
            if any(pt is None for pt in refined_pts):
                return None

            refined = np.array(refined_pts, dtype=np.float32)
            if np.any(refined[:, 0] < -rw * 0.05) or np.any(refined[:, 0] > rw * 1.05):
                return None
            if np.any(refined[:, 1] < -rh * 0.05) or np.any(refined[:, 1] > rh * 1.05):
                return None

            old_area = abs(cv2.contourArea(quad))
            new_area = abs(cv2.contourArea(refined))
            if old_area <= 1 or new_area < old_area * 0.78 or new_area > old_area * 1.18:
                return None

            return np.array(_order_points(refined), dtype=np.float32)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            bbox_area = cw * ch
            if bbox_area < rw * rh * 0.22:
                continue
            if cw < rw * 0.45 or ch < rh * 0.35:
                continue

            touches_all = (
                x <= 2 and y <= 2 and x + cw >= rw - 3 and y + ch >= rh - 3
            )
            if touches_all:
                continue

            peri = cv2.arcLength(cnt, True)
            quad = None
            for eps in (0.015, 0.025, 0.04, 0.065):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype("float32")
                    break

            if quad is None:
                hull = cv2.convexHull(cnt).reshape(-1, 2).astype("float32")
                if len(hull) >= 4:
                    quad = _quad_from_points(hull)

            if quad is None:
                continue

            tl, tr, br, bl = _order_points(quad)
            area = abs(cv2.contourArea(np.array([tl, tr, br, bl], dtype=np.float32)))
            if area < rw * rh * 0.20 or area > rw * rh * 0.98:
                continue

            top_w = np.linalg.norm(tr - tl)
            bottom_w = np.linalg.norm(br - bl)
            left_h = np.linalg.norm(bl - tl)
            right_h = np.linalg.norm(br - tr)
            avg_w = (top_w + bottom_w) / 2.0
            avg_h = (left_h + right_h) / 2.0
            ratio = avg_w / max(avg_h, 1)
            if ratio < 1.0 or ratio > 3.8:
                continue

            # Prefer a frame-like contour that spans most of the current
            # rectified monitor but does not include outside background.
            coverage = area / max(rw * rh, 1)
            score = area * (1.0 - 0.15 * abs(ratio - ASPECT_RATIO))
            if coverage > 0.90:
                score *= 0.6

            if score > best_score:
                best_score = score
                best_quad = np.array([tl, tr, br, bl], dtype=np.float32)

        if best_quad is None:
            return rectified, None

        edge_refined_quad = refine_quad_edges(best_quad)
        if edge_refined_quad is not None:
            best_quad = edge_refined_quad

        old_area = float(rw * rh)
        new_area = abs(cv2.contourArea(best_quad))
        if new_area > old_area * 0.985:
            return rectified, None

        dst2 = np.array(
            [[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]],
            dtype=np.float32,
        )
        try:
            old_to_new = cv2.getPerspectiveTransform(best_quad, dst2)
            new_to_old = cv2.getPerspectiveTransform(dst2, best_quad)
            refined = cv2.warpPerspective(rectified, old_to_new, (rw, rh))
        except cv2.error:
            return rectified, None

        return refined, new_to_old

    def refine_inner_bezel(rectified):
        if _photo_sign_mode:
            return rectified, None

        rh, rw = rectified.shape[:2]
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        dark = (gray < 55).astype(np.float32)

        x0 = int(rw * 0.12)
        x1 = max(x0 + 1, int(rw * 0.88))
        y0 = int(rh * 0.12)
        y1 = max(y0 + 1, int(rh * 0.88))
        row_score = dark[:, x0:x1].mean(axis=1)
        col_score = dark[y0:y1, :].mean(axis=0)

        def find_inset(scores, limit):
            limit = min(limit, len(scores) // 3)
            if limit <= 2:
                return 0

            border_threshold = 0.38
            content_threshold = 0.24
            if np.max(scores[:max(2, limit // 3)]) < border_threshold:
                return 0

            for idx in range(1, limit):
                local = float(np.mean(scores[idx:min(limit, idx + 3)]))
                if local < content_threshold:
                    return idx
            return 0

        top = find_inset(row_score, int(rh * 0.08))
        bottom = find_inset(row_score[::-1], int(rh * 0.08))
        left = find_inset(col_score, int(rw * 0.08))
        right = find_inset(col_score[::-1], int(rw * 0.08))

        if min(top, bottom, left, right) < 2:
            return rectified, None

        min_inset = max(1, min(top, bottom, left, right))
        max_inset = max(top, bottom, left, right)
        if max_inset > min_inset * 5:
            return rectified, None

        crop_left = float(left)
        crop_top = float(top)
        crop_right = float(rw - 1 - right)
        crop_bottom = float(rh - 1 - bottom)
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        if crop_w < rw * 0.86 or crop_h < rh * 0.82:
            return rectified, None

        inner_ratio = crop_w / max(crop_h, 1.0)
        if inner_ratio <= ASPECT_RATIO * 1.001:
            return rectified, None
        if inner_ratio > ASPECT_RATIO * 1.04:
            return rectified, None

        full_left = crop_left
        full_top = crop_top
        full_right_inset = float(rw - 1) - crop_right
        full_bottom_inset = float(rh - 1) - crop_bottom

        def ratio_at(strength):
            source_w = float(rw - 1) - (full_left + full_right_inset) * strength
            source_h = float(rh - 1) - (full_top + full_bottom_inset) * strength
            return source_w / max(source_h, 1.0)

        # Inner black borders are useful, but their exact inside edge can be
        # off by a few pixels depending on antialiasing and generated artwork.
        # Use the detected inset fully only when it keeps the source ratio near
        # 16:9; otherwise solve for the strongest inset that stays within a
        # small geometry-derived tolerance.
        target_excess = min(0.006, max(0.0025, 2.7 / max(float(rw - 1), 1.0)))
        target_ratio = ASPECT_RATIO * (1.0 + target_excess)

        if inner_ratio <= target_ratio:
            inset_strength = 1.0
        else:
            lo = 0.0
            hi = 1.0
            for _ in range(24):
                mid = (lo + hi) / 2.0
                if ratio_at(mid) <= target_ratio:
                    lo = mid
                else:
                    hi = mid
            inset_strength = lo

        if inset_strength < 0.05:
            return rectified, None

        crop_left = full_left * inset_strength
        crop_top = full_top * inset_strength
        crop_right = float(rw - 1) - full_right_inset * inset_strength
        crop_bottom = float(rh - 1) - full_bottom_inset * inset_strength

        src_crop = np.array(
            [
                [crop_left, crop_top],
                [crop_right, crop_top],
                [crop_right, crop_bottom],
                [crop_left, crop_bottom],
            ],
            dtype=np.float32,
        )
        dst_full = np.array(
            [[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]],
            dtype=np.float32,
        )

        try:
            crop_to_full = cv2.getPerspectiveTransform(src_crop, dst_full)
            full_to_crop = cv2.getPerspectiveTransform(dst_full, src_crop)
            refined = cv2.warpPerspective(rectified, crop_to_full, (rw, rh))
        except cv2.error:
            return rectified, None

        return refined, full_to_crop

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_right = np.linalg.norm(bottom_right - top_right)
    height_left = np.linalg.norm(bottom_left - top_left)

    observed_w = max(width_top, width_bottom, 1.0)
    observed_h = max(height_right, height_left, 1.0)
    avg_observed_w = (width_top + width_bottom) / 2.0
    avg_observed_h = (height_right + height_left) / 2.0

    portrait_monitor = avg_observed_h > avg_observed_w

    # The rule says the long side is 16 and the short side is 9.  If the
    # detected monitor is portrait, rotate the source point order so the
    # long side becomes the horizontal 16 part of the rectified image.
    # Choose a similar output area to the observed quadrilateral, but force
    # the coordinate system used for line detection and angle calculation to
    # the physical monitor aspect ratio.
    target_area = max(observed_w * observed_h, ASPECT_W * ASPECT_H)
    output_scale = max(40, int(round(math.sqrt(target_area / (ASPECT_W * ASPECT_H)))))
    output_scale = min(output_scale, 72)
    output_width = int(ASPECT_W * output_scale) + 1
    output_height = int(ASPECT_H * output_scale) + 1

    if portrait_monitor:
        src = np.array([bottom_left, top_left, top_right, bottom_right], dtype="float32")
    else:
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    dst = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    rectified_to_original = cv2.getPerspectiveTransform(dst, src)
    rectified = cv2.warpPerspective(image, transform, (output_width, output_height))

    inner_refined, inner_to_rectified = refine_inner_bezel(rectified)
    if inner_to_rectified is not None:
        _rectified_to_original = rectified_to_original @ inner_to_rectified
        return inner_refined

    _rectified_to_original = rectified_to_original
    return rectified


def detect_line(rectified):
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    margin_x = max(3, int(w * 0.035))
    margin_y = max(3, int(h * 0.055))
    roi_bottom = h - margin_y

    if _bright_sign_mode:
        margin_x = max(3, int(w * 0.025))
        margin_y = max(3, int(h * 0.045))
        roi_bottom = max(margin_y + 10, int(h * 0.955))

    roi = rectified[margin_y:roi_bottom, margin_x:w - margin_x]
    if roi.size == 0:
        roi = rectified
        margin_x = 0
        margin_y = 0

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    roi_area = roi.shape[0] * roi.shape[1]
    yellow_mask = (
        (hue >= 12) & (hue <= 48) & (saturation > 35) & (value > 80)
    ).astype(np.uint8) * 255
    green_mask = (
        (hue > 48) & (hue < 78) & (saturation > 35) & (value > 75)
    ).astype(np.uint8) * 255
    cyan_mask = (
        (hue >= 72) & (hue <= 108) & (saturation > 22) & (value > 82)
    ).astype(np.uint8) * 255
    blue_mask = (
        (hue > 108) & (hue <= 132) & (saturation > 35) & (value > 70)
    ).astype(np.uint8) * 255
    magenta_mask = (
        (hue >= 133) & (hue <= 168) & (saturation > 35) & (value > 75)
    ).astype(np.uint8) * 255
    red_mask = (
        ((hue <= 8) | (hue >= 169)) & (saturation > 35) & (value > 75)
    ).astype(np.uint8) * 255
    color_mask = ((saturation > 55) & (value > 75)).astype(np.uint8) * 255
    dark_mask = ((roi_gray < 70) & (saturation < 150)).astype(np.uint8) * 255
    bright_mask = ((roi_gray > 185) & (saturation < 90)).astype(np.uint8) * 255
    edge_mask = cv2.Canny(roi_gray, 32, 105)
    local_blur = cv2.GaussianBlur(roi_gray, (0, 0), 3)
    contrast = cv2.absdiff(roi_gray, local_blur)
    contrast_threshold = max(7.0, float(np.mean(contrast) + 1.15 * np.std(contrast)))
    contrast_mask = (contrast > contrast_threshold).astype(np.uint8) * 255
    contrast_soft_threshold = max(
        3.0,
        float(np.mean(contrast) + 0.72 * np.std(contrast)),
        float(np.percentile(contrast, 92)),
    )
    contrast_soft_mask = (contrast > contrast_soft_threshold).astype(np.uint8) * 255
    low_edge_mask = cv2.Canny(roi_gray, 14, 46)

    sat_blur = cv2.GaussianBlur(saturation, (0, 0), 4)
    val_blur = cv2.GaussianBlur(value, (0, 0), 4)
    sat_diff = cv2.absdiff(saturation, sat_blur)
    val_diff = cv2.absdiff(value, val_blur)
    color_residual = cv2.max(sat_diff, val_diff)
    residual_threshold = max(
        4.0,
        float(np.mean(color_residual) + 0.80 * np.std(color_residual)),
        float(np.percentile(color_residual, 93)),
    )
    residual_mask = (color_residual > residual_threshold).astype(np.uint8) * 255

    mask_options = []

    def add_mask(mask, is_color, bias, min_pixels, kind):
        count = np.count_nonzero(mask)
        if count >= min_pixels and count < roi_area * 0.45:
            mask_options.append((mask, is_color, bias, kind))

    yellow_min_pixels = max(8, roi_area * 0.00008)
    color_min_pixels = max(20, roi_area * 0.0007)
    has_color_signal = (
        np.count_nonzero(yellow_mask) >= yellow_min_pixels
        or np.count_nonzero(green_mask) >= yellow_min_pixels
        or np.count_nonzero(cyan_mask) >= yellow_min_pixels
        or np.count_nonzero(blue_mask) >= yellow_min_pixels
        or np.count_nonzero(magenta_mask) >= yellow_min_pixels
        or np.count_nonzero(red_mask) >= yellow_min_pixels
        or np.count_nonzero(color_mask) >= color_min_pixels
    )

    add_mask(cyan_mask, True, 1.10, yellow_min_pixels, "cyan")
    add_mask(blue_mask, True, 1.05, yellow_min_pixels, "blue")
    add_mask(green_mask, True, 1.00, yellow_min_pixels, "green")
    add_mask(yellow_mask, True, 0.92, yellow_min_pixels, "yellow")
    add_mask(magenta_mask, True, 0.96, yellow_min_pixels, "magenta")
    add_mask(red_mask, True, 0.94, yellow_min_pixels, "red")
    if np.count_nonzero(color_mask) >= color_min_pixels and len(mask_options) <= 1:
        add_mask(color_mask, True, 0.72, color_min_pixels, "color")

    add_mask(dark_mask, False, 0.98, max(20, roi_area * 0.0007), "dark")
    add_mask(bright_mask, False, 0.94, max(20, roi_area * 0.0007), "bright")

    color_union = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(yellow_mask, green_mask), cyan_mask), blue_mask)
    color_union = cv2.bitwise_or(cv2.bitwise_or(color_union, magenta_mask), red_mask)
    combined_mask = cv2.bitwise_or(dark_mask, color_union)
    add_mask(combined_mask, False, 0.58, max(20, roi_area * 0.0007), "combined")

    edge_density = np.count_nonzero(edge_mask) / max(roi_area, 1)
    contrast_density = np.count_nonzero(contrast_mask) / max(roi_area, 1)
    low_texture_screen = (
        not _photo_sign_mode
        and edge_density < 0.070
        and contrast_density < 0.090
    )

    if not mask_options or (low_texture_screen and not has_color_signal):
        # Edge/contrast masks help with low-contrast synthetic screens, but on
        # photo-like screens text and logo edges are often longer than the
        # actual answer line.
        add_mask(edge_mask, False, 0.58, max(18, roi_area * 0.0005), "edge")
        add_mask(contrast_mask, False, 0.55, max(18, roi_area * 0.0005), "contrast")

    if low_texture_screen:
        add_mask(contrast_soft_mask, False, 0.64, max(12, roi_area * 0.00035), "contrast_soft")
        add_mask(low_edge_mask, False, 0.54, max(12, roi_area * 0.00035), "low_edge")
        add_mask(residual_mask, True, 0.70, max(12, roi_area * 0.00035), "residual")

    if not mask_options:
        mask_options.append((cv2.Canny(roi_gray, 50, 150), False, 0.50, "fallback"))

    if len(mask_options) > 6:
        mask_options.sort(key=lambda item: item[2], reverse=True)
        mask_options = mask_options[:6]

    best_line = None
    best_score = -1.0
    best_color_line = None
    best_color_score = -1.0
    best_color_kind = None
    k = max(2, int(min(h, w) * 0.004))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    def consider_line(line, score, is_color, kind):
        nonlocal best_line, best_score, best_color_line, best_color_score, best_color_kind
        if is_color and score > best_color_score:
            best_color_score = score
            best_color_line = line
            best_color_kind = kind
        if score > best_score:
            best_score = score
            best_line = line

    for base_mask, is_color_mask, mask_bias, mask_kind in mask_options:
        if is_color_mask:
            # Thin colored answer lines are often only one or two pixels wide
            # after rectification; opening can erase them completely.
            color_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            line_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, color_kernel, iterations=1)
        else:
            line_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask_count = np.count_nonzero(line_mask)
        if mask_count == 0:
            continue

        mask_density = mask_count / max(roi_area, 1)
        density_penalty = 1.0 / (1.0 + max(0.0, mask_density - 0.07) * 4.0)
        score_bias = mask_bias * density_penalty
        if _photo_sign_mode and has_color_signal and not is_color_mask:
            score_bias *= 0.52

        if is_color_mask and _photo_sign_mode:
            min_len = max(10, int(min(h, w) * 0.055))
            max_gap = max(14, int(min(h, w) * 0.12))
            hough_threshold = max(7, int(min(h, w) * 0.016))
        else:
            min_len = max(18, int(min(h, w) * 0.10))
            max_gap = max(5, int(min(h, w) * 0.020))
            hough_threshold = max(14, int(min(h, w) * 0.028))

        mask_ys, mask_xs = np.where(line_mask > 0)
        mask_pts = np.column_stack((mask_xs, mask_ys)).astype(np.float32)
        _, component_labels, _, _ = cv2.connectedComponentsWithStats(line_mask, 8)
        mask_component_labels = component_labels[mask_ys, mask_xs]
        raw_ys, raw_xs = np.where(base_mask > 0)
        raw_mask_pts = np.column_stack((raw_xs, raw_ys)).astype(np.float32)

        def measure_supported_line(x1, y1, x2, y2):
            if len(mask_pts) == 0:
                return None, -1.0

            dx = float(x2 - x1)
            dy = float(y2 - y1)
            base_length = math.hypot(dx, dy)
            if base_length < max(8, min_len * 0.55):
                return None, -1.0

            unit = np.array([dx / base_length, dy / base_length], dtype=np.float32)
            normal = np.array([-unit[1], unit[0]], dtype=np.float32)
            origin = np.array([float(x1), float(y1)], dtype=np.float32)

            rel = mask_pts - origin
            ts = rel @ unit
            ds = np.abs(rel @ normal)

            band = max(2.5, min(h, w) * 0.006)
            pad = max(5.0, min(h, w) * 0.012)
            keep = (ds <= band) & (ts >= -pad) & (ts <= base_length + pad)
            if np.count_nonzero(keep) < max(8, min_len * 0.20):
                return None, -1.0

            kept_t = np.sort(ts[keep])
            if is_color_mask and (_photo_sign_mode or _bright_sign_mode):
                gap_ratio = 0.030
            else:
                gap_ratio = 0.009 if _photo_sign_mode else 0.006
            gap_limit = max(3.0, min(h, w) * gap_ratio)

            best_start = float(kept_t[0])
            best_end = float(kept_t[0])
            best_count = 1
            cur_start = float(kept_t[0])
            cur_end = float(kept_t[0])
            cur_count = 1
            segments = []

            for idx in range(1, len(kept_t)):
                t = float(kept_t[idx])
                if t - cur_end <= gap_limit:
                    cur_end = t
                    cur_count += 1
                else:
                    segments.append((cur_start, cur_end, cur_count))
                    if cur_end - cur_start > best_end - best_start:
                        best_start = cur_start
                        best_end = cur_end
                        best_count = cur_count
                    cur_start = t
                    cur_end = t
                    cur_count = 1

            segments.append((cur_start, cur_end, cur_count))
            if cur_end - cur_start > best_end - best_start:
                best_start = cur_start
                best_end = cur_end
                best_count = cur_count

            color_fragment_bonus = 1.0
            useful_segments = [
                seg for seg in segments
                if seg[1] - seg[0] >= max(3.0, min(h, w) * 0.006)
            ]
            if is_color_mask and len(useful_segments) >= 2:
                full_start = float(kept_t[0])
                full_end = float(kept_t[-1])
                full_span = full_end - full_start
                full_bin = max(2.0, min(h, w) * 0.0055)
                full_bins = np.floor((kept_t - full_start) / full_bin).astype(np.int32)
                full_total = max(1, int(math.ceil(full_span / full_bin)))
                full_continuity = len(np.unique(full_bins)) / max(full_total, 1)

                # A dashed/occluded colored answer line should still project to
                # one long span.  Random colored artwork usually has either too
                # many tiny fragments or very poor projected occupancy.
                if (
                    full_span >= max(float(min_len), (best_end - best_start) * 1.22)
                    and len(useful_segments) <= 9
                    and full_continuity >= 0.075
                ):
                    best_start = full_start
                    best_end = full_end
                    best_count = int(len(kept_t))
                    color_fragment_bonus = 1.0 + min(0.18, 0.035 * (len(useful_segments) - 1))

            supported_length = best_end - best_start
            if supported_length < min_len:
                return None, -1.0

            in_interval = keep & (ts >= best_start - 1.0) & (ts <= best_end + 1.0)
            if np.count_nonzero(in_interval) < max(8, supported_length * 0.18):
                return None, -1.0

            continuity_bin = max(2.0, min(h, w) * 0.004)
            interval_bins = np.floor((ts[in_interval] - best_start) / continuity_bin).astype(np.int32)
            total_bins = max(1, int(math.ceil(supported_length / continuity_bin)))
            occupied_bins = len(np.unique(interval_bins))
            continuity = occupied_bins / max(total_bins, 1)
            min_continuity = 0.10 if is_color_mask and (_photo_sign_mode or _bright_sign_mode) else 0.18
            if continuity < min_continuity:
                return None, -1.0

            labels_in = mask_component_labels[in_interval]
            labels_in = labels_in[labels_in > 0]
            component_purity = 1.0
            component_span_ratio = 1.0
            if len(labels_in) > 0:
                counts = np.bincount(labels_in.astype(np.int32))
                if len(counts) > 1:
                    dominant_label = int(np.argmax(counts[1:]) + 1)
                    component_purity = float(counts[dominant_label]) / max(float(len(labels_in)), 1.0)
                    dominant_interval = in_interval & (mask_component_labels == dominant_label)
                    if np.count_nonzero(dominant_interval) >= 2:
                        dominant_ts = ts[dominant_interval]
                        component_span_ratio = float(dominant_ts.max() - dominant_ts.min()) / max(supported_length, 1.0)

            # Text and repeated small strokes can align along one Hough line,
            # but they do not belong to one continuous component.  A real
            # solid line should usually have one dominant connected support.
            if is_color_mask:
                if component_purity < 0.22 and component_span_ratio < 0.30:
                    return None, -1.0
            elif component_purity < 0.34 and component_span_ratio < 0.42:
                return None, -1.0

            wide_band = band * 3.2
            near_interval = (
                (np.abs(ds) <= wide_band)
                & (ts >= best_start - 1.0)
                & (ts <= best_end + 1.0)
            )
            clutter_count = int(np.count_nonzero(near_interval & ~in_interval))
            line_count = max(int(np.count_nonzero(in_interval)), 1)
            clutter_ratio = clutter_count / line_count

            thickness = float(np.percentile(ds[in_interval], 90) * 2.0 + 1.0)
            if thickness > max(11, min(h, w) * 0.032):
                return None, -1.0

            fit_pts = mask_pts[in_interval]
            if len(raw_mask_pts) > 0:
                raw_rel = raw_mask_pts - origin
                raw_ts = raw_rel @ unit
                raw_ds = np.abs(raw_rel @ normal)
                raw_keep = (
                    (raw_ds <= max(band * 0.75, thickness * 0.65))
                    & (raw_ts >= best_start - 1.5)
                    & (raw_ts <= best_end + 1.5)
                )
                if np.count_nonzero(raw_keep) >= max(8, supported_length * 0.12):
                    fit_pts = raw_mask_pts[raw_keep]

            def centerline_points(points):
                rel_points = points - origin
                point_ts = rel_points @ unit
                point_ds = rel_points @ normal
                bin_width = max(1.5, min(h, w) * 0.0035)
                bin_ids = np.floor((point_ts - best_start) / bin_width).astype(np.int32)
                centers = []

                for bin_id in np.unique(bin_ids):
                    in_bin = bin_ids == bin_id
                    if np.count_nonzero(in_bin) < 2:
                        continue

                    bin_t = point_ts[in_bin]
                    bin_d = point_ds[in_bin]
                    center_t = float(np.median(bin_t))
                    center_d = float(np.median(bin_d))
                    centers.append(origin + unit * center_t + normal * center_d)

                if len(centers) < max(8, int(supported_length / max(bin_width * 5.0, 1.0))):
                    return None

                return np.array(centers, dtype=np.float32)

            center_pts = centerline_points(fit_pts)
            if center_pts is not None:
                fit_pts = center_pts

            try:
                vx, vy, x0, y0 = cv2.fitLine(
                    fit_pts.astype(np.float32),
                    cv2.DIST_L2,
                    0,
                    0.01,
                    0.01,
                ).reshape(-1)
            except cv2.error:
                return None, -1.0

            fit_unit = np.array([float(vx), float(vy)], dtype=np.float32)
            fit_norm = float(np.linalg.norm(fit_unit))
            if fit_norm < 1e-6:
                return None, -1.0
            fit_unit /= fit_norm
            if float(np.dot(fit_unit, unit)) < 0:
                fit_unit = -fit_unit

            fit_origin = np.array([float(x0), float(y0)], dtype=np.float32)
            fit_ts = (fit_pts - fit_origin) @ fit_unit
            refit_length = float(fit_ts.max() - fit_ts.min())
            if refit_length < min_len:
                return None, -1.0

            if len(fit_pts) >= 12:
                order = np.argsort(fit_ts)
                end_count = max(4, min(len(order) // 4, int(len(order) * 0.16)))
                start_center = np.median(fit_pts[order[:end_count]], axis=0)
                end_center = np.median(fit_pts[order[-end_count:]], axis=0)
                end_vec = end_center - start_center
                end_norm = float(np.linalg.norm(end_vec))

                if end_norm >= min_len * 0.65:
                    end_unit = (end_vec / end_norm).astype(np.float32)
                    if float(np.dot(end_unit, fit_unit)) < 0:
                        end_unit = -end_unit

                    dot = float(np.clip(np.dot(end_unit, fit_unit), -1.0, 1.0))
                    diff_angle = math.degrees(math.acos(abs(dot)))
                    if diff_angle <= 2.5:
                        mixed_unit = fit_unit * 0.45 + end_unit * 0.55
                        mixed_norm = float(np.linalg.norm(mixed_unit))
                        if mixed_norm > 1e-6:
                            fit_unit = mixed_unit / mixed_norm
                            fit_origin = np.median(fit_pts, axis=0).astype(np.float32)
                            fit_ts = (fit_pts - fit_origin) @ fit_unit
                            refit_length = float(fit_ts.max() - fit_ts.min())

            density = best_count / max(supported_length, 1.0)
            score = refit_length * (0.85 + 0.10 * min(density, 3.0))
            score *= min(2.15, max(1.0, (refit_length / max(float(min_len), 1.0)) ** 0.42))
            score /= 1.0 + 0.085 * max(0.0, thickness - 4.0)
            if continuity < 0.42:
                score *= 0.35 + continuity
            elif continuity > 0.72:
                score *= 1.05
            if clutter_ratio > 2.4:
                score *= 0.42
            elif clutter_ratio > 1.25:
                score *= 0.72
            component_quality = max(component_purity, component_span_ratio)
            if component_quality < 0.52:
                score *= 0.35 + 0.85 * component_quality
            elif component_quality > 0.82:
                score *= 1.08

            p1 = fit_origin + fit_unit * fit_ts.min()
            p2 = fit_origin + fit_unit * fit_ts.max()
            edge_margin_x = max(4.0, roi.shape[1] * 0.040)
            edge_margin_y = max(4.0, roi.shape[0] * 0.045)
            if _photo_sign_mode or _bright_sign_mode:
                edge_margin_x = max(edge_margin_x, roi.shape[1] * 0.055)
                edge_margin_y = max(edge_margin_y, roi.shape[0] * 0.070)
            near_left = max(p1[0], p2[0]) < edge_margin_x
            near_right = min(p1[0], p2[0]) > roi.shape[1] - edge_margin_x
            near_top = max(p1[1], p2[1]) < edge_margin_y
            near_bottom = min(p1[1], p2[1]) > roi.shape[0] - edge_margin_y
            if near_left or near_right or near_top or near_bottom:
                if not is_color_mask and refit_length > max(roi.shape[0], roi.shape[1]) * 0.55:
                    return None, -1.0
                score *= 0.22 if is_color_mask else 0.055
            if is_color_mask:
                if mask_kind == "cyan":
                    score *= 1.14
                elif mask_kind == "blue":
                    score *= 1.10
                elif mask_kind == "yellow" and _photo_sign_mode:
                    score *= 0.88
                else:
                    score *= 1.04
                score *= color_fragment_bonus
            score *= score_bias

            line = np.array(
                [
                    float(p1[0] + margin_x),
                    float(p1[1] + margin_y),
                    float(p2[0] + margin_x),
                    float(p2[1] + margin_y),
                ],
                dtype=np.float32,
            )

            return line, score

        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < max(6, min(h, w) * 0.003):
                continue

            _, _, cw, ch = cv2.boundingRect(cnt)
            box_long = max(cw, ch)
            box_short = max(1, min(cw, ch))
            if box_long < min_len or box_long / box_short < 2.0:
                continue

            pts = cnt.reshape(-1, 2).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
            theta = math.atan2(float(vy), float(vx))
            unit = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            origin = np.array([float(x0), float(y0)], dtype=np.float32)
            ts = (pts - origin) @ unit

            p1 = origin + unit * ts.min()
            p2 = origin + unit * ts.max()
            candidate_line, score = measure_supported_line(p1[0], p1[1], p2[0], p2[1])
            if candidate_line is not None:
                consider_line(candidate_line, score, is_color_mask, mask_kind)

        edges = cv2.Canny(line_mask, 40, 120)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        if lines is None:
            continue

        if len(lines) > 80:
            raw_lines = lines[:, 0, :]
            lengths = np.hypot(
                raw_lines[:, 2].astype(np.float32) - raw_lines[:, 0].astype(np.float32),
                raw_lines[:, 3].astype(np.float32) - raw_lines[:, 1].astype(np.float32),
            )
            keep_idx = np.argsort(lengths)[-80:]
            lines = lines[keep_idx]

        for raw_line in lines[:, 0, :]:
            x1, y1, x2, y2 = raw_line.astype(float)
            if math.hypot(x2 - x1, y2 - y1) < min_len:
                continue

            candidate_line, score = measure_supported_line(x1, y1, x2, y2)
            if candidate_line is not None:
                consider_line(candidate_line, score, is_color_mask, mask_kind)

    if best_color_line is not None and has_color_signal:
        if best_color_kind in ("cyan", "blue"):
            color_keep_ratio = 0.44 if (_photo_sign_mode or _bright_sign_mode) else 0.58
        elif best_color_kind == "yellow":
            color_keep_ratio = 0.78 if (_photo_sign_mode or _bright_sign_mode) else 0.66
        else:
            color_keep_ratio = 0.58 if (_photo_sign_mode or _bright_sign_mode) else 0.64
        if best_line is None or best_color_score >= best_score * color_keep_ratio:
            return best_color_line

    return best_line


def calculate_angle(line):
    if line is None:
        return 0.0
    return _angle_to_vertical(line)


def viz_result(image, top_left, top_right, bottom_right, bottom_left, rectified, line):
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    vis_monitor = image.copy()
    vis_rectified = image.copy() if rectified is None else rectified
    vis_line = vis_rectified.copy()

    if top_left is not None or top_right is not None or bottom_right is not None or bottom_left is not None:
        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        cv2.polylines(vis_monitor, [pts], isClosed=True, color=(255, 0, 0), thickness=3)

        corner_names = ["TL", "TR", "BR", "BL"]
        for pt, name in zip(pts, corner_names):
            x, y = pt
            cv2.circle(vis_monitor, (x, y), 6, (255, 0, 0), -1)
            cv2.putText(
                vis_monitor,
                name,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    if line is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in line]
        cv2.line(vis_line, (x1, y1), (x2, y2), (0, 0, 255), 3)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(vis_monitor)
    plt.title("Detected Monitor")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(vis_rectified)
    plt.title("Rectified Monitor")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(vis_line)
    plt.title("Detected Line")
    plt.axis("off")

    plt.tight_layout()
    plt_show()
