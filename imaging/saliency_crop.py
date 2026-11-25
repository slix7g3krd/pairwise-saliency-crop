"""显著性自动裁剪核心模块。

针对六类问题的修复：
1. inverse_map_coords  —— 缩略图坐标反映射回原图时，除缩放外补上缩略图的裁切偏移；
2. keep_user_result    —— 检测异步返回时，若用户已拖动裁剪框则保留用户结果不覆盖；
3. subject_after_ratio —— 比例约束只扩不缩、先与主体求并集、越界平移钳回，保证主体留在框内；
4. cache_by_image      —— 显著性缓存按图片 ID 隔离，批量导出互不覆盖；
5. align_pixels        —— 裁剪框对齐整像素（起点 floor、终点 ceil），消除半像素；
6. trim_black_edge     —— 裁掉缩放后落在边上的黑边。
"""

import math

__all__ = [
    "inverse_map_coords", "keep_user_result", "subject_after_ratio",
    "cache_by_image", "align_pixels", "trim_black_edge",
    "detect", "crop", "export", "preview", "validate_crop",
    "metrics", "audit", "fallback", "determinism",
]


def _norm_box(box):
    """把多种形式的框统一为 (x, y, w, h) 浮点元组。"""
    if isinstance(box, dict):
        return (
            float(box.get("x", 0.0)), float(box.get("y", 0.0)),
            float(box.get("w", 1.0)), float(box.get("h", 1.0)),
        )
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return tuple(float(v) for v in box)
    if isinstance(box, (int, float)):
        return (0.0, 0.0, float(box), float(box))
    return (0.0, 0.0, 1.0, 1.0)


def _is_image(pixels):
    return (
        isinstance(pixels, (list, tuple))
        and len(pixels) > 0
        and isinstance(pixels[0], (list, tuple))
    )


def _luma(px):
    if isinstance(px, (list, tuple)):
        return sum(float(c) for c in px[:3]) / max(len(px[:3]), 1)
    return float(px)


# 1. 坐标反映射：缩略图 -> 原图，必须补上缩略图相对原图的裁切偏移。
def inverse_map_coords(box, mapping=None):
    """把缩略图坐标系下的框映射回原图坐标系。

    mapping = {"scale": s, "offset_x": ox, "offset_y": oy}，含义：
    缩略图 = 原图先按 1/s 缩放、再裁掉左上角 (ox, oy) 偏移得到。
    因此 原图坐标 = 缩略图坐标 * s + (ox, oy)，漏掉偏移会把框偏到主体外。
    """
    m = {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}
    if isinstance(mapping, dict):
        m.update({k: float(mapping[k]) for k in m if k in mapping})
    x, y, w, h = _norm_box(box)
    s = m["scale"]
    return (
        x * s + m["offset_x"],
        y * s + m["offset_y"],
        w * s,
        h * s,
    )


# 2. 异步结果不覆盖：检测返回时用户已拖过框，则保留用户框。
def keep_user_result(state=None):
    """根据状态决定采用用户框还是自动框。

    state = {"user_modified": bool, "user_box": box, "auto_box": box}。
    检测异步返回时若 user_modified 为真，丢弃迟到的 auto_box。
    """
    if not isinstance(state, dict):
        state = {}
    user_box = state.get("user_box")
    auto_box = state.get("auto_box")
    if state.get("user_modified") and user_box is not None:
        return _norm_box(user_box)
    if auto_box is not None:
        return _norm_box(auto_box)
    if user_box is not None:
        return _norm_box(user_box)
    return (0.0, 0.0, 1.0, 1.0)


# 3. 约束后保留主体：比例约束只向外扩，先并上主体，越界平移而非压扁钳回中心。
def subject_after_ratio(box, subject=None, ratio=None, image_size=None):
    """在目标宽高比约束下调整框，保证主体仍在框内。

    - 先把框与主体框取并集，避免比例约束把主体挤到边上；
    - 比例调整只扩边不缩边（缩边会压扁框使主体贴边）；
    - 越界时整体平移钳回图像范围，而不是重新居中导致主体被裁。
    """
    x, y, w, h = _norm_box(box)
    if subject is not None:
        sx, sy, sw, sh = _norm_box(subject)
        x0, y0 = min(x, sx), min(y, sy)
        x1, y1 = max(x + w, sx + sw), max(y + h, sy + sh)
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
    if ratio:
        ratio = float(ratio)
        if h <= 0:
            h = w / ratio
        if w / h < ratio:
            new_w = h * ratio
            x -= (new_w - w) / 2.0
            w = new_w
        else:
            new_h = w / ratio
            y -= (new_h - h) / 2.0
            h = new_h
    if image_size is not None:
        iw, ih = float(image_size[0]), float(image_size[1])
        w = min(w, iw)
        h = min(h, ih)
        x = min(max(x, 0.0), iw - w)
        y = min(max(y, 0.0), ih - h)
    return (x, y, w, h)


# 4. 缓存按图 ID：每张图的显著性结果独立存取。
class SaliencyCache:
    """按图片 ID 隔离的显著性缓存，批量导出时后一张不会覆盖前一张。"""

    def __init__(self):
        self._store = {}

    def get(self, image_id):
        return self._store.get(image_id)

    def set(self, image_id, saliency):
        self._store[image_id] = saliency
        return saliency

    def clear(self):
        self._store.clear()


_CACHE = SaliencyCache()


def cache_by_image(image_id, saliency=None, cache=None):
    """按图片 ID 读/写显著性缓存。未命中时返回空 dict（不为 None）。"""
    store = cache if cache is not None else _CACHE
    if saliency is None:
        return store.get(image_id) or {}
    return store.set(image_id, saliency)


# 5. 框对齐像素：起点向下取整、终点向上取整，避免边框落在半像素上。
def align_pixels(box):
    """把框对齐到整像素网格，返回 (x, y, w, h) 整数元组。"""
    x, y, w, h = _norm_box(box)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = int(math.ceil(x + w))
    y1 = int(math.ceil(y + h))
    return (x0, y0, x1 - x0, y1 - y0)


# 6. 黑边裁掉：返回四边需要裁掉的像素数 (left, top, right, bottom)。
def trim_black_edge(pixels, threshold=0):
    """统计图像四边纯黑（<= threshold）的边框宽度，用于裁掉缩放产生的黑边。"""
    if not _is_image(pixels):
        return (0, 0, 0, 0)
    height = len(pixels)
    width = len(pixels[0])

    def black(px):
        return _luma(px) <= threshold

    top = 0
    while top < height and all(black(p) for p in pixels[top]):
        top += 1
    bottom = 0
    while bottom < height - top and all(black(p) for p in pixels[height - 1 - bottom]):
        bottom += 1
    left = 0
    while left < width and all(black(pixels[r][left]) for r in range(top, height - bottom)):
        left += 1
    right = 0
    while right < width - left and all(
        black(pixels[r][width - 1 - right]) for r in range(top, height - bottom)
    ):
        right += 1
    return (left, top, right, bottom)


def _detect_impl(image):
    """在（缩略图）像素上做简易显著性检测：高于均值的像素质心包围盒。"""
    if not _is_image(image):
        return {"box": (0.0, 0.0, 1.0, 1.0), "score": 0.0}
    height = len(image)
    width = len(image[0])
    values = [_luma(px) for row in image for px in row]
    mean = sum(values) / len(values)
    xs, ys = [], []
    for r, row in enumerate(image):
        for c, px in enumerate(row):
            if _luma(px) > mean:
                xs.append(c)
                ys.append(r)
    if not xs:
        fb = fallback((width, height))
        return {"box": fb, "score": 0.0}
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {
        "box": (float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)),
        "score": 1.0,
    }


def detect(image, image_id=None, mapping=None):
    """显著性检测。结果按 image_id 缓存；给定 mapping 时把框反映射回原图坐标。"""
    if image_id is not None:
        cached = cache_by_image(image_id)
        if cached:
            return cached
    result = _detect_impl(image)
    if mapping is not None:
        result = dict(result)
        result["box"] = inverse_map_coords(result["box"], mapping)
    if image_id is not None:
        cache_by_image(image_id, result)
    return result


def crop(image, box=None):
    """按框裁剪。框先对齐整像素；输入为二维像素列表时返回裁剪后的像素。"""
    if box is None:
        box = detect(image)["box"] if _is_image(image) else (0.0, 0.0, 1.0, 1.0)
    x, y, w, h = align_pixels(box)
    if _is_image(image):
        return [list(row[x:x + w]) for row in image[y:y + h]]
    return (x, y, w, h)


def export(items):
    """批量导出：每张图用自己的 ID 取缓存，显著性图互不覆盖。"""
    if not isinstance(items, (list, tuple)):
        return []
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_id = item.get("id")
        image = item.get("image")
        det = detect(image, image_id=image_id, mapping=item.get("mapping"))
        box = align_pixels(det["box"])
        results.append({"id": image_id, "box": box})
    return results


def preview(image, box=None):
    """生成预览信息：对齐后的整像素框。"""
    if box is None:
        box = detect(image)["box"]
    return {"box": align_pixels(box)}


def validate_crop(box, image_size=None):
    """校验裁剪框：尺寸为正、不越界。返回 {"ok": bool, "errors": [...]}。"""
    x, y, w, h = _norm_box(box)
    errors = []
    if w <= 0 or h <= 0:
        errors.append("non-positive size")
    if x < 0 or y < 0:
        errors.append("negative origin")
    if image_size is not None:
        iw, ih = float(image_size[0]), float(image_size[1])
        if x + w > iw or y + h > ih:
            errors.append("out of bounds")
    return {"ok": not errors, "errors": errors, "box": (x, y, w, h)}


def metrics(box, reference=None):
    """计算裁剪指标；给定参考框时返回 IoU。"""
    x, y, w, h = _norm_box(box)
    if reference is None:
        return {"iou": 1.0, "box": (x, y, w, h)}
    rx, ry, rw, rh = _norm_box(reference)
    ix0, iy0 = max(x, rx), max(y, ry)
    ix1, iy1 = min(x + w, rx + rw), min(y + h, ry + rh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = w * h + rw * rh - inter
    iou = inter / union if union > 0 else 0.0
    return {"iou": iou, "box": (x, y, w, h)}


def audit(record=None):
    """审计：列出本次裁剪管线中生效的修复项。"""
    fixes = [
        "inverse_map_coords", "keep_user_result", "subject_after_ratio",
        "cache_by_image", "align_pixels", "trim_black_edge",
    ]
    report = {"fixes_applied": fixes, "ok": True}
    if isinstance(record, dict):
        report["record"] = record
    return report


def fallback(image_size=None):
    """检测失败时的兜底框：居中、占图 80%。"""
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        iw, ih = float(image_size[0]), float(image_size[1])
    else:
        iw = ih = 1.0
    w, h = iw * 0.8, ih * 0.8
    return ((iw - w) / 2.0, (ih - h) / 2.0, w, h)


def determinism(image=None, runs=2):
    """确定性校验：同一输入多次检测结果必须完全一致。"""
    runs = max(int(runs), 2)
    results = [_detect_impl(image) for _ in range(runs)]
    return {"deterministic": all(r == results[0] for r in results), "runs": runs}
