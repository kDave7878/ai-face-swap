# -*- coding: utf-8 -*-
"""
三版换脸效果对照脚本
把同一对图片分别用三种 warp 策略跑一遍，直接把区别看出来：
    v1 原版        : float64 蒙版 + BORDER_TRANSPARENT   -> 黑块
    v2 上轮补丁    : float64 蒙版 + BORDER_REPLICATE     -> 黑块变模糊/高亮 smear
    v3 真正修复    : float32 蒙版 + BORDER_TRANSPARENT + 覆盖保险栓  -> 正常

用法（放在和 shape_predictor_68_face_landmarks.dat 同一目录下运行）：
    py 对比脚本_三版效果.py 底图.jpg 人脸源.jpg
    py 对比脚本_三版效果.py            # 不带参数则尝试用目录里的 person2.png / person1.png

输出：同目录下 v1_orig.jpg / v2_patch.jpg / v3_real.jpg 与 对比_三版效果.jpg
"""

import os
import sys

import cv2
import dlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DAT_NAME = "shape_predictor_68_face_landmarks.dat"   # dlib 只能用相对名（中文绝对路径会打不开）

FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT_NAME)


def imread_unicode(path):
    """cv2.imread 读不了中文路径，用 imdecode 兜底"""
    im = None
    try:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        pass
    if im is None:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return im


def landmarks(im):
    rects = detector(im, 1)
    if len(rects) != 1:
        raise RuntimeError("需要恰好 1 张脸，实际检测到 %d 张" % len(rects))
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()], dtype=np.float64)


def transformation_from_points(points1, points2):
    """返回把 points1 映射到 points2 的仿射矩阵"""
    p1 = points1.astype(np.float64)
    p2 = points2.astype(np.float64)
    c1, c2 = p1.mean(0), p2.mean(0)
    p1, p2 = p1 - c1, p2 - c2
    s1, s2 = np.std(p1), np.std(p2)
    p1, p2 = p1 / s1, p2 / s2
    U, S, Vt = np.linalg.svd(p1.T @ p2)
    R = (U @ Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, (c2.T - (s2 / s1) * R @ c1.T).reshape(2, 1))),
                      np.array([[0., 0., 1.]])])


def build_mask(lm, h, w):
    m = np.zeros((h, w), np.float64)
    for group in OVERLAY_POINTS:
        cv2.fillConvexPoly(m, cv2.convexHull(np.int32(lm[group]).reshape(-1, 1, 2)), 1)
    m = np.dstack([m, m, m])
    m = (cv2.GaussianBlur(m, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    return cv2.GaussianBlur(m, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)


def warp(im, M, dshape, border, strict_float32):
    """strict_float32=True 即 v3 的做法：float64 先转 float32"""
    is_f64 = (im.dtype == np.float64)
    work = im.astype(np.float32) if (is_f64 and strict_float32) else im
    out = np.zeros(dshape, dtype=work.dtype)
    cv2.warpAffine(work, M[:2], (dshape[1], dshape[0]), dst=out,
                   borderMode=border, flags=cv2.WARP_INVERSE_MAP)
    return out.astype(np.float64) if is_f64 else out


def run(im1, im2, lm1, lm2, border, strict_float32, label, out_name):
    M = transformation_from_points(lm1[ALIGN_POINTS], lm2[ALIGN_POINTS])

    warped_mask = warp(build_mask(lm2, *im2.shape[:2]), M, im1.shape, border, strict_float32)
    combined = np.max([build_mask(lm1, *im1.shape[:2]), warped_mask], axis=0)

    if strict_float32:
        # 保险栓：把"采样落到源图之外"的像素从蒙版里剔除
        cover = warp(np.full(im2.shape, 255, np.uint8), M, im1.shape,
                     cv2.BORDER_CONSTANT, True)[:, :, 0] > 0
        combined = combined * cover[:, :, None]

    warped_im2 = warp(im2, M, im1.shape, border, strict_float32)

    blur = int(COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        lm1[LEFT_EYE_POINTS].mean(0) - lm1[RIGHT_EYE_POINTS].mean(0)))
    blur += (blur % 2 == 0)
    im1_blur = cv2.GaussianBlur(im1, (blur, blur), 0)
    im2_blur = cv2.GaussianBlur(warped_im2, (blur, blur), 0).astype(np.float64)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    corrected = warped_im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur

    out = np.clip(im1 * (1.0 - combined) + corrected * combined, 0, 255).astype(np.uint8)

    core = combined[:, :, 0] > 0.25
    g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    g_in = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    introduced = ((g_in >= 40) & (g_out < 25) & core).sum()
    print("%-26s 融合区 %6d px   引入黑块 %6d px" % (label, core.sum(), introduced))

    cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(out_name)
    return out


def main():
    if not os.path.exists(DAT_NAME):
        print("✗ 找不到 %s，请先放到本脚本同目录" % DAT_NAME)
        return 2

    if len(sys.argv) >= 3:
        base_path, face_path = sys.argv[1], sys.argv[2]
    else:
        base_path, face_path = "person2.png", "person1.png"

    im1 = imread_unicode(base_path)
    im2 = imread_unicode(face_path)
    if im1 is None or im2 is None:
        print("✗ 读不到图片: %s / %s" % (base_path, face_path))
        return 3

    print("底图   : %s  %dx%d" % (base_path, im1.shape[1], im1.shape[0]))
    print("人脸源 : %s  %dx%d" % (face_path, im2.shape[1], im2.shape[0]))
    print()

    lm1, lm2 = landmarks(im1), landmarks(im2)

    a = run(im1, im2, lm1, lm2, cv2.BORDER_TRANSPARENT, False,
            "v1 原版 float64+TRANSPARENT", "v1_orig.jpg")
    b = run(im1, im2, lm1, lm2, cv2.BORDER_REPLICATE, False,
            "v2 补丁 float64+REPLICATE", "v2_patch.jpg")
    c = run(im1, im2, lm1, lm2, cv2.BORDER_TRANSPARENT, True,
            "v3 真修复 float32+覆盖保险", "v3_real.jpg")

    h = min(a.shape[0], b.shape[0], c.shape[0])
    w = min(a.shape[1], b.shape[1], c.shape[1])
    gap = np.full((h, 10, 3), 255, np.uint8)
    row = np.hstack([a[:h, :w], gap, b[:h, :w], gap, c[:h, :w]])
    bar = np.full((42, row.shape[1], 3), 255, np.uint8)
    canvas = np.vstack([bar, row])
    cv2.putText(canvas, "V1 ORIGINAL", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    cv2.putText(canvas, "V2 PATCH", (w + 24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 200), 2)
    cv2.putText(canvas, "V3 REAL FIX", (2 * w + 34, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 0), 2)
    cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile("对比_三版效果.jpg")
    print("\n已生成: v1_orig.jpg / v2_patch.jpg / v3_real.jpg / 对比_三版效果.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
