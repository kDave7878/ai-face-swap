# -*- coding: utf-8 -*-
"""
面部替换（换脸 / 深度伪造）—— 不覆盖旧结果的稳健版

用法：
    py face_swap_fixed.py <底图> <提供人脸的图>
    py face_swap_fixed.py person1.png person2.png
    py face_swap_fixed.py person1.png person2.png 我的结果.jpg

与原脚本的区别：
1. 输出文件名自动带时间戳（如 output_faceswap_20261010_152503.jpg），不再覆盖旧图
2. 也可以自己指定输出文件名（第三个参数）
3. 保存失败会如实报错（原脚本 imwrite 失败也打印“✓”）
4. 自动切到脚本所在目录；dlib 用相对文件名打开模型（避免中文绝对路径报错）
5. ★ 修复黑块：warp 时把 float64 蒙版转为 float32（OpenCV 对 CV_64F 处理
   WARP_INVERSE_MAP 的方向与 CV_8U/CV_32F 相反，会让蒙版落到错误位置）
6. 加强保险：蒙版乘上"有效覆盖区"，越界像素不参与合成
7. 修复噪点：合成结果先 clip 到 [0,255] 再转 uint8，避免溢出环绕
"""

import os
import sys
import datetime

import cv2
import dlib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DAT_NAME = "shape_predictor_68_face_landmarks.dat"   # dlib 只能吃相对名（中文路径会挂）
DAT_ABS = os.path.join(SCRIPT_DIR, DAT_NAME)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]


def read_im_and_landmarks(fname):
    im = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"无法读取图像: {fname}")
    im = cv2.resize(im, dsize=(int(im.shape[1] * SCALE_FACTOR), int(im.shape[0] * SCALE_FACTOR)))
    return im, get_landmarks(im)


def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces("检测到多个人脸，换一张单人照")
    if len(rects) == 0:
        raise NoFaces("没有检测到人脸")
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()], dtype=np.float64)


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 = points1 - c1
    points2 = points2 - c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 = points1 / s1
    points2 = points2 / s2
    U, S, Vt = np.linalg.svd(points1.T @ points2)
    R = (U @ Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, (c2.T - (s2 / s1) * R @ c1.T).reshape(2, 1))),
                      np.array([[0., 0., 1.]])])


def warp_im(im, M, dshape):
    """
    对齐用 warp。

    ★ 真正的坑在这里：OpenCV 的 cv2.warpAffine 对 CV_64F(float64) 输入，
    WARP_INVERSE_MAP 的作用方向与 CV_8U / CV_32F **相反**。
    蒙版(get_face_mask)是 float64，直接 warp 会落到完全错误的位置，
    于是 combined_mask 覆盖到源图范围之外，合成结果就出现大片黑块
    （换 REPLICATE 只是把黑块换成模糊/高亮的 smear，治标不治本）。

    解决：float64 先转 float32 走同一套语义，warp 完再转回来。
    """
    is_f64 = (im.dtype == np.float64)
    work = im.astype(np.float32) if is_f64 else im
    output_im = np.zeros(dshape, dtype=work.dtype)
    cv2.warpAffine(work, M[:2], (dshape[1], dshape[0]), dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output_im.astype(np.float64) if is_f64 else output_im


def get_face_mask(im, landmarks):
    mask = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        hull = cv2.convexHull(np.int32(landmarks[group]))
        cv2.fillConvexPoly(mask, hull, color=1)
    mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    mask = (cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    return cv2.GaussianBlur(mask, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur = im2_blur.astype(np.float64)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur)


def imwrite_unicode(path, img):
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1] or ".jpg"
    try:
        if cv2.imwrite(path, img):
            return True, path
    except Exception as e:
        last = f"imwrite 异常: {e}"
    else:
        last = "imwrite 返回 False（权限/占用/路径问题）"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True, path
            last += "；imencode 写后文件为空"
        else:
            last += "；imencode 编码失败"
    except Exception as e:
        last += f"；imencode 异常: {e}"
    return False, last


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


detector = dlib.get_frontal_face_detector()
predictor = None


def main():
    global predictor

    if not os.path.exists(DAT_ABS):
        print("✗ 找不到 shape_predictor_68_face_landmarks.dat，无法继续。")
        return 2
    if len(sys.argv) < 3:
        print("用法: py face_swap_fixed.py <底图> <提供人脸的图> [输出文件名]")
        imgs = [f for f in os.listdir(".") if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        print("\n当前目录可用图片:")
        for f in imgs:
            print(f"  - {f}")
        return 1

    src1, src2 = sys.argv[1], sys.argv[2]
    for f in (src1, src2):
        if not os.path.exists(f):
            print(f"✗ 找不到图片: {os.path.abspath(f)}")
            return 3

    if len(sys.argv) > 3:
        out_path = sys.argv[3]
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output_faceswap_{ts}.jpg"

    try:
        predictor = dlib.shape_predictor(DAT_NAME)
    except RuntimeError as e:
        print(f"✗ dlib 打开模型失败: {e}")
        return 7

    try:
        print(f"底图   : {src1}")
        im1, landmarks1 = read_im_and_landmarks(src1)
        print(f"人脸源 : {src2}")
        im2, landmarks2 = read_im_and_landmarks(src2)
    except (TooManyFaces, NoFaces, FileNotFoundError) as e:
        print(f"✗ 错误: {e}")
        return 4

    print("正在合成...")
    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
    # 保险栓：把"采样落到源图之外"的像素从蒙版里剔除，
    # 这样即使出现越界也不会被合成进结果（不会再有黑块/糊块）。
    cover = warp_im(np.full(im2.shape, 255, np.uint8), M, im1.shape)[:, :, 0] > 0
    combined_mask = combined_mask * cover[:, :, None]
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    # 颜色校正用的除法可能把值推到 255 以上，不裁剪直接转 uint8 会发生
    # 环绕（如 360 -> 104），表现为散落的暗点/亮点噪点。
    output_im = np.clip(output_im, 0, 255).astype(np.uint8)

    ok, info = imwrite_unicode(out_path, output_im)
    if ok:
        print(f"\n✓ 已保存 : {info}  ({os.path.getsize(info)} 字节)")
        return 0
    print(f"\n✗ 保存失败 : {info}")
    print("  排查方向：文件被占用、目录只读、杀毒软件/受控文件夹访问")
    return 6


if __name__ == "__main__":
    sys.exit(main())
