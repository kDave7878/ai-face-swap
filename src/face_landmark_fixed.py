# -*- coding: utf-8 -*-
"""
人脸特征点检测（稳健版 / 带真实报错）

用法（在任意目录都可以，脚本会自己切到所在目录找模型）：
    py face_landmark_fixed.py person1.png
    py face_landmark_fixed.py person2.png out_mark2.jpg

与原脚本的区别：
1. 不再依赖“当前工作目录”，自动定位 shape_predictor_68_face_landmarks.dat
2. 保存时检查真实结果；cv2.imwrite 失败会自动改用 imencode+tofile（兼容中文路径）
3. 失败时打印真实原因（而不是假装成功）
4. 输出文件名自动带时间戳，不再覆盖上一次的结果
"""

import os
import sys
import datetime

import cv2
import dlib
import numpy as np

# 关键：把工作目录切到脚本所在目录，保证能找到 .dat 模型
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# 注意：dlib 是 C++ 实现，内部用窄字符 fopen，
# 传含中文的“绝对路径”会报 Unable to open ...
# 所以这里刻意使用“不含中文的相对文件名”（此时 CWD 已经是脚本目录）。
DAT_NAME = "shape_predictor_68_face_landmarks.dat"
PREDICTOR_PATH = DAT_NAME
DAT_ABS = os.path.join(SCRIPT_DIR, DAT_NAME)


def imwrite_unicode(path, img):
    """支持中文路径的写图；返回 (是否成功, 说明)"""
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path += ext

    # 方案 A：常规 imwrite
    try:
        if cv2.imwrite(path, img):
            return True, path
    except Exception as e:
        last = f"imwrite 异常: {e}"
    else:
        last = "imwrite 返回 False（通常是权限/占用/路径问题）"

    # 方案 B：imencode + tofile（对中文路径更稳）
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True, path
            last += "；imencode 方案写入后文件为空"
        else:
            last += "；imencode 编码失败"
    except Exception as e:
        last += f"；imencode 方案异常: {e}"

    return False, last


def main():
    print("=" * 56)
    print("环境自检")
    print("=" * 56)
    print(f"脚本目录 : {SCRIPT_DIR}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"模型文件 : {'存在' if os.path.exists(DAT_ABS) else '缺失 !!!'}")
    print(f"目录可写 : {os.access(SCRIPT_DIR, os.W_OK)}")
    if not SCRIPT_DIR.isascii():
        print("提示     : 目录含非 ASCII 字符（中文），dlib 必须用相对文件名打开，已自动处理")

    if len(sys.argv) < 2:
        print("\n用法: py face_landmark_fixed.py <图片文件名> [输出文件名]")
        imgs = [f for f in os.listdir(".") if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        print("\n当前目录可用图片:")
        for f in imgs:
            print(f"  - {f}")
        return 1

    if not os.path.exists(DAT_ABS):
        print("\n✗ 找不到 shape_predictor_68_face_landmarks.dat，无法继续。")
        return 2

    src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = f"output_landmarked_{ts}.jpg"

    if not os.path.exists(src):
        print(f"\n✗ 找不到图片: {os.path.abspath(src)}")
        return 3

    print("\n" + "=" * 56)
    print(f"开始检测: {src}")
    print("=" * 56)

    im = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print("✗ 图片解码失败（文件损坏或不是图片）")
        return 4
    print(f"图片尺寸 : {im.shape[1]} x {im.shape[0]}")

    detector = dlib.get_frontal_face_detector()
    rects = detector(im, 1)
    print(f"检测到人脸数: {len(rects)}")

    if len(rects) == 0:
        print("✗ 没有检测到人脸 —— 这就是原脚本不产出文件的原因之一")
        return 5
    if len(rects) > 1:
        print("✗ 检测到多个人脸（原脚本会报 TooManyFaces 并中止）")

    try:
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except RuntimeError as e:
        print(f"✗ dlib 打开模型失败: {e}")
        print("  原因通常是路径含中文 —— 请确认脚本与 .dat 在同一个目录下运行")
        return 7
    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    print(f"特征点数 : {len(landmarks)}")

    canvas = im.copy()
    for point in landmarks:
        cv2.circle(canvas, (int(point[0, 0]), int(point[0, 1])), 3, (0, 255, 255), -1)

    ok, info = imwrite_unicode(dst, canvas)
    if ok:
        print(f"\n✓ 已保存 : {info}  ({os.path.getsize(info)} 字节)")
        return 0
    else:
        print(f"\n✗ 保存失败 : {info}")
        print("  排查方向：文件被杀毒软件/另一个程序占用、目录只读、受控文件夹访问")
        return 6


if __name__ == "__main__":
    sys.exit(main())
