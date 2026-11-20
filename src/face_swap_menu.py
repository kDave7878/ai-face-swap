"""
基于深度学习的人脸识别与替换工具（时间戳版）
在原作者脚本基础上只做最小改动：
1. 输出文件名自动追加时间戳，不再覆盖上一次的结果
2. imwrite 的真实结果会被检查，失败时如实报错（不再假装“✓ 完成”）
3. ★ 修复黑块：warp 时把 float64 蒙版转为 float32
   （OpenCV 对 CV_64F 处理 WARP_INVERSE_MAP 的方向与 CV_8U/CV_32F 相反，
     会让蒙版落到错误位置，合成区覆盖到源图之外 → 黑块）
4. 加强保险：蒙版乘上"有效覆盖区"，越界像素不参与合成
5. 修复噪点：合成结果先 clip 到 [0,255] 再转 uint8，避免溢出环绕

其余逻辑（菜单、特征点检测、面部替换算法）与原脚本完全一致。
包含两个主要功能：
1. 人脸特征点检测与标记
2. 面部替换（深度伪造技术）
"""

import datetime
import os

import cv2
import dlib
import numpy as np

# 配置常量
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6

# 定义面部特征点的索引范围
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# 用于面部对齐的关键特征点和重叠区域的特征点
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# 使用 dlib 进行面部检测和特征点预测
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# 自定义异常类
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def timestamped(basename, ext='.jpg'):
    """生成带时间戳的输出文件名，例如 output_faceswap_20261010_153100.jpg"""
    return '{}_{}{}'.format(basename, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), ext)


def save_image(path, im):
    """保存图片并返回真实结果（原脚本忽略了 imwrite 的返回值）"""
    try:
        if cv2.imwrite(path, im):
            print("✓ 保存成功: {}  ({} 字节)".format(os.path.abspath(path), os.path.getsize(path)))
            return True
    except Exception as e:
        print("✗ 保存异常: {}".format(e))
        return False
    print("✗ 保存失败: {}  —— 可能原因：文件被占用 / 目录只读 / 杀毒软件拦截".format(path))
    return False


# ==================== 功能1：人脸特征点检测与标记 ====================

def get_landmarks(im):
    """获取图像中的面部特征点"""
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces("检测到多个人脸")
    if len(rects) == 0:
        raise NoFaces("没有检测到任何人脸")
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    """在图像上标记特征点"""
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(im, pos, radius=3, color=(0, 255, 255), thickness=-1)
    return im


def read_im_and_landmarks(fname):
    """读取图像并获取特征点"""
    try:
        im_array = np.fromfile(fname, dtype=np.uint8)
        im = cv2.imdecode(im_array, cv2.IMREAD_COLOR)
    except Exception:
        im = cv2.imread(fname, cv2.IMREAD_COLOR)

    if im is None:
        raise FileNotFoundError("无法读取图像: {}".format(fname))
    im = cv2.resize(im, dsize=(int(im.shape[1] * SCALE_FACTOR), int(im.shape[0] * SCALE_FACTOR)))
    s = get_landmarks(im)
    return im, s


def main_face_landmark_detection():
    """功能1主函数：人脸特征点检测与标记"""
    print("=" * 50)
    print("功能1：人脸特征点检测与标记")
    print("=" * 50)
    print("\n当前目录的图片文件:")
    images = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for img in images:
        print("  - {}".format(img))

    image_path = input("\n输入图片路径或文件名: ")
    try:
        im, landmarks = read_im_and_landmarks(image_path)
        im_with_landmarks = annotate_landmarks(im, landmarks)
        output_path = timestamped('output_landmarked')          # ← 改动：加时间戳
        save_image(output_path, im_with_landmarks)              # ← 改动：检查真实结果
        print("✓ 共检测到 {} 个特征点".format(len(landmarks)))
    except (TooManyFaces, NoFaces) as e:
        print("✗ 错误: {}".format(e))
    except FileNotFoundError as e:
        print("✗ 错误: {}".format(e))
    except Exception as e:
        print("✗ 未知错误: {}".format(e))


# ==================== 功能2：面部替换（深度伪造） ====================

def transformation_from_points(points1, points2):
    """根据两组特征点生成仿射变换矩阵"""
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                c2.T - (s2 / s1) * R * c1.T)),
                     np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    """
    对图像进行仿射变换。

    ★ 黑块的真正原因：OpenCV 对 CV_64F(float64) 输入处理 WARP_INVERSE_MAP
    的方向与 CV_8U / CV_32F 相反。人脸蒙版是 float64，直接 warp 会落到完全
    错误的位置，导致 combined_mask 覆盖到源图范围之外 → 合成出大片黑块。
    （改成 BORDER_REPLICATE 只是把黑块换成模糊/高亮的 smear，治标不治本。）

    这里 float64 先转 float32，warp 完再转回来，语义就一致了。
    """
    is_f64 = (im.dtype == np.float64)
    work = im.astype(np.float32) if is_f64 else im
    output_im = np.zeros(dshape, dtype=work.dtype)
    cv2.warpAffine(work,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im.astype(np.float64) if is_f64 else output_im


def get_face_mask(im, landmarks):
    """获取面部的掩码，用于后续替换操作"""
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        hull = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, hull, color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def correct_colours(im1, im2, landmarks1):
    """对两张图像进行颜色校正，使它们的色调更加一致"""
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def main_face_swap():
    """功能2主函数：面部替换"""
    print("=" * 50)
    print("功能2：面部替换（深度伪造技术）")
    print("=" * 50)
    print("\n当前目录的图片文件:")
    images = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for img in images:
        print("  - {}".format(img))

    image_path1 = input("\n输入第一张图片路径或文件名（目标人脸）: ")
    image_path2 = input("输入第二张图片路径或文件名（源人脸）: ")

    try:
        print("正在处理第一张图片...")
        im1, landmarks1 = read_im_and_landmarks(image_path1)
        print("✓ 第一张图片处理完成")

        print("正在处理第二张图片...")
        im2, landmarks2 = read_im_and_landmarks(image_path2)
        print("✓ 第二张图片处理完成")

        print("正在进行面部替换...")
        M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
        mask = get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)
        combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
        # 保险栓：把"采样落到源图之外"的像素从蒙版里剔除
        cover = warp_im(np.full(im2.shape, 255, np.uint8), M, im1.shape)[:, :, 0] > 0
        combined_mask = combined_mask * cover[:, :, None]
        warped_im2 = warp_im(im2, M, im1.shape)
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        # 先裁剪到 [0,255] 再转 uint8，避免颜色校正溢出后环绕成噪点
        output_im = np.clip(output_im, 0, 255).astype(np.uint8)
        output_path = timestamped('output_faceswap')            # ← 改动：加时间戳
        save_image(output_path, output_im)                      # ← 改动：检查真实结果

    except (TooManyFaces, NoFaces) as e:
        print("✗ 错误: {}".format(e))
    except FileNotFoundError as e:
        print("✗ 错误: {}".format(e))
    except Exception as e:
        print("✗ 未知错误: {}".format(e))


# ==================== 主程序入口 ====================

def main():
    """主函数：选择功能"""
    print("\n" + "=" * 50)
    print("基于深度学习的人脸识别与替换工具（时间戳版）")
    print("=" * 50)
    print("\n请选择功能:")
    print("1. 人脸特征点检测与标记")
    print("2. 面部替换（深度伪造）")
    print("0. 退出")

    while True:
        choice = input("\n请输入选项 (0/1/2): ").strip()

        if choice == '0':
            print("退出程序")
            break
        elif choice == '1':
            main_face_landmark_detection()
            break
        elif choice == '2':
            main_face_swap()
            break
        else:
            print("✗ 无效选项，请重新输入")


if __name__ == '__main__':
    if not os.path.exists(PREDICTOR_PATH):
        print("=" * 50)
        print("警告: 未找到特征点预测模型文件!")
        print("请确保 '{}' 文件存在于当前目录".format(PREDICTOR_PATH))
        print("下载地址: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("=" * 50)
    else:
        main()
