import cv2
import numpy as np

# 定义标签对应的BGR颜色（注意：OpenCV使用的是BGR格式）
COLORS = {
    'ALL': (0, 0, 0),
    'VEGETATION': (159, 255, 84),  # RGB反写为BGR
    'BUILDING': (34, 180, 238),
    'WATER': (255, 191, 0),
    'ROAD': (38, 71, 139)
}

# 定义标签值
LABELS = {
    'ALL': 0,
    'VEGETATION': 1,
    'BUILDING': 2,
    'WATER': 3,
    'ROAD': 4
}


def colorize_image(input_src, input_mask):
    mask = cv2.imread(input_mask, cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(input_src)

    if mask is None or src is None:
        print("Error: Could not open or find the images!")
        return -1

    for i in range(src.shape[0]):  # 遍历每一行
        for j in range(src.shape[1]):  # 遍历每一列
            label = mask[i, j]
            if label == LABELS['VEGETATION']:
                src[i, j] = COLORS['VEGETATION']
            elif label == LABELS['ROAD']:
                src[i, j] = COLORS['ROAD']
            elif label == LABELS['BUILDING'] or label == 255:
                src[i, j] = COLORS['BUILDING']
            elif label == LABELS['WATER']:
                src[i, j] = COLORS['WATER']

    cv2.imwrite("stack.png", src)
    print("Colored image saved as stack.png")
    return 0


if __name__ == "__main__":
    # 直接在这里指定输入图像和掩码的路径
    input_src_path = "C:/Users/public_account/Desktop/3.png"  # 替换为你的源图像路径
    input_mask_path = "C:/Users/public_account/PycharmProjects/PythonProject/low_airspace/predict/pre_1.png"  # 替换为你的掩码图像路径

    # 调用函数进行图像着色
    result = colorize_image(input_src_path, input_mask_path)

    if result != 0:
        print("An error occurred while processing the images.")
