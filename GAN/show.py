import cv2
import os
import time


# 指定图片所在文件夹路径
img_folder = "./images/gan/"
# 获取文件夹中所有灰度图片文件的文件名，并按名称排序
img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')], key=lambda x: int(x[:-4]))
# 循环展示每张图片
for img_file in img_files:
    # 读取灰度图片
    img_path = os.path.join(img_folder, img_file)
    img = cv2.imread(img_path, 0)
    # 放大图片
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 显示图片并等待按下任意键
    cv2.imshow(img_path, img)
    # time.sleep(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
