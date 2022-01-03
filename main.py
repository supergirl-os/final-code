# 主入口函数
from models.PlateLocation import *
from models.ColorLocation import *
from models.YOLO_detection import *
from utils.DataLoader import *

PATH = "data/"


# 传统形状筛选法
def method_1():
    dst = "dst_1/"
    if not os.path.exists(dst):
        # 不存在则创建
        os.mkdir(dst)
    for file in os.listdir(PATH):
        imagePath = PATH + file
        image = cv2.imread(imagePath)
        solver = PlateLocation(image)
        patch = solver.locate()
        cv2.imwrite(dst+file, patch)


# 传统颜色筛选法
def method_2():
    dst = "dst_2/"
    iou = []        # 存放每张图片的IoU
    if not os.path.exists(dst):
        # 不存在则创建
        os.mkdir(dst)
    for file in os.listdir(PATH):
        imagePath = PATH + file
        image = cv2.imread(imagePath)
        solver = ColorLocation(image,file)
        patch, aver_iou = solver.locate()
        iou.append(aver_iou)
        cv2.imwrite(dst + file, patch)
    total = 0
    for i in iou:
        total += i
    average = total/75
    print("数据集的IoU:", average)
    draw_iou(iou)


# 机器学习模型
def method_3():
    test = YOLO_detection()
    test.locate()



method_3()