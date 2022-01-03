# 是否开启调试模式
debug = True

# === For PlateLocation-> locate ===
# 高斯模糊所用变量
GaussianBlurSize = (5, 5)  # 可以修改为3，7，0，分别查看结果
# 轮廓面积最小值，用于筛除过小的轮廓，单位px
MIN_AREA = 50

# === For PlateLocation-> judge ===
Height = 14.0  # 中国车牌高大约为14cm
Width = 44.0  # 中国车牌长大约为44cm
Aspect = 3.2  # 车牌纵横比，西班牙标准车牌纵横比为4.7272，中国车牌纵横比为3.142857
Error = 0.3  # 车牌纵横比误差
StandardArea = Height * Width  # 标准面积

VerifyMin = 1.5  # 最小面积的放缩系数
VerifyMax = 120  # 最大面积的放缩系数

# === For ColorLocation ===
# ColorMin = [70, 160, 130]  # 设置色域上下界HSV
# ColorMax = [130, 255,255]
ColorMin = [50, 60, 56]  # 设置色域上下界HSV
ColorMax = [124, 255, 255]
blur = False  # 是否进行降噪

# === For YOLO_detection ===
Recognize = True
