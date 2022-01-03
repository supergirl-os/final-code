## 数字图像处理期末大作业

#### **基于给定车牌数据集的车牌定位问题**

- 形状筛选
- 颜色形状筛选
- 目标检测模型（ML）

#### 代码目录结构描述

├── Readme.md                   // help
├── main.py                     // 包含三种方法，直接调用即可分别查看每种方法的效果
├── params.py               	// 参数设置
├── train.py          			// 本地训练目标检测模型
├── utils                     	// 各种辅助方法类
│   ├── Dataloader.py			// 数据加载器，包含两个类，产生DataSet
│   ├── utils.py         		// 各种方法，包括对于数据集处理、结果可视化、模型构建等
│   └── transforms.py        	// 数据变形方法
│   └── modules.py       		// 构建网络的功能块
│   └── torch_utils.py       	// 官方提供方法
│   └── goole_utils.py          // 官方提供方法
├── models                   	// 各种方法类
│   ├── PlateLocation.py		// 形状筛选方法，方案一
│   ├── ColorLocation.py		// 形状颜色筛选方法，方案二
│   ├── YOLO_detection.py		// 目标检测的机器学习方法
│   ├── LPRNet.py				// LPRNet网络的实现
│   ├── MTCNN.py				// MTCNN网络的实现
│   ├── STNet.py				// STNet网络的实现
│   └── yolo.py			    	// YOLO网络实现 
├── traind_models               // 预训练模型
│   └── lastforyolo.pt			// 车牌定位模型
│   └─  Final_LPRNet_model.pth	// 车牌识别模型







