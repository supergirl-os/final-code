from utils.utils import train_one_epoch, evaluate, collate_fn
from utils.DataLoader import *
from models.STNet import *


root = r'data_for_ML/'
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 2类 背景和车牌
num_classes = 2

# 准备数据集
dataset = DataLoader(root)
dataset_test = DataLoader(root)

# 数据集一共有75张图，差不多训练测试4:1
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-100])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, collate_fn=collate_fn)

# 获取模型
model = STNet()
model.to(device)

# 设置优化器
params = [p for p in model.parameters() if p.requires_grad]
# SGD
optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)

# 初始化学习率
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

# let's train it for epochs
num_epochs = 31

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

    lr_scheduler.step()

    # 测试集上验证
    evaluate(model, data_loader_test, device=device)

    print('==================================================')


print("Finished training! ")

# 保存模型
torch.save(model, r'model.pkl')