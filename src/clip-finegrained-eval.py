import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_ROOT = 'CUB_200_2011'  # 数据集路径
BATCH_SIZE = 64
NUM_EPOCHS_RESNET = 10       # 为了快速演示，可适当增加
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#设置随机种子
def set_seed(seed=42):
    random.seed(seed)       # 锁住 Python 内置的 random 模块
    np.random.seed(seed)    # 锁住 NumPy 的随机数生成
    torch.manual_seed(seed) # 锁住 PyTorch 在 CPU 上的随机
    torch.cuda.manual_seed_all(seed)            # 锁住 PyTorch 在 GPU 上的随机
    torch.backends.cudnn.deterministic = True   # 强制 cuDNN（NVIDIA 的深度学习加速库）使用确定性算法
    torch.backends.cudnn.benchmark = False      # 关闭“自动寻找最优算法”功能

set_seed(42)

class CUBDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.images_dir = os.path.join(root, 'images')      #图像路径

        # 读取标签和划分
        labels = pd.read_csv(os.path.join(root, 'image_class_labels.txt'),
                             sep=' ', header=None, names=['image_id', 'class_id'])
        split_df = pd.read_csv(os.path.join(root, 'train_test_split.txt'),
                               sep=' ', header=None, names=['image_id', 'is_train'])
        images = pd.read_csv(os.path.join(root, 'images.txt'),
                             sep=' ', header=None, names=['image_id', 'path'])
        classes = pd.read_csv(os.path.join(root, 'classes.txt'),
                              sep=' ', header=None, names=['class_id', 'class_name'])
        # sep:分割符  header = 0（默认）：第一行是列名，跳过     header = None ：第一行也是数据，要从第一行开始读     names : 手动命名
        # pandas.read_csv()返回DataFrame结构

        # 合并，将四个DataFrame合并，最终 data 包含以下列：
        # image_id, class_id, is_train, path, class_name
        data = labels.merge(split_df, on='image_id').merge(images, on='image_id')
        data = data.merge(classes, on='class_id')

        # 筛选 split
        if split == 'train':
            data = data[data['is_train'] == 1]  #data['is_train'] == 1返回布尔掩码，data[<bool>]只取为true的行，相当于筛选。
        elif split == 'test':
            data = data[data['is_train'] == 0]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.data = data.reset_index(drop=True)     #丢弃旧的索引，让筛选后的data索引正常
        self.class_names = list(classes['class_name'])  # 完整类别名，如 '001.Black_footed_Albatross'
        # list()将其他类型转换为Python列表
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):     #返回第idx个样本
        row = self.data.iloc[idx]       #取出第idx行的信息
        img_path = os.path.join(self.images_dir, row['path'])   #得到第idx个样本的图片的真实地址
        image = Image.open(img_path).convert('RGB')         #打开为RGB三通道图片
        label = row['class_id'] - 1  # 转为 0~199,符合 PyTorch 损失函数的要求
        if self.transform:      #预处理transformer
            image = self.transform(image)
        return image, label

#文本提示词生成
def get_text_prompts(class_names):
    prompts = []
    for name in class_names:
        # 去掉前面的编号，如 '001.Black_footed_Albatross' -> 'Black footed Albatross'
        clean_name = name.split('.', 1)[1].replace('_', ' ')
        # split('.',1) 将字符串按“.”划分为两部分
        # replace('_',' ')将字符串中的下划线变为空格
        prompts.append(f"a photo of a {clean_name}")
        # append() 将元素加入到列表末尾，不返回任何值
    return prompts

#CLIP特征提取
def extract_clip_features(dataset, model, preprocess, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)#num_workers=4 开四个进程，加快读取速度
    features_list, labels_list = [], [] #两个空列表来存放特征和标签
    model.eval()            #变为评估模式
    with torch.no_grad():   #关闭梯度计算
        for images, labels in tqdm(loader, desc='Extracting features'): # desc 文字提示
            images = images.to(DEVICE)                  #将图片从CPU搬到GPU显存里，准备计算
            feats = model.encode_image(images)          #调用CLIP模型的encode_image()函数，将图片变为特征向量
            feats /= feats.norm(dim=-1, keepdim=True)   # 归一化，norm(dim=-1)求最后一维的长度，Keepdim = true，保持维度不变，便于运算
            features_list.append(feats.cpu())           #feats.cpu() 将特征搬回CPU内存
            labels_list.append(labels)
    return torch.cat(features_list), torch.cat(labels_list) #torch.cat()拼接为大张量


#ResNet训练函数
def train_resnet(train_loader, test_loader, num_epochs=10):
    # 1. 在函数内部定义设备（自动检测）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ResNet 正在使用设备: {device}")
    # 2. 加载模型
    model = models.resnet18(pretrained=True)        #预训练完毕
    num_ftrs = model.fc.in_features                 #fc 全连接层  in_features 输入到全连接层的特征数量
    model.fc = nn.Linear(num_ftrs, 200)  #修改ResNet最后的全连接层，让它输出200个分类
    model = model.to(device)                        #模型移入GPU

    criterion = nn.CrossEntropyLoss()               #损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)           #SGD优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    #学习率调度器，每7轮，将学习率变为原来的0.1倍

    best_acc = 0.0          #记录最高分
    for epoch in range(num_epochs):
        model.train()       #进入训练模式（微调）
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'ResNet Epoch {epoch + 1}/{num_epochs}'):    #tqdm 进度条
            images, labels = images.to(DEVICE), labels.to(DEVICE)          #图片和标签移入GPU
            optimizer.zero_grad()                                          #梯度清零
            outputs = model(images)                                        #向前传播
            loss = criterion(outputs, labels)                              #记录损失
            loss.backward()                                                #利用损失进行梯度下降反向传播
            optimizer.step()                                               #反向传播
            running_loss += loss.item()                                    #累计损失，记录一轮所有样本的损失总和
        scheduler.step()                                         #如果轮次是7的倍数，学习率就缩小10倍

        # 评估
        model.eval()                #进入测试模式
        correct, total = 0, 0
        with torch.no_grad():       #训练时不用梯度，节省算力
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)       #将图片和标签导入GPU
                outputs = model(images)                                     # outputs 是[64,200]的张量
                _, preds = torch.max(outputs, 1)    #torch.max()返回(最大值，最大值在的下标)，torch.max(outputs,1)表示在维度1找最大值
                correct += (preds == labels).sum().item()                   #预测正确的数量
                total += labels.size(0)                                     #样本总量
        acc = correct / total                                               #正确率
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Test Acc: {acc * 100:.2f}%')
        if acc > best_acc:
            best_acc = acc          #筛选最高正确率
    return best_acc                 #返回最高正确率


def main():
    # 加载 CLIP 模型和预处理
    print("Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    #open_clip.create_model_and_transforms()创建CLIP模型，VIT-B-32，使用base大小参数量的VIT，将图片切成32*32个小块，下载openai训练好的参数
    #返回三个：CLIP模型，模型的配置，模型预训练处理流程
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    #open_clip.get_tokenizer()获得文字分词器，将输入的英文映射为数字，VIT的型号必须和模型一致
    clip_model = clip_model.to(DEVICE)  #将模型放入GPU
    clip_model.eval()                   #进入测试模式

    # 创建数据集（先用 CLIP 预处理）
    print("Loading datasets...")
    train_dataset_clip = CUBDataset(DATA_ROOT, split='train', transform=clip_preprocess)
    test_dataset_clip = CUBDataset(DATA_ROOT, split='test', transform=clip_preprocess)

    # CLIP 零样本评估
    print("Computing CLIP zero-shot accuracy...")
    class_names = train_dataset_clip.class_names
    text_prompts = get_text_prompts(class_names)            #将类名改为一句提示词
    text_tokens = clip_tokenizer(text_prompts).to(DEVICE)   #将提示词映射为token，并搬到GPU上准备计算
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens) #将token转换为特征向量
        text_features /= text_features.norm(dim=-1, keepdim=True)   #将特征向量归一化

    test_loader_clip = DataLoader(test_dataset_clip, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    image_features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader_clip, desc='Zero-shot inference'):
            images = images.to(DEVICE)
            feats = clip_model.encode_image(images)         #将图片转化为特征向量
            feats /= feats.norm(dim=-1, keepdim=True)       #将图片的特征向量也归一化，这样和文字特征向量做点积得相似度
            image_features_list.append(feats.cpu())
            labels_list.append(labels)
    image_features = torch.cat(image_features_list)         #将小块合并为大张量
    labels = torch.cat(labels_list)

    image_features = image_features.to(DEVICE)
    labels = labels.to(DEVICE)
    similarity = image_features @ text_features.T
    # @ 矩阵乘法  .T 矩阵转置
    #image_features [N,512] text_features [200,512]   similarity [N,200] 这个矩阵中的每个数代表一张图片和一段文字的匹配程度，都在(-1,1)之间
    preds = similarity.argmax(dim=1)        #在[N,200]中的200个类中找最大值对应的索引
    zero_shot_acc = (preds == labels).float().mean().item()
    #.float() 将true变为1，false变为0
    #.mean() 求平均值
    #.item() 将张量里的数字变为Python数字
    print(f"CLIP Zero-shot Accuracy: {zero_shot_acc * 100:.2f}%")

    # 4. 线性探针（使用逻辑回归）
    print("Extracting features for linear probe...")
    train_feats, train_labels = extract_clip_features(train_dataset_clip, clip_model, clip_preprocess)
    test_feats, test_labels = extract_clip_features(test_dataset_clip, clip_model, clip_preprocess)
    #用CLIP模型对测试集和训练集提取特征

    print("Training linear probe (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000, C=0.316, n_jobs=-1, random_state=42)
    #逻辑回归分类器，允许最多迭代1000次，
    # C=0.316 正则化强度的倒数，越小，正则化越强（防止过拟合）
    clf.fit(train_feats.numpy(), train_labels.numpy())  #fit() 进行映射（训练）
    preds = clf.predict(test_feats.numpy())     #.predict() 进行预测返回编号
    linear_probe_acc = accuracy_score(test_labels.numpy(), preds)   #根据线性层预测结果计算准确率
    print(f"Linear Probe Accuracy: {linear_probe_acc * 100:.2f}%")

    # ResNet-18 基线
    print("Training ResNet-18 baseline...")
    # ResNet 使用不同的预处理
    resnet_transform = transforms.Compose([
        transforms.Resize(256),         #调整尺寸
        transforms.CenterCrop(224),     #裁剪
        transforms.ToTensor(),          #转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #归一化
    ])
    train_dataset_res = CUBDataset(DATA_ROOT, split='train', transform=resnet_transform)
    test_dataset_res = CUBDataset(DATA_ROOT, split='test', transform=resnet_transform)
    train_loader_res = DataLoader(train_dataset_res, batch_size=32, shuffle=True, num_workers=4)
    test_loader_res = DataLoader(test_dataset_res, batch_size=32, shuffle=False, num_workers=4)

    resnet_acc = train_resnet(train_loader_res, test_loader_res, num_epochs=NUM_EPOCHS_RESNET)
    print(f"ResNet-18 Best Accuracy: {resnet_acc * 100:.2f}%")

    # 结果汇总与可视化
    methods = ['CLIP Zero-shot', 'Linear Probe', 'ResNet-18']
    accs = [zero_shot_acc, linear_probe_acc, resnet_acc]

    # 保存到 CSV
    results_df = pd.DataFrame({'Method': methods, 'Accuracy': accs})
    results_df.to_csv('results_accuracies.csv', index=False)
    print("\nResults saved to results_accuracies.csv")

    # 绘制条形图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, [a * 100 for a in accs], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Accuracy (%)')
    plt.title('CUB-200-2011 Classification Accuracy')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{acc * 100:.2f}%', ha='center')
    plt.ylim(0, 100)
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to accuracy_comparison.png")


if __name__ == '__main__':
    main()
