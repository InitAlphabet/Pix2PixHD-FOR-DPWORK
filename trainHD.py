from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os

import tool
from DataManager import Pix2PixDataset
from p2phdconfigs import train_config as config_f
from model_genrator import p2phdModel

config = config_f
pre_models = None
if config_f.EXIST_MODEL_PT:
    pre_models = torch.load(config_f.MODEL_PT_PATH)
    config = pre_models['config']
    config.START_EPOCH = config.EPOCHS
    config.EPOCHS = config_f.EPOCHS
    config.EXIST_MODEL_PT = True
# 固定随机种子
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)

# 数据预处理操作
transformer = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化调整特征尺度。
])

# 数据加载
train_dataset = Pix2PixDataset(config.TRAIN_DATA_PATH, config.TRAIN_TARGET_PATH, transformer)
test_dataset = Pix2PixDataset(config.TEST_DATA_PATH, config.TEST_TARGET_PATH, transformer)

# 数据转换
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = p2phdModel(config)

if pre_models:
    model.load_state_dict(pre_models['pts'])
model.to(device)


def save_model(path):
    config.START_EPOCH += config.EPOCHS
    model_dicts = {
        'config': config,
        'pts': model.state_dict()
    }
    save_name = f"ck_points_{config.START_EPOCH}"
    torch.save(model_dicts, path + '/' + save_name)


# 创建可视化输出文件夹
output_dir = tool.create_next_train_directory(config.TRAIN_OUTPUT_PATH, "train")
model_save_dir = output_dir + "/model"
epoch_save_dir = output_dir + "/epochs"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(epoch_save_dir, exist_ok=True)
GenLosses, DisLosses = [], []

# 开始训练
start_epoch = config.START_EPOCH
epochs = config.EPOCHS
curr_dis = 0

for epoch in range(start_epoch, start_epoch + epochs):
    model.train()
    gen_loss, disc_loss = 0, 0
    for img, target in train_loader:
        img, target = img.to(device), target.to(device)
        losses = model(img, target)
        losses = [torch.mean(x) if not isinstance(x, float) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))
        loss_D = (loss_dict['D_fake_loss'] + loss_dict['D_real_loss']) * 0.5
        loss_G = loss_dict['GAN_loss'] + loss_dict.get('GAN_feature_loss', 0) + loss_dict.get('VGG_loss', 0)
        model.optimizer_G.zero_grad()
        loss_G.backward()
        model.optimizer_G.step()
        model.optimizer_D.zero_grad()
        loss_D.backward()
        model.optimizer_D.step()
        gen_loss = loss_G.item()
        disc_loss = loss_D.item()
    GenLosses.append(gen_loss)
    DisLosses.append(disc_loss)

    print(
        f"Epoch [{epoch + 1}/{start_epoch + config.EPOCHS}], Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")  # 可视化生成结果
    if (epoch + 1) % config.VISUALIZE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            img, target = next(iter(test_loader))
            img, target = img.to(device), target.to(device)
            gen_output = model(img)

        # 绘制图像
        fig, ax = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(4):
            # 转置matplotlib格式，进行还原图像。
            ax[0, i].imshow(((img[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[1, i].imshow(((target[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[2, i].imshow(((gen_output[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
            ax[0, i].set_title("Input")
            ax[1, i].set_title("Target")
            ax[2, i].set_title("Generated")
            for j in range(3):
                ax[j, i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{epoch_save_dir}/epoch_{epoch + 1}.png")
        plt.close(fig)

# 损失可视化
plt.figure(figsize=(8, 6))
epochs = list(range(1, config.EPOCHS + 1))
plt.plot(epochs, GenLosses, label='Gen', color='blue', marker='o')
plt.plot(epochs, DisLosses, label='Dis', color='red', marker='s')
plt.title('LOSSES')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# 显示网格
plt.grid(True)
# 添加图注（legend）
plt.legend()
plt.savefig(f"{output_dir}/LOSS.png")
save_model(model_save_dir)
