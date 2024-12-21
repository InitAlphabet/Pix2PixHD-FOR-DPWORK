# ------------------计算FID
import glob
from torchvision import models
from torchvision.models import Inception_V3_Weights
import torch
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from torchvision import transforms
from p2phdconfigs import test_config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = test_config.TEST_OUTPUT_PATH+"/*.png"

target_path = test_config.TEST_TARGET_PATH+"/*.png"

# 加载InceptionV3模型并使用最新的预训练权重
# 加载InceptionV3模型并使用最新的预训练权重
inception_model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception_model.eval().to(device)


def get_inception_features(img_paths):
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    features = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = inception_model(img)
            features.append(feature.cpu().numpy().squeeze())  # 移除批次维度
    features = np.array(features)
    # 确保特征是二维的
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    return features


def calculate_fid(real_features, generated_features):
    # 计算均值和协方差
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # 计算FID
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


# 获取真实图像和生成图像的InceptionV3特征
real_img_paths = sorted(glob.glob(target_path))
generated_img_paths = sorted(glob.glob(output_path))  # 假设你已经保存了生成的图像

real_features = get_inception_features(real_img_paths)
generated_features = get_inception_features(generated_img_paths)

fid_score = calculate_fid(real_features, generated_features)
print(f"FID: {fid_score}")
