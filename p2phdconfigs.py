from tool import ConfigDict

train_config = ConfigDict({
    "TRAIN_DATA_PATH": "data/data/train_A",  # 训练集
    "TRAIN_TARGET_PATH": "data/data/train_B",  # 训练集标准输出
    "TEST_DATA_PATH": "data/data/test_A",  # 测试集
    "TEST_TARGET_PATH": "data/data/test_B",  # 测试集标注输出
    "IMAGE_SIZE": 512,  # 图片大小
    "EPOCHS": 50,  # 训练伦次
    "START_EPOCH": 0,  # 起始epoch 不要修改，用于保存
    "VISUALIZE_EVERY": 1,  # 训练时测试频率
    "TRAIN_OUTPUT_PATH": "output/train",  # 训练结果的输出位置
    "SEED": 233,  # 随机种子
    "G_LR": 2e-4,  # 生成器学习率
    "D_LR": 2e-4,  # 判别器学习率
    "BATCH_SIZE": 8,  # 批大小
    "G_BETA": (0.5, 0.999),  # 生成器adam参数
    "D_BETA": (0.5, 0.999),  # 判别器adam参数
    "EXIST_MODEL_PT": True,  # 是否加载已经训练的权重,为true时需要MODEL_PT_PATH 不能为空
    "MODEL_PT_PATH": "output/train/train1/model/ck_points_50",  # 预训练模型权重
    "NEED_FEATURE_MAPS": True,  # 空间感知代替l1损失
    "NUM_DIS": 3,  # 多层感知器层数
    "LAYER_NUM": 3,  # 感知器下采样层数
    "FEATURE_BETA": 10  # 感知损失系数
})
test_config = ConfigDict({
    "TEST_DATA_PATH": "data/data/test_A",  # 测试集
    "TEST_TARGET_PATH": "data/data/test_B",  # 测试集标注输出
    "TEST_OUTPUT_PATH": "output/test/test1",  # 测试输出位置
    "MODEL_PT_PATH": "",  # 用于测试的模型权重
    "IMAGE_SIZE": 512  # 图片大小
})
