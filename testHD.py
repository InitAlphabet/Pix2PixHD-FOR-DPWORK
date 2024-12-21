import glob
import os
import torch
from PIL import Image
from torchvision.transforms import transforms

from model_genrator import p2phdModel
from p2phdconfigs import test_config
# 加载模型
model_path = test_config.MODEL_PT_PATH
output_dir = test_config.TEST_OUTPUT_PATH
target_dir = test_config.TEST_DATA_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(output_dir, exist_ok=True)
pre_model = torch.load(model_path)
config = pre_model['config']
model = p2phdModel(config)
model.load_state_dict(pre_model['pts'])
model.to(device)
model.eval()

# 数据预处理操作
transformer = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化调整特征尺度。
])


# 生成并保存图像的函数
def generate_and_save_images(input_image_paths):
    os.makedirs(output_dir, exist_ok=True)
    generated_img_paths = []
    for i, input_image_path in enumerate(input_image_paths):
        img = Image.open(input_image_path).convert("RGB")
        img_tensor = transformer(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            gen_output = model(img_tensor)
            gen_output = gen_output.squeeze(0)
            gen_output = gen_output.cpu().numpy().transpose(1, 2, 0)
            gen_output = (gen_output + 1) * 127.5
            gen_output = gen_output.astype('uint8')
            gen_output = Image.fromarray(gen_output)
        # 保存生成的图像
        generated_image_path = os.path.join(output_dir, f"{i}.png")
        # Image.fromarray(gen_output).save(generated_image_path)
        gen_output.save(generated_image_path)
        generated_img_paths.append(generated_image_path)

    return generated_img_paths


target_png_paths = sorted(glob.glob((target_dir + "/*.png")))
generate_and_save_images(target_png_paths)
