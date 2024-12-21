import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from model_genrator import p2phdModel
import torchvision.transforms as transforms
from p2phdconfigs import test_config as configs

# 加载模型
model_path = configs.MODEL_PT_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_model = torch.load(model_path)
config = pre_model['config']
model = p2phdModel(config)
model.load_state_dict(pre_model['pts'])
model.to(device)
model.eval()

# 创建 Tkinter 窗口
root = tk.Tk()
root.title("Image Prediction")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_height = 500
window_width = 600
# 计算窗口位置，使其居中显示
position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)
# 设置窗口位置和大小
root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
# 设置窗口大小
root.config(bg='#5752C6')
# 创建画布显示图像
canvas = tk.Canvas(root, width=600, height=400)
canvas.pack(side=tk.TOP, padx=44, pady=10, expand=True)

# 数据预处理操作
transformer = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化调整特征尺度。
])


def load_and_predict_image():
    # 弹出文件选择对话框，让用户选择一张图片
    input_image_path = filedialog.askopenfilename()
    if not input_image_path:
        return
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
        gen_output.save('output/test/test.png')
        # 调整生成图像的大小，使其与原始图像相同
    img = img.resize(gen_output.size)
    # 拼接原始图像和预测图像
    combined_image = Image.new('RGB', (img.width + img.width + 10, img.height))
    combined_image.paste(img, (0, 0))
    combined_image.paste(gen_output, (gen_output.width + 10, 0))

    # 显示拼接后的图像
    combined_image = ImageTk.PhotoImage(combined_image)
    canvas.create_image(0, 10, anchor=tk.NW, image=combined_image)
    canvas.image = combined_image  # 保持对图像的引用


# 创建按钮，用于选择图片并进行预测
predict_button = tk.Button(root, text="Select and Predict Image", command=load_and_predict_image)
predict_button.pack(side=tk.BOTTOM, pady=20)

# 运行 GUI
root.mainloop()
