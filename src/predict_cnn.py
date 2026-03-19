import torch
import numpy as np
from model_cnn import FashionCNN  # 从同一 src 文件夹导入模型

device = torch.device("cpu")

# 加载训练好的模型
model = FashionCNN()
model.load_state_dict(torch.load("./models/best_fashion_cnn.pth", map_location=device))

model.eval()

def predict_cnn(image_array):
    image_array = image_array / 255.0
    tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred