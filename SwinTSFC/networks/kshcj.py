import matplotlib.pyplot as plt
import os
import numpy as np
import cv2



batch, channels, height, width = x.shape # batch, channels, height, width
blocks = torch.chunk(x[0].cpu(), channels, dim=0)
output_folder = 'channel_outputs1'
os.makedirs(output_folder, exist_ok=True)

for i in range(x.shape[1]):
channel_output = blocks[i].squeeze().detach().numpy()
channel_output = (channel_output - np.min(channel_output)) / (
np.max(channel_output) - np.min(channel_output)) # 缩放到0到1的范围
channel_output = (channel_output * 255).astype(np.uint8) # 缩放到0到255的范围，并转换为np.uint8类型
channel_output = cv2.applyColorMap(channel_output, cv2.COLORMAP_JET)
plt.imshow(channel_output, cmap=plt.cm.jet) # 使用合适的颜色映射
plt.title(f'Channel {i + 1}')
plt.axis('off')
output_path = os.path.join(output_folder, f'channel_{i + 1}.png')
plt.savefig(output_path)
plt.clf()

