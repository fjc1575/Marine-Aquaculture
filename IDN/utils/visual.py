import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_output(batch_output, batch_index=0, channel_index=None, mode='single', save_path=r'C:\polsar\code\PolSARSeg\asset\output\output'):
    """
    可视化批次输出中的某个图片的特定通道或所有通道的结果，并可选地保存图片。

    参数:
    - batch_output: Tensor或numpy数组, 形状为 (batch_size, num_channels, H, W)
    - batch_index: 要查看的批次中图片的索引 (默认 0)
    - channel_index: 要查看的特定通道索引，若为 None 则可视化所有通道 (默认 None)
    - mode: 可视化模式 - 'single' 单独可视化每个通道, 'mean' 平均化所有通道, 'max' 最大值投影 (默认 'single')
    - save_path: 保存图片的路径和文件名，若为 None 则不保存 (默认 None)
    """
    # 转换为numpy数组
    if isinstance(batch_output, torch.Tensor):
        batch_output = batch_output.detach().cpu().numpy()

    # 提取指定批次中的图片
    image_output = batch_output[batch_index]

    if mode == 'single':
        if channel_index is not None:
            # 可视化单个通道
            plt.imshow(image_output[channel_index], cmap='viridis')
            plt.xticks([])
            plt.yticks([])

            if save_path:
                plt.savefig(f"{save_path}_batch{batch_index}_channel{channel_index}.png",
                            bbox_inches='tight', pad_inches=0)
            plt.show()

        else:
            # 可视化所有通道
            num_channels = image_output.shape[0]
            for i in range(num_channels):
                plt.figure()
                plt.imshow(image_output[i], cmap='viridis')
                plt.xticks([])
                plt.yticks([])
                plt.title(f'Batch {batch_index}, Channel {i}')
                plt.colorbar()

                if save_path:
                    plt.savefig(f"{save_path}_batch{batch_index}_channel{i}.png",
                                bbox_inches='tight', pad_inches=0)
                plt.show()

    elif mode == 'mean':
        # 平均化所有通道
        output_mean = np.mean(image_output, axis=0)
        plt.imshow(output_mean, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Batch {batch_index}, Mean of All Channels')
        plt.colorbar()

        if save_path:
            plt.savefig(f"{save_path}_batch{batch_index}_mean.png",
                        bbox_inches='tight', pad_inches=0)
        plt.show()

    elif mode == 'max':
        # 最大值投影
        output_max = np.max(image_output, axis=0)
        plt.imshow(output_max, cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Batch {batch_index}, Max Projection of All Channels')
        plt.colorbar()

        if save_path:
            plt.savefig(f"{save_path}_batch{batch_index}_max.png",
                        bbox_inches='tight', pad_inches=0)
        plt.show()

    else:
        raise ValueError("Unsupported mode. Use 'single', 'mean', or 'max'.")

