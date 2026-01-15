"""
剪枝掩码可视化工具
将剪枝决策掩码映射回原图像，展示哪些图像区域被保留或剪枝
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict
import os


class PruningMaskVisualizer:
    """剪枝掩码可视化器"""
    
    def __init__(
        self, 
        image_size: int = 336,
        patch_size: int = 14,
        num_patches_per_side: int = 24,  # 576 = 24*24
    ):
        """
        Args:
            image_size: 输入图像大小（预处理后）
            patch_size: 每个patch的大小
            num_patches_per_side: 每边的patch数量（对于576 tokens，通常是24）
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = num_patches_per_side
        self.num_patches = num_patches_per_side ** 2
        
    def mask_to_image_overlay(
        self, 
        image: Image.Image,
        mask: torch.Tensor,
        alpha: float = 0.6,
        retained_color: Tuple[int, int, int] = (0, 0, 0),  # 绿色表示保留
        pruned_color: Tuple[int, int, int] = (180, 180, 180),     # 红色表示剪枝
        original_image_shape: Optional[int] = None,  # 原始图像token数量（如576）
        current_token_num: Optional[int] = None,  # 当前mask对应的token数量（输入token数）
    ) -> Image.Image:
        """
        将剪枝掩码叠加到原图像上
        
        Args:
            image: 原始PIL图像
            mask: 剪枝掩码 [L_v]，布尔张量，True表示保留（top-k），False表示剪枝
            alpha: 叠加透明度
            retained_color: 保留区域的标记颜色（RGB）
            pruned_color: 剪枝区域的标记颜色（RGB）
            original_image_shape: 原始图像token数量（如576）
            current_token_num: 当前mask对应的输入token数量（已废弃，保留兼容性）
            
        Returns:
            可视化后的图像
        """
        # 确保mask是1D的
        if mask.dim() > 1:
            mask = mask.squeeze()
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_np = mask_np.astype(bool)
        
        # 始终使用原始图像大小进行可视化，确保完整覆盖
        if original_image_shape is None:
            original_image_shape = self.num_patches_per_side * self.num_patches_per_side  # 默认576
        
        num_patches_per_side = self.num_patches_per_side
        
        # mask应该已经是原始图像大小了（因为我们在收集时就映射回原始图像）
        # 这里只需要确保大小正确
        if len(mask_np) != original_image_shape:
            if len(mask_np) < original_image_shape:
                # 如果mask较小，填充False
                mask_np = np.pad(mask_np, (0, original_image_shape - len(mask_np)), constant_values=False)
            else:
                # 如果mask较大，截断
                mask_np = mask_np[:original_image_shape]
        
        # 确保mask_np的大小匹配num_patches_per_side^2
        expected_size = num_patches_per_side * num_patches_per_side
        if len(mask_np) != expected_size:
            if len(mask_np) < expected_size:
                # 填充到期望大小
                mask_np = np.pad(mask_np, (0, expected_size - len(mask_np)), constant_values=False)
            else:
                # 截断到期望大小
                mask_np = mask_np[:expected_size]
        
        # 调整图像大小到标准尺寸
        image_resized = image.resize((self.image_size, self.image_size))
        
        # 创建覆盖层
        overlay = Image.new('RGBA', image_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 将mask重塑为2D网格
        mask_2d = mask_np.reshape(num_patches_per_side, num_patches_per_side)
        
        # 绘制每个patch
        patch_size_px = self.image_size // num_patches_per_side
        
        for i in range(num_patches_per_side):
            for j in range(num_patches_per_side):
                x1 = j * patch_size_px
                y1 = i * patch_size_px
                x2 = x1 + patch_size_px
                y2 = y1 + patch_size_px
                
                # 绘制每个patch
                if mask_2d[i, j]:
                    # 保留的区域：绿色半透明
                    color = retained_color + (int(255 * alpha),)
                else:
                    # 剪枝的区域：红色半透明
                    color = pruned_color + (int(255 * alpha),)
                
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # 将覆盖层叠加到原图
        result = Image.alpha_composite(image_resized.convert('RGBA'), overlay)
        return result.convert('RGB')
    
    def visualize_layer2(
        self,
        image: Image.Image,
        mask: torch.Tensor,
        save_path: Optional[str] = None,
        topk_retained: Optional[int] = None,
        original_image_shape: Optional[int] = None,
    ) -> Image.Image:
        """
        可视化Layer 2的剪枝掩码（第一层稀疏化）
        
        Args:
            image: 原始图像
            mask: Layer 2的剪枝掩码 [576]，布尔张量，True表示保留
            save_path: 保存路径（可选）
            topk_retained: top-k保留的token数量（可选）
            original_image_shape: 原始图像token数量（默认576）
            
        Returns:
            可视化图像
        """
        if original_image_shape is None:
            original_image_shape = self.num_patches_per_side * self.num_patches_per_side  # 576
        
        # 计算保留率
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        retained_count = mask_np.sum()
        retention_rate = retained_count / original_image_shape * 100
        
        # # 创建可视化
        vis_image = self.mask_to_image_overlay(image, mask, original_image_shape=original_image_shape)
        
        # # 创建对比图（原图 + 剪枝结果）
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # # 原图
        # axes[0].imshow(image)
        # axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        # axes[0].axis('off')
        
        # # 剪枝结果
        # axes[1].imshow(vis_image)
        # if topk_retained is not None:
        #     retention_text = f'Layer 2 Pruning\nRetention: {retention_rate:.1f}% (top-k: {topk_retained}/{original_image_shape})'
        # else:
        #     retention_text = f'Layer 2 Pruning\nRetention: {retention_rate:.1f}% ({int(retained_count)}/{original_image_shape})'
        # axes[1].set_title(retention_text, fontsize=12, fontweight='bold')
        # axes[1].axis('off')
        
        # # 添加图例
        # legend_elements = [
        #     mpatches.Patch(facecolor=(0, 1, 0, 0.5), label='Retained Patches'),
        #     mpatches.Patch(facecolor=(1, 0, 0, 0.5), label='Pruned Patches')
        # ]
        # fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12)
        
        # plt.tight_layout()
        
        # if save_path:
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
        #     print(f"Visualization saved to: {save_path}")
        
        # # 转换为PIL图像返回
        # fig.canvas.draw()
        # try:
        #     if hasattr(fig.canvas, 'buffer_rgba'):
        #         buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        #         width, height = fig.canvas.get_width_height()
        #         buf = buf.reshape((height, width, 4))
        #         buf = buf[:, :, :3]
        #         plt.close(fig)
        #         return Image.fromarray(buf)
        #     elif hasattr(fig.canvas, 'tostring_rgb'):
        #         buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #         buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #         plt.close(fig)
        #         return Image.fromarray(buf)
        # except Exception:
        #     pass
        
        # from io import BytesIO
        # buf = BytesIO()
        # fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        # buf.seek(0)
        # plt.close(fig)
        # return Image.open(buf)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_image.save(save_path)
            print(f"Pruning visualization saved to: {save_path}")

        return vis_image
    

