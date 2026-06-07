
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict
import os


class PruningMaskVisualizer:
    
    def __init__(
        self, 
        image_size: int = 336,
        patch_size: int = 14,
        num_patches_per_side: int = 24,  # 576 = 24*24
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches_per_side = num_patches_per_side
        self.num_patches = num_patches_per_side ** 2
        
    def mask_to_image_overlay(
        self, 
        image: Image.Image,
        mask: torch.Tensor,
        alpha: float = 0.6,
        retained_color: Tuple[int, int, int] = (0, 0, 0),
        pruned_color: Tuple[int, int, int] = (180, 180, 180),
        original_image_shape: Optional[int] = None,
        current_token_num: Optional[int] = None,
    ) -> Image.Image:

        if mask.dim() > 1:
            mask = mask.squeeze()
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_np = mask_np.astype(bool)
        

        if original_image_shape is None:
            original_image_shape = self.num_patches_per_side * self.num_patches_per_side
        
        num_patches_per_side = self.num_patches_per_side
        


        if len(mask_np) != original_image_shape:
            if len(mask_np) < original_image_shape:

                mask_np = np.pad(mask_np, (0, original_image_shape - len(mask_np)), constant_values=False)
            else:

                mask_np = mask_np[:original_image_shape]
        

        expected_size = num_patches_per_side * num_patches_per_side
        if len(mask_np) != expected_size:
            if len(mask_np) < expected_size:

                mask_np = np.pad(mask_np, (0, expected_size - len(mask_np)), constant_values=False)
            else:

                mask_np = mask_np[:expected_size]
        

        image_resized = image.resize((self.image_size, self.image_size))
        

        overlay = Image.new('RGBA', image_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        

        mask_2d = mask_np.reshape(num_patches_per_side, num_patches_per_side)
        

        patch_size_px = self.image_size // num_patches_per_side
        
        for i in range(num_patches_per_side):
            for j in range(num_patches_per_side):
                x1 = j * patch_size_px
                y1 = i * patch_size_px
                x2 = x1 + patch_size_px
                y2 = y1 + patch_size_px
                

                if mask_2d[i, j]:

                    color = retained_color + (int(255 * alpha),)
                else:

                    color = pruned_color + (int(255 * alpha),)
                
                draw.rectangle([x1, y1, x2, y2], fill=color)
        

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
        if original_image_shape is None:
            original_image_shape = self.num_patches_per_side * self.num_patches_per_side  # 576
        

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        retained_count = mask_np.sum()
        retention_rate = retained_count / original_image_shape * 100
        

        vis_image = self.mask_to_image_overlay(image, mask, original_image_shape=original_image_shape)
        

        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        

        # axes[0].imshow(image)
        # axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        # axes[0].axis('off')
        

        # axes[1].imshow(vis_image)
        # if topk_retained is not None:
        #     retention_text = f'Layer 2 Pruning\nRetention: {retention_rate:.1f}% (top-k: {topk_retained}/{original_image_shape})'
        # else:
        #     retention_text = f'Layer 2 Pruning\nRetention: {retention_rate:.1f}% ({int(retained_count)}/{original_image_shape})'
        # axes[1].set_title(retention_text, fontsize=12, fontweight='bold')
        # axes[1].axis('off')
        

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
    

