"""
FFT 频谱和功率谱可视化工具
用于分析不同注意力头策略下的 relation_vis 的频域特征
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os


class FFTVisualizer:
    """FFT 频谱和功率谱可视化器"""
    
    def __init__(self):
        """初始化 FFT 可视化器"""
        pass
    
    def compute_fft(self, relation_vis: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 relation_vis 的 FFT 频谱和功率谱
        
        Args:
            relation_vis: 形状为 [B, L] 或 [L] 的张量，表示视觉token的注意力分数
            
        Returns:
            spectrum: 频谱（复数）
            power_spectrum: 功率谱（实数）
        """
        # 转换为 numpy 并确保是1D
        if isinstance(relation_vis, torch.Tensor):
            relation_vis_np = relation_vis.cpu().numpy()
        else:
            relation_vis_np = relation_vis
        
        # 如果是2D，取第一个batch
        if relation_vis_np.ndim == 2:
            relation_vis_np = relation_vis_np[0]
        
        # 确保是1D
        relation_vis_1d = relation_vis_np.flatten()
        
        # 计算 FFT
        fft_result = np.fft.fft(relation_vis_1d)
        
        # 计算频谱（幅度）
        spectrum = np.abs(fft_result)
        
        # 计算功率谱（频谱的平方）
        power_spectrum = spectrum ** 2
        
        return spectrum, power_spectrum
    
    def visualize_spectrum(
        self,
        spectrum: np.ndarray,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        head_id: Optional[int] = None,
        is_average: bool = False
    ) -> None:
        """
        可视化频谱
        
        Args:
            spectrum: 频谱数组
            save_path: 保存路径
            title: 图表标题
            head_id: 注意力头ID（如果适用）
            is_average: 是否为平均策略
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算频率轴
        n = len(spectrum)
        freqs = np.fft.fftfreq(n)
        
        # 只显示正频率部分
        positive_freqs = freqs[:n//2]
        positive_spectrum = spectrum[:n//2]
        
        ax.plot(positive_freqs, positive_spectrum, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif is_average:
            ax.set_title('FFT Spectrum - Average Strategy', fontsize=14, fontweight='bold')
        elif head_id is not None:
            ax.set_title(f'FFT Spectrum - Head {head_id}', fontsize=14, fontweight='bold')
        else:
            ax.set_title('FFT Spectrum', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)  # 只显示到奈奎斯特频率
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def visualize_power_spectrum(
        self,
        power_spectrum: np.ndarray,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        head_id: Optional[int] = None,
        is_average: bool = False
    ) -> None:
        """
        可视化功率谱
        
        Args:
            power_spectrum: 功率谱数组
            save_path: 保存路径
            title: 图表标题
            head_id: 注意力头ID（如果适用）
            is_average: 是否为平均策略
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算频率轴
        n = len(power_spectrum)
        freqs = np.fft.fftfreq(n)
        
        # 只显示正频率部分
        positive_freqs = freqs[:n//2]
        positive_power = power_spectrum[:n//2]
        
        ax.plot(positive_freqs, positive_power, linewidth=1.5, alpha=0.8, color='red')
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        elif is_average:
            ax.set_title('FFT Power Spectrum - Average Strategy', fontsize=14, fontweight='bold')
        elif head_id is not None:
            ax.set_title(f'FFT Power Spectrum - Head {head_id}', fontsize=14, fontweight='bold')
        else:
            ax.set_title('FFT Power Spectrum', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)  # 只显示到奈奎斯特频率
        ax.set_yscale('log')  # 使用对数刻度，因为功率谱通常跨度很大
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def visualize_fft_comparison(
        self,
        spectra_dict: dict,
        power_spectra_dict: dict,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        可视化多个注意力头的频谱和功率谱对比
        
        Args:
            spectra_dict: {head_id: spectrum} 或 {'average': spectrum}
            power_spectra_dict: {head_id: power_spectrum} 或 {'average': power_spectrum}
            save_path: 保存路径
            title: 图表标题
        """
        num_heads = len(spectra_dict)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 计算频率轴（假设所有频谱长度相同）
        first_spectrum = list(spectra_dict.values())[0]
        n = len(first_spectrum)
        freqs = np.fft.fftfreq(n)
        positive_freqs = freqs[:n//2]
        
        # 绘制频谱对比
        ax1 = axes[0]
        for head_id, spectrum in spectra_dict.items():
            positive_spectrum = spectrum[:n//2]
            label = f'Head {head_id}' if isinstance(head_id, int) else 'Average'
            ax1.plot(positive_freqs, positive_spectrum, label=label, alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('FFT Spectrum Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.5)
        
        # 绘制功率谱对比
        ax2 = axes[1]
        for head_id, power_spectrum in power_spectra_dict.items():
            positive_power = power_spectrum[:n//2]
            label = f'Head {head_id}' if isinstance(head_id, int) else 'Average'
            ax2.plot(positive_freqs, positive_power, label=label, alpha=0.7, linewidth=1.5)
        
        ax2.set_xlabel('Frequency', fontsize=12)
        ax2.set_ylabel('Power (log scale)', fontsize=12)
        ax2.set_title('FFT Power Spectrum Comparison', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 0.5)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

