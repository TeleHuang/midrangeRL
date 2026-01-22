# -*- coding: utf-8 -*-
import warp as wp
import torch

_WARP_INITIALIZED = False

def init_warp():
    """初始化 Warp"""
    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True
        print("[WarpUtils] Warp initialized successfully")

def get_warp_device(device_str: str):
    """获取 Warp 设备对象"""
    if device_str == 'cuda':
        # 默认使用第一个 CUDA 设备
        return wp.get_device('cuda:0')
    elif device_str == 'cpu':
        return wp.get_device('cpu')
    else:
        return wp.get_device(device_str)

def from_torch(t: torch.Tensor, dtype=None):
    """从 Torch Tensor 创建 Warp Array (Zero-copy)"""
    return wp.from_torch(t, dtype=dtype)

def to_torch(a: wp.array):
    """从 Warp Array 创建 Torch Tensor (Zero-copy)"""
    return wp.to_torch(a)
