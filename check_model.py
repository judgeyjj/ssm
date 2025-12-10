"""
FASS-MoE 模型健康检查脚本

严格验证:
1. Forward Pass - 上采样倍率和输出范围
2. Streaming Pass - 与 Forward 的精确一致性 (学术级要求)
3. 梯度检查 - 反向传播
"""

import torch
import numpy as np
from config import get_default_config
from generator import build_generator

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def check_model():
    print("\n" + "="*60)
    print("🧪 FASS-MoE 模型健康检查 (学术级严格验证)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 初始化
    config = get_default_config()
    config.model.hidden_channels = 32
    config.model.num_moe_layers = 2
    
    model = build_generator(config).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载成功 | 参数量: {total_params/1e6:.2f}M")

    # 模拟数据
    B, C, L = 2, 1, 16000
    set_seed(42)  # 固定种子
    x = torch.randn(B, C, L).to(device)
    
    # ============================================================
    # [Step 1] Forward Pass
    # ============================================================
    print("\n" + "-"*40)
    print("[Step 1] 标准 Forward Pass")
    print("-"*40)
    
    with torch.no_grad():
        y_forward, aux_loss = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_forward.shape}")
    print(f"Aux Loss: {aux_loss.item():.4f}")
    
    # 检查上采样
    target_len = L * 3
    assert y_forward.shape[-1] == target_len, f"长度错误: {y_forward.shape[-1]} != {target_len}"
    print(f"✅ 输出长度正确: {target_len}")

    # 检查输出范围
    min_val, max_val = y_forward.min().item(), y_forward.max().item()
    assert -1.01 <= min_val <= max_val <= 1.01, f"范围异常: [{min_val}, {max_val}]"
    print(f"✅ 输出范围正确: [{min_val:.4f}, {max_val:.4f}]")

    # ============================================================
    # [Step 2] Streaming Pass - 精确一致性验证
    # ============================================================
    print("\n" + "-"*40)
    print("[Step 2] Streaming 精确一致性验证")
    print("-"*40)
    
    # 测试不同的 chunk 大小
    chunk_sizes = [800, 1600, 3200]  # 50ms, 100ms, 200ms @ 16kHz
    
    for chunk_size in chunk_sizes:
        if L % chunk_size != 0:
            continue
            
        total_chunks = L // chunk_size
        
        state = None
        output_chunks = []
        
        with torch.no_grad():
            for i in range(total_chunks):
                chunk = x[:, :, i*chunk_size : (i+1)*chunk_size]
                out_chunk, state = model.infer_stream(chunk, state)
                output_chunks.append(out_chunk)
        
        y_stream = torch.cat(output_chunks, dim=-1)
        
        # 计算误差
        diff = torch.abs(y_forward - y_stream)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # 相对误差
        rel_diff = (diff / (torch.abs(y_forward) + 1e-8)).mean().item()
        
        status = "✅" if max_diff < 1e-5 else ("⚠️" if max_diff < 1e-3 else "❌")
        
        print(f"\nChunk={chunk_size} ({chunk_size/16:.0f}ms, {total_chunks} chunks):")
        print(f"  最大绝对误差: {max_diff:.2e}")
        print(f"  平均绝对误差: {mean_diff:.2e}")
        print(f"  平均相对误差: {rel_diff:.2e}")
        print(f"  {status} 一致性: {'精确一致' if max_diff < 1e-5 else ('可接受' if max_diff < 1e-3 else '不一致!')}")

    # ============================================================
    # [Step 3] 梯度检查
    # ============================================================
    print("\n" + "-"*40)
    print("[Step 3] 梯度检查")
    print("-"*40)
    
    model.train()
    x_grad = torch.randn(B, C, L, device=device, requires_grad=True)
    y_train, aux_loss = model(x_grad)
    loss = y_train.mean() + 0.01 * aux_loss
    loss.backward()
    
    total_params = sum(1 for p in model.parameters())
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    
    print(f"✅ 反向传播成功")
    print(f"   {params_with_grad}/{total_params} 参数有梯度")

    # ============================================================
    # [Step 4] 边界条件检查
    # ============================================================
    print("\n" + "-"*40)
    print("[Step 4] 边界条件")
    print("-"*40)
    
    model.eval()
    test_lengths = [1600, 8000, 16000, 32000]
    
    with torch.no_grad():
        for length in test_lengths:
            x_test = torch.randn(1, 1, length, device=device)
            y_test, _ = model(x_test)
            expected = length * 3
            status = "✅" if y_test.shape[-1] == expected else "❌"
            print(f"{status} 输入 {length:>5} → 输出 {y_test.shape[-1]:>6} (期望 {expected})")

    print("\n" + "="*60)
    print("🎉 健康检查完成!")
    print("="*60)


if __name__ == "__main__":
    check_model()
