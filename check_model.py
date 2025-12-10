import torch
import torch.nn as nn
import numpy as np
from config import get_default_config
from generator import build_generator

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def check_causality_and_streaming():
    print("\n" + "="*50)
    print("🧪 开始全面 Sanity Check")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. 初始化
    config = get_default_config()
    # 为了测试方便，减小模型规模
    config.model.hidden_channels = 32
    config.model.num_moe_layers = 2
    
    model = build_generator(config).to(device)
    model.eval()
    
    print(f"✅ 模型加载成功 | 参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. 模拟数据
    # 1秒 16kHz 音频 -> 16000 采样点
    # Batch size = 2
    B, C, L = 2, 1, 16000
    x = torch.randn(B, C, L).to(device)
    
    # 3. 测试 Forward (并行模式)
    print("\n[Step 1] 测试标准 Forward Pass (Parallel Mode)...")
    with torch.no_grad():
        y_parallel, aux_loss = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_parallel.shape}")
    print(f"Aux Loss: {aux_loss.item():.4f}")
    
    # 检查上采样倍率
    target_len = L * 3
    if y_parallel.shape[-1] == target_len:
        print(f"✅ 输出长度正确 ({target_len})")
    else:
        print(f"❌ 输出长度错误! 期望 {target_len}, 实际 {y_parallel.shape[-1]}")
        return

    # 检查输出范围 (Refiner 有 Tanh)
    min_val, max_val = y_parallel.min().item(), y_parallel.max().item()
    if min_val >= -1.01 and max_val <= 1.01: # 留一点浮点误差余量
        print("✅ 输出范围正确 [-1, 1]")
    else:
        print(f"❌ 输出范围异常: [{min_val:.4f}, {max_val:.4f}]")

    # 4. 测试流式推理 (Streaming Mode)
    print("\n[Step 2] 测试流式推理 (Chunk-by-Chunk)...")
    
    # 模拟流式：切分成小块 (例如 20ms = 320 点)
    # chunk_size = 320 # 真实场景
    chunk_size = 1600 # 稍微大点方便测试，必须能被 L 整除
    total_chunks = L // chunk_size
    
    buffer_dict = None # 初始化 buffer
    output_chunks = []
    
    try:
        with torch.no_grad():
            for i in range(total_chunks):
                chunk = x[:, :, i*chunk_size : (i+1)*chunk_size]
                
                # 调用流式接口
                out_chunk, buffer_dict = model.infer_stream(chunk, buffer_dict)
                output_chunks.append(out_chunk)
                
            y_stream = torch.cat(output_chunks, dim=-1)
            
        print(f"流式输出形状: {y_stream.shape}")
        
        # 5. 验证一致性 (Causality Check)
        # 流式处理的结果应该与一次性 Forward 的结果（几乎）完全一致
        
        # 对齐长度（如果有 padding 差异）
        min_len = min(y_parallel.shape[-1], y_stream.shape[-1])
        
        # 只比较中间部分，避开初始状态带来的差异（Initial transient）
        # 对于因果卷积，通常前几个点会有差异，因为 Forward 模式默认 pad 0，
        # 而 Streaming 模式第一块也 pad 0，理论上应该一致。
        diff = torch.abs(y_parallel[..., :min_len] - y_stream[..., :min_len])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"最大误差: {max_diff:.2e}")
        print(f"平均误差: {mean_diff:.2e}")
        
        if mean_diff < 1e-4: # 宽松一点
            print("✅ Streaming 一致性检查通过 (模型是严格因果的)")
        else:
            print("⚠️ Streaming 一致性检查警告: 误差较大，可能存在非因果操作或状态管理 Bug")
            
    except Exception as e:
        print(f"❌ 流式推理崩溃: {e}")
        import traceback
        traceback.print_exc()

    # 6. 梯度反向传播检查
    print("\n[Step 3] 梯度检查...")
    model.train()
    # 需要重新 forward 因为之前的是 no_grad
    x.requires_grad = True
    y_train, aux_loss = model(x)
    loss = y_train.mean() + aux_loss
    
    try:
        loss.backward()
        print("✅ 反向传播成功")
        
        # 检查是否有参数没有梯度 (Dead parameters)
        # 注意：Switch Transformer 的 aux loss 有时会导致未选中的 expert 无梯度，这在单次迭代中是正常的
        no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
        
        if len(no_grad_params) > 0:
            print(f"ℹ️  本次迭代有 {len(no_grad_params)} 个参数没有梯度 (MoE 中未被选中的 Expert 无梯度属正常现象):")
            # for name in no_grad_params[:5]:
            #     print(f"  - {name}")
        else:
            print("✅ 所有参数都有梯度")
            
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")

if __name__ == "__main__":
    set_seed()
    check_causality_and_streaming()

