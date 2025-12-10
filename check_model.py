"""
FASS-MoE Model Sanity Check Script.

验证模型的:
1. Forward pass 输出形状和范围
2. Streaming 和 Forward 的一致性 (使用 RMSNorm + WeightNorm)
3. 梯度传播
"""

import torch
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    print("\n" + "=" * 60)
    print("🧪 FASS-MoE Model Sanity Check")
    print("=" * 60)
    
    # Import inside function to avoid issues during error reporting
    from config import get_default_config
    from generator import build_generator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Use DEFAULT config to check real parameter count
    config = get_default_config()
    # config.model.hidden_channels = 32 # Commented out to see real size
    # config.model.num_moe_layers = 2   # Commented out to see real size
    
    set_seed(42)
    model = build_generator(config).to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Model loaded | Parameters: {param_count:.2f}M")
    
    # Test input
    B, C, L = 1, 1, 16000  # 1 second at 16kHz (Batch 1 to save memory)
    x = torch.randn(B, C, L, device=device)
    
    # =========================================
    # [Step 1] Forward Pass
    # =========================================
    print("\n" + "-" * 40)
    print("[Step 1] Testing Forward Pass (Parallel Mode)")
    print("-" * 40)
    
    set_seed(42)
    with torch.no_grad():
        y_forward, aux_loss = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y_forward.shape}")
    print(f"Aux Loss:     {aux_loss.item():.4f}")
    
    expected_len = L * 3  # 3x upsampling
    if y_forward.shape[-1] == expected_len:
        print(f"✅ Output length correct ({expected_len})")
    else:
        print(f"❌ Output length wrong! Expected {expected_len}, got {y_forward.shape[-1]}")
        return
    
    out_min, out_max = y_forward.min().item(), y_forward.max().item()
    if -1.0 <= out_min and out_max <= 1.0:
        print(f"✅ Output range correct: [{out_min:.4f}, {out_max:.4f}]")
    else:
        print(f"⚠️  Output range warning: [{out_min:.4f}, {out_max:.4f}]")
        print("   (Expected [-1, 1] due to tanh)")
    
    # =========================================
    # [Step 2] Streaming Consistency
    # =========================================
    print("\n" + "-" * 40)
    print("[Step 2] Testing Streaming Consistency")
    print("-" * 40)
    
    chunk_size = 1600  # 100ms chunks at 16kHz
    num_chunks = L // chunk_size
    
    print(f"Processing {num_chunks} chunks of {chunk_size} samples each")
    
    set_seed(42)
    state = None
    output_chunks = []
    
    try:
        with torch.no_grad():
            for i in range(num_chunks):
                chunk = x[:, :, i * chunk_size : (i + 1) * chunk_size]
                out_chunk, state = model.infer_stream(chunk, state)
                output_chunks.append(out_chunk)
            
            y_stream = torch.cat(output_chunks, dim=-1)
        
        print(f"Stream output shape: {y_stream.shape}")
        
        # Compare
        min_len = min(y_forward.shape[-1], y_stream.shape[-1])
        diff = torch.abs(y_forward[..., :min_len] - y_stream[..., :min_len])
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Max difference:  {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")
        
        # Strict threshold for streaming consistency
        if mean_diff < 1e-5:
            print("✅ Streaming PERFECTLY consistent with forward()")
        elif mean_diff < 1e-3:
            print("✅ Streaming consistent (minor numerical differences)")
        elif mean_diff < 1e-2:
            print("⚠️  Streaming mostly consistent, small differences")
        else:
            print("❌ Streaming INCONSISTENT with forward()")
            
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================
    # [Step 3] Gradient Check
    # =========================================
    print("\n" + "-" * 40)
    print("[Step 3] Testing Gradient Propagation")
    print("-" * 40)
    
    model.train()
    y_train, aux_loss = model(x)
    loss = y_train.mean() + aux_loss
    
    try:
        loss.backward()
        print("✅ Backward pass successful")
        
        no_grad_params = [name for name, p in model.named_parameters() if p.grad is None]
        if len(no_grad_params) == 0:
            print("✅ All parameters have gradients")
        else:
            print(f"⚠️  {len(no_grad_params)} parameters without gradients:")
            
    except Exception as e:
        print(f"❌ Backward failed: {e}")
        return
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Model Parameters: {param_count:.2f}M")
    print(f"Streaming Error:  {mean_diff:.2e}")
    print(f"Output Range:     [{out_min:.4f}, {out_max:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
