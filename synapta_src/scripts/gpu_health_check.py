#!/usr/bin/env python3
"""
GPU/CUDA Health Diagnostic

Comprehensive diagnostic to identify and suggest fixes for the
CUDA driver issue blocking the GC-LoRI pipeline.

Checks:
1. nvidia-smi availability and output
2. CUDA runtime libraries
3. PyTorch CUDA support
4. Driver version compatibility
5. Kernel module status
6. Device file permissions

Usage:
    python scripts/gpu_health_check.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_cmd(cmd: str, timeout: int = 10) -> tuple:
    """Run command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() + result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("GPU/CUDA HEALTH DIAGNOSTIC")
    print("=" * 60)
    issues = []
    fixes = []

    # 1. nvidia-smi
    print("\n[1/8] nvidia-smi...")
    if shutil.which("nvidia-smi"):
        ok, out = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
        if ok:
            print(f"  ✅ {out}")
        else:
            print(f"  ❌ nvidia-smi failed: {out[:200]}")
            issues.append("nvidia-smi installed but cannot communicate with driver")
            fixes.append("sudo modprobe nvidia")
            fixes.append("sudo systemctl restart nvidia-persistenced")
    else:
        print("  ❌ nvidia-smi not found")
        issues.append("NVIDIA driver not installed")
        fixes.append("sudo apt install nvidia-driver-570  # or latest version")

    # 2. NVIDIA kernel module
    print("\n[2/8] Kernel modules...")
    ok, out = run_cmd("lsmod | grep -i nvidia")
    if ok and out:
        modules = [line.split()[0] for line in out.split("\n") if line.strip()]
        print(f"  ✅ Loaded: {', '.join(modules[:5])}")
    else:
        print("  ❌ No NVIDIA kernel modules loaded")
        issues.append("NVIDIA kernel modules not loaded")
        fixes.append("sudo modprobe nvidia nvidia_uvm nvidia_modeset")

    # 3. Device files
    print("\n[3/8] Device files...")
    dev_files = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
    for df in dev_files:
        if os.path.exists(df):
            perms = oct(os.stat(df).st_mode)[-3:]
            print(f"  ✅ {df} (perms: {perms})")
        else:
            print(f"  ❌ {df} missing")
            issues.append(f"Device file {df} missing")
            fixes.append("sudo nvidia-smi  # triggers device file creation")

    # 4. CUDA libraries
    print("\n[4/8] CUDA libraries...")
    cuda_paths = [
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ]
    for cp in cuda_paths:
        exists = os.path.exists(cp) or len(list(Path(cp).parent.glob(Path(cp).name + "*"))) > 0
        if exists:
            print(f"  ✅ {cp}")
        else:
            print(f"  ⚠️  {cp} not found (may be in alternate location)")

    # Check LD_LIBRARY_PATH
    ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"  LD_LIBRARY_PATH: {ldpath or '(not set)'}")
    if "/usr/local/cuda" not in ldpath:
        fixes.append("export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")

    # 5. PyTorch CUDA
    print("\n[5/8] PyTorch CUDA...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"  ✅ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA version: {torch.version.cuda}")
            # Quick compute test
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            print(f"  ✅ Compute test passed: {y.shape}")
        else:
            print("  ❌ torch.cuda.is_available() == False")
            issues.append("PyTorch cannot see CUDA")
            
            # Check build info
            print(f"  CUDA built with: {torch.version.cuda}")
            
            # Try to get more diagnostic info
            try:
                print(f"  cudnn available: {torch.backends.cudnn.is_available()}")
            except:
                pass
    except ImportError:
        print("  ❌ PyTorch not installed")
        issues.append("PyTorch not installed")

    # 6. Driver version check
    print("\n[6/8] Driver compatibility...")
    ok, driver_version = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null")
    if ok and driver_version:
        print(f"  Driver: {driver_version}")
        try:
            import torch
            cuda_ver = torch.version.cuda
            print(f"  PyTorch CUDA: {cuda_ver}")
            # Basic compatibility check
            driver_major = int(driver_version.split(".")[0])
            cuda_major = int(cuda_ver.split(".")[0])
            if cuda_major >= 12 and driver_major < 525:
                print("  ⚠️  CUDA 12.x requires driver >= 525")
                issues.append("Driver too old for CUDA 12.x")
                fixes.append(f"sudo apt install nvidia-driver-{max(535, driver_major + 10)}")
            else:
                print("  ✅ Version compatibility OK")
        except:
            pass
    else:
        print("  ⚠️  Cannot determine driver version")

    # 7. Process holding GPU
    print("\n[7/8] GPU process check...")
    ok, out = run_cmd("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null")
    if ok and out:
        print(f"  Processes on GPU:\n{out}")
    else:
        print("  No GPU processes (or nvidia-smi unavailable)")

    # 8. Secure boot (common issue)
    print("\n[8/8] Secure Boot...")
    ok, out = run_cmd("mokutil --sb-state 2>/dev/null")
    if ok:
        print(f"  {out}")
        if "enabled" in out.lower():
            print("  ⚠️  Secure Boot is ENABLED — this can block NVIDIA kernel modules")
            issues.append("Secure Boot enabled")
            fixes.append("Disable Secure Boot in BIOS, or sign the NVIDIA kernel module")
    else:
        print("  Could not determine Secure Boot status")

    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✅ ALL CHECKS PASSED — GPU is healthy!")
        print("   You can proceed with: ./synapta_src/scripts/gc_lori_pipeline.sh")
    else:
        print(f"❌ FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print(f"\n💡 SUGGESTED FIXES:")
        seen = set()
        for fix in fixes:
            if fix not in seen:
                print(f"   $ {fix}")
                seen.add(fix)
        
        print("\n🔄 QUICK FIX SEQUENCE (try in order):")
        print("   1. sudo modprobe nvidia nvidia_uvm nvidia_modeset")
        print("   2. sudo nvidia-smi")
        print("   3. sudo systemctl restart nvidia-persistenced")
        print("   4. If still broken: sudo apt install --reinstall nvidia-driver-570")
        print("   5. Reboot and try again")
    
    print("=" * 60)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
