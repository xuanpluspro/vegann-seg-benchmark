"""Single-image eager PyTorch FP32 complexity and GPU forward timing."""
import platform
import statistics
import torch
from fvcore.nn import FlopCountAnalysis

@torch.inference_mode()
def benchmark_model(model, size=512, warmup=50, iterations=200):
    if warmup < 0 or iterations < 1: raise ValueError('Invalid benchmark repetitions')
    device=next(model.parameters()).device
    was_training=model.training
    cudnn_tf32=torch.backends.cudnn.allow_tf32
    matmul_tf32=torch.backends.cuda.matmul.allow_tf32
    cudnn_benchmark=torch.backends.cudnn.benchmark
    cudnn_deterministic=torch.backends.cudnn.deterministic
    result=dict(parameters=sum(p.numel() for p in model.parameters()),
                params_m=sum(p.numel() for p in model.parameters())/1e6,
                gflops=None,counted_gflops=None,flops_complete=False,
                unsupported_ops={},flops_error=None,latency_ms=None,
                latency_median_ms=None,latency_p95_ms=None,fps=None,
                input_shape=[1,3,size,size],precision='FP32 (TF32 disabled)',
                timing_scope='GPU model forward only; logits; no transfers/preprocessing/postprocessing',
                flops_convention='fvcore: one fused multiply-add counts as one operation',
                warmup=warmup,iterations=iterations,torch_version=torch.__version__,
                cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                python_version=platform.python_version(),device=str(device))
    try:
        model.eval()
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=False
        x=torch.zeros(1,3,size,size,device=device,dtype=torch.float32)
        with torch.autocast(device_type=device.type,enabled=False):
            try:
                analysis=FlopCountAnalysis(model,(x,))
                analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                total=analysis.total()/1e9
                unsupported=dict(analysis.unsupported_ops())
                result.update(counted_gflops=total,unsupported_ops=unsupported,
                              flops_complete=not unsupported,gflops=total if not unsupported else None)
            except Exception as exc:
                result['flops_error']=f'{type(exc).__name__}: {exc}'
            if device.type!='cuda':
                result['timing_error']='CUDA unavailable; GPU FPS not measured'
                return result
            result['gpu_name']=torch.cuda.get_device_name(device)
            result['gpu_total_memory_bytes']=torch.cuda.get_device_properties(device).total_memory
            for _ in range(warmup): model(x)
            torch.cuda.synchronize(device)
            starts=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends=[torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            with torch.cuda.device(device):
                for start,end in zip(starts,ends):
                    start.record(); model(x); end.record()
                torch.cuda.synchronize(device)
            durations=sorted(s.elapsed_time(e) for s,e in zip(starts,ends))
            mean=statistics.mean(durations)
            result.update(latency_ms=mean,latency_median_ms=statistics.median(durations),
                          latency_p95_ms=durations[min(iterations-1, int(.95*iterations))],
                          fps=1000/mean if mean>0 else None)
        return result
    finally:
        model.train(was_training)
        torch.backends.cudnn.allow_tf32=cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32=matmul_tf32
        torch.backends.cudnn.benchmark=cudnn_benchmark
        torch.backends.cudnn.deterministic=cudnn_deterministic
