#!/usr/bin/env python3
import sys, re, numpy as np
from pathlib import Path

def load_errors(path: Path):
    txt = path.read_text(encoding='utf-8', errors='ignore')
    # 파일에 숫자 외 텍스트가 섞여있어도 모든 실수 추출
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', txt)
    arr = np.array([float(x) for x in nums], dtype=float)
    return arr

def summarize(arr: np.ndarray):
    if arr.size == 0:
        return None
    signed_mean = float(np.mean(arr))
    signed_var  = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0

    abs_arr = np.abs(arr)
    abs_mean = float(np.mean(abs_arr))
    abs_var  = float(np.var(abs_arr, ddof=1)) if arr.size > 1 else 0.0
    return signed_mean, np.sqrt(signed_var), abs_mean, np.sqrt(abs_var), int(arr.size)

def main():
    # 인자 없으면 기본 파일 이름 사용
    files = sys.argv[1:] or [
        "obb_angle_errors_var1.txt",
        "obb_angle_errors_var2.txt",
        "obb_angle_errors_var3.txt",
        "obb_angle_errors_var4.txt",
    ]

    print("== Angle Error Statistics ==")
    print("(variance = sample variance, ddof=1)")
    print(f"{'file':<22} {'N':>6} {'mean':>12} {'var':>12} {'|mean|':>12} {'|var|':>12}")

    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"{f:<22} {'-':>6} {'(missing)':>12}")
            continue

        arr = load_errors(p)
        stats = summarize(arr)
        if stats is None:
            print(f"{f:<22} {0:>6} {'nan':>12} {'nan':>12} {'nan':>12} {'nan':>12}")
        else:
            mean, var, amean, avar, n = stats
            print(f"{f:<22} {n:>6d} {mean:>12.4f} {var:>12.4f} {amean:>12.4f} {avar:>12.4f}")

if __name__ == "__main__":
    main()
