"""
Numba JIT 加速的向量化因子运算核心

设计目标：
- 输入输出均为 2D numpy.ndarray (T x N)，T=时间步, N=股票数
- 利用 @njit 编译，绕开 Python GIL 与类型解释开销
- 对标 vectorbt 的 Numba backend：常用算子纯 JIT
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

# 让 numba 在第一次调用时及时编译
@njit(cache=True, fastmath=True, parallel=True)
def numba_ma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    简单移动平均 - 逐列计算
    
    参数:
        arr: (T, N) 2D array, NaN 表示缺失
        window: 窗口大小
    
    返回:
        (T, N) array, 前 window-1 行填 NaN
    """
    T, N = arr.shape
    out = np.full_like(arr, np.nan)
    for n in prange(N):
        s = 0.0
        cnt = 0
        for t in range(T):
            v = arr[t, n]
            if not np.isnan(v):
                s += v
                cnt += 1
            if t >= window:
                old = arr[t - window, n]
                if not np.isnan(old):
                    s -= old
                    cnt -= 1
            if cnt > 0:
                out[t, n] = s / cnt
    return out


@njit(cache=True, fastmath=True, parallel=True)
def numba_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """同 numba_ma，作为独立 API 暴露"""
    return numba_ma(arr, window)


@njit(cache=True, fastmath=True, parallel=True)
def numba_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    滚动标准差 - Welford 在线算法 + 滑动窗口
    """
    T, N = arr.shape
    out = np.full_like(arr, np.nan)
    for n in prange(N):
        # 维护一个简单环形 buffer + 当前窗口的 sum/sumsq
        buf = np.full(window, np.nan, dtype=np.float64)
        idx = 0
        s = 0.0
        ssq = 0.0
        cnt = 0
        for t in range(T):
            v = arr[t, n]
            old = buf[idx]
            buf[idx] = v
            idx = (idx + 1) % window
            if not np.isnan(v):
                s += v
                ssq += v * v
                cnt += 1
            if not np.isnan(old):
                s -= old
                ssq -= old * old
                cnt -= 1
            if cnt >= 2:
                m = s / cnt
                var = ssq / cnt - m * m
                if var < 0:
                    var = 0.0
                out[t, n] = np.sqrt(var)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def numba_ema(arr: np.ndarray, window: int) -> np.ndarray:
    """
    指数移动平均 - 简单 alpha = 2/(window+1)
    """
    T, N = arr.shape
    out = np.full_like(arr, np.nan)
    alpha = 2.0 / (window + 1)
    for n in prange(N):
        prev = np.nan
        for t in range(T):
            v = arr[t, n]
            if np.isnan(v):
                continue
            if np.isnan(prev):
                prev = v
            else:
                prev = alpha * v + (1 - alpha) * prev
            out[t, n] = prev
    return out


@njit(cache=True, fastmath=True, parallel=True)
def numba_rsi(arr: np.ndarray, window: int = 14) -> np.ndarray:
    """
    相对强弱指标 (RSI)
    使用 Wilder 平滑: avg_gain/avg_loss
    """
    T, N = arr.shape
    out = np.full_like(arr, np.nan)
    for n in prange(N):
        prev = np.nan
        avg_gain = 0.0
        avg_loss = 0.0
        # 累计 window 个 delta
        gains_buf = np.full(window, 0.0, dtype=np.float64)
        losses_buf = np.full(window, 0.0, dtype=np.float64)
        idx = 0
        cnt = 0
        for t in range(T):
            v = arr[t, n]
            if np.isnan(prev) or np.isnan(v):
                out[t, n] = np.nan
                continue
            delta = v - prev
            g = delta if delta > 0 else 0.0
            l = -delta if delta < 0 else 0.0
            if cnt < window:
                gains_buf[idx] = g
                losses_buf[idx] = l
                idx = (idx + 1) % window
                cnt += 1
                if cnt == window:
                    avg_gain = gains_buf.sum() / window
                    avg_loss = losses_buf.sum() / window
            else:
                avg_gain = (avg_gain * (window - 1) + g) / window
                avg_loss = (avg_loss * (window - 1) + l) / window
            if avg_loss == 0:
                out[t, n] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[t, n] = 100.0 - 100.0 / (1.0 + rs)
            prev = v
    return out


@njit(cache=True, fastmath=True, parallel=True)
def numba_rolling_corr(
    a: np.ndarray, b: np.ndarray, window: int
) -> np.ndarray:
    """
    滚动相关系数 - 同时维护 sum/sum2/sumb/sumb2/sumab
    """
    T, N = a.shape
    out = np.full_like(a, np.nan)
    for n in prange(N):
        s1 = 0.0
        s2 = 0.0
        sb1 = 0.0
        sb2 = 0.0
        sab = 0.0
        cnt = 0
        for t in range(T):
            v1 = a[t, n]
            v2 = b[t, n]
            if t >= window:
                ov1 = a[t - window, n]
                ov2 = b[t - window, n]
                if not np.isnan(ov1) and not np.isnan(ov2):
                    s1 -= ov1
                    s2 -= ov1 * ov1
                    sb1 -= ov2
                    sb2 -= ov2 * ov2
                    sab -= ov1 * ov2
                    cnt -= 1
            if not np.isnan(v1) and not np.isnan(v2):
                s1 += v1
                s2 += v1 * v1
                sb1 += v2
                sb2 += v2 * v2
                sab += v1 * v2
                cnt += 1
            if cnt >= 2:
                m1 = s1 / cnt
                m2 = sb1 / cnt
                v1_ = s2 / cnt - m1 * m1
                v2_ = sb2 / cnt - m2 * m2
                cv = sab / cnt - m1 * m2
                if v1_ > 0 and v2_ > 0:
                    out[t, n] = cv / np.sqrt(v1_ * v2_)
                else:
                    out[t, n] = np.nan
    return out


@njit(cache=True, fastmath=True, parallel=True)
def numba_cross_section_rank(arr: np.ndarray) -> np.ndarray:
    """
    截面排名 - 每个时间点对所有股票做 %rank
    """
    T, N = arr.shape
    out = np.full_like(arr, np.nan)
    for t in prange(T):
        # 提取非 NaN
        valid = 0
        for n in range(N):
            if not np.isnan(arr[t, n]):
                valid += 1
        if valid == 0:
            continue
        # 简单排序计数
        for n in range(N):
            v = arr[t, n]
            if np.isnan(v):
                continue
            cnt_less = 0
            cnt_eq = 0
            for m in range(N):
                w = arr[t, m]
                if np.isnan(w):
                    continue
                if w < v:
                    cnt_less += 1
                elif w == v:
                    cnt_eq += 1
            out[t, n] = (cnt_less + 0.5 * cnt_eq) / valid
    return out
