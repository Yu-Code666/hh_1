import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os

# ---------- 0) 参数 ----------
WINDOW_NS = np.int64(4 * 3600 * 10**9)  # 4 小时窗口（纳秒）

# ---------- 1) 多进程初始化 ----------
def init_pool(grouped_arrays):
    """给子进程初始化轻量只读数据：{segment: (ts_ns_sorted, prices_sorted)}"""
    global GROUPED
    GROUPED = grouped_arrays

def process_row(task):
    """
    task: (row_idx, segment, dep_ts_ns, unit_price_hh)
    返回: (row_idx, competitor_price)
    """
    row_idx, seg, t_ns, unit_price_hh = task
    pair = GROUPED.get(seg)
    if pair is None:
        return row_idx, unit_price_hh

    ts, prices = pair
    if ts.size == 0:
        return row_idx, unit_price_hh

    # 二分查找 [t-4h, t+4h]
    L = np.searchsorted(ts, t_ns - WINDOW_NS, side='left')
    R = np.searchsorted(ts, t_ns + WINDOW_NS, side='right')

    # 1) 窗口内有数据
    if L < R:
        return row_idx, float(prices[L:R].min())

    # 2) 窗口为空 -> 取最近
    cand_idx = []
    if L - 1 >= 0:
        cand_idx.append(L - 1)
    if R < ts.size:
        cand_idx.append(R)
    if not cand_idx:
        return row_idx, unit_price_hh

    diffs = [abs(ts[i] - t_ns) for i in cand_idx]
    best = cand_idx[int(np.argmin(diffs))]
    return row_idx, float(prices[best])

# ---------- 2) 主逻辑 ----------
if __name__ == "__main__":
    mp.freeze_support()
    print("🚀 启动多进程竞争价格计算 ...")

    # ---------- 读取缓存 ----------
    df_all_path = "D:/haihang/cache/df_all_cache.csv"
    df_hh_path  = "D:/haihang/cache/df_hh_cache.csv"
    if not (os.path.exists(df_all_path) and os.path.exists(df_hh_path)):
        raise FileNotFoundError("❌ 未找到缓存数据，请先在 Notebook 中运行保存缓存步骤！")

    # 只读取必要列用于计算
    calc_cols = ["segment", "flt_date", "dep_time", "unit_price"]
    df_all = pd.read_csv(df_all_path, usecols=calc_cols)
    df_hh_calc  = pd.read_csv(df_hh_path,  usecols=calc_cols)

    # 同时保留完整 df_hh 用于后面合并
    df_hh_full = pd.read_csv(df_hh_path)
    print(f"✅ 成功读取 df_all {df_all.shape} 和 df_hh {df_hh_calc.shape}")

    # ---------- 时间解析 ----------
    # 统一拼接格式 "YYYY-MM-DD HH:MM:SS"
    df_all["dep_time"] = pd.to_datetime(
        df_all["flt_date"].astype(str) + " " + df_all["dep_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df_hh_calc["dep_time"] = pd.to_datetime(
        df_hh_calc["flt_date"].astype(str) + " " + df_hh_calc["dep_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # ---------- 清洗 ----------
    df_all["unit_price"] = pd.to_numeric(df_all["unit_price"], errors="coerce")
    df_hh_calc["unit_price"] = pd.to_numeric(df_hh_calc["unit_price"], errors="coerce")

    df_all = df_all.dropna(subset=["dep_time", "unit_price"])
    df_hh_calc = df_hh_calc.dropna(subset=["dep_time", "unit_price"])

    # 转换为纳秒时间戳
    df_all["ts_ns"] = df_all["dep_time"].view("int64")
    df_hh_calc["ts_ns"] = df_hh_calc["dep_time"].view("int64")

    # ---------- 分组 ----------
    grouped = {}
    for seg, g in df_all.groupby("segment", sort=False):
        g2 = g[["ts_ns", "unit_price"]].to_numpy()
        order = np.argsort(g2[:, 0], kind="mergesort")
        ts_sorted = g2[order, 0].astype(np.int64, copy=False)
        pr_sorted = g2[order, 1].astype(float,   copy=False)
        grouped[seg] = (ts_sorted, pr_sorted)
    print(f"📦 segment 分组就绪：{len(grouped)} 组")

    # ---------- 构造任务 ----------
    cols = ["segment", "unit_price", "ts_ns"]
    tasks = [
        (i, seg, np.int64(t_ns), float(price))
        for i, (seg, price, t_ns) in enumerate(
            df_hh_calc[cols].itertuples(index=False, name=None)
        )
    ]

    cpu = mp.cpu_count()
    chunksize = max(1, len(tasks) // (cpu * 8))

    # ---------- 并行 ----------
    with mp.Pool(processes=cpu, initializer=init_pool, initargs=(grouped,)) as pool:
        results = list(tqdm(pool.imap(process_row, tasks, chunksize=chunksize), total=len(tasks)))

    # ---------- 回填结果 ----------
    comp = np.empty(len(df_hh_calc), dtype=float)
    for idx, val in results:
        comp[idx] = val
    df_hh_calc["competitor_price"] = comp

    # ---------- 合并回完整数据 ----------
    # 确保 merge 键类型一致
    # 确保 merge 键类型一致
    df_hh_full["dep_time"] = pd.to_datetime(
            df_hh_full["flt_date"].astype(str).str.strip() + " " + df_hh_full["dep_time"].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
    )
    df_hh_full["dep_time"] = df_hh_full["dep_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df_hh_calc["dep_time"] = df_hh_calc["dep_time"].dt.strftime("%Y-%m-%d %H:%M:%S")


    df_result = pd.merge(
        df_hh_full,
        df_hh_calc[["segment", "flt_date", "dep_time", "competitor_price"]],
        on=["segment", "flt_date", "dep_time"],
        how="left"
    )

    # ---------- 保存 ----------
    out = "D:/haihang/result/data_with_competitor_prices.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_result.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ 计算完成，结果已保存到：{out}")