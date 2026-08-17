# -*- coding: utf-8 -*-
"""
年金保険料払込額運用モンテカルロシミュレーション
=====================================================

【目的】
毎月の厚生年金保険料払込額(標準報酬月額×18.3%)を、実際に納付する代わりに
GPIFの基本ポートフォリオ(国内債券25%/外国債券25%/国内株式25%/外国株式25%)で
運用した場合、65歳(運用終了年)時点でいくらになるかをモンテカルロ法で試算する。

【データソース】
- 毎月積立額:69,396円
  (東京都・標準報酬月額の平均379,213円[船員を除く・男女計] × 保険料率18.3%)
  出典:e-Stat「厚生年金保険・国民年金事業統計」厚生年金保険・国民年金事業月報
      (速報)令和8年 第1表(令和8年3月末現在)

- 資産配分・期待リターン・リスク・相関係数・乖離許容幅:
  GPIF「第5期中期目標期間における基本ポートフォリオについて ～詳細～」
  (2025年4月1日適用)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import japanize_matplotlib  # 日本語フォント設定(文字化け防止)

mpl.rcParams["axes.unicode_minus"] = False

# =====================================================================
# 1. パラメータ設定
# =====================================================================

# ---- 運用期間 ----------------------------------------------------
START_YEAR = 2026          # 運用開始年(自由に変更可)
END_YEAR   = 2056          # 運用終了年(自由に変更可。65歳到達年などを想定)
N_YEARS    = END_YEAR - START_YEAR
N_MONTHS   = N_YEARS * 12

# ---- 毎月の積立額(掛け金) -----------------------------------------
MONTHLY_CONTRIBUTION = 69396  # 円(東京都・標準報酬月額379,213円×18.3%)

# ---- モンテカルロ試行回数 ------------------------------------------
N_SIMULATIONS = 10000

# ---- 資産クラス(GPIF基本ポートフォリオの4資産) ----------------------
ASSET_NAMES = ["国内債券", "外国債券", "国内株式", "外国株式"]

# 基本ポートフォリオ:資産構成割合(目標配分)
TARGET_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])

# 期待リターン(年率):GPIF資料の4経済シナリオの単純平均を使用
# 高成長実現/成長型経済移行・継続/過去30年投影/1人当たりゼロ成長 の平均
SCENARIO_RETURNS = {
    "国内債券": [3.2, 2.7, 0.5, -0.3],
    "外国債券": [4.9, 4.4, 2.2, 1.4],
    "国内株式": [7.5, 7.0, 4.8, 4.0],
    "外国株式": [8.1, 7.6, 5.4, 4.6],
}
EXPECTED_RETURNS = np.array(
    [np.mean(SCENARIO_RETURNS[a]) / 100 for a in ASSET_NAMES]
)  # 年率(小数)

# リスク(標準偏差・年率):GPIF資料の値をそのまま使用
ANNUAL_STD = np.array([2.60, 9.72, 19.19, 20.35]) / 100

# 資産間の相関係数行列:GPIF資料の値をそのまま使用
CORR_MATRIX = np.array([
    [1.000,  0.073, -0.254, -0.125],
    [0.073,  1.000,  0.271,  0.560],
    [-0.254, 0.271,  1.000,  0.692],
    [-0.125, 0.560,  0.692,  1.000],
])

# ---- リバランスルール(乖離許容幅):GPIF基本ポートフォリオ資料より -----
# 各資産の乖離許容幅(国内債券/外国債券/国内株式/外国株式)
ASSET_TOLERANCE = np.array([0.06, 0.05, 0.06, 0.06])
# 債券合計(国内債券+外国債券)・株式合計(国内株式+外国株式)の乖離許容幅
BOND_TOLERANCE  = 0.09
STOCK_TOLERANCE = 0.09

# 乱数シード(結果を再現したい場合に固定。Noneなら毎回変わる)
RANDOM_SEED = 42

# =====================================================================
# 2. 月次リターンの前提を構築
# =====================================================================

def build_monthly_distribution(annual_returns, annual_std, corr_matrix):
    """年率の期待リターン・標準偏差・相関係数から、月次の平均・共分散行列を作る。"""
    monthly_mean = annual_returns / 12
    monthly_std = annual_std / np.sqrt(12)
    # 相関係数は年率・月率で同一と仮定し、月次の共分散行列を構築
    cov_monthly = np.outer(monthly_std, monthly_std) * corr_matrix
    return monthly_mean, cov_monthly


# =====================================================================
# 3. モンテカルロシミュレーション本体
# =====================================================================

def run_simulation(
    n_months,
    n_sims,
    monthly_contribution,
    target_weights,
    monthly_mean,
    cov_monthly,
    asset_tolerance,
    bond_tolerance,
    stock_tolerance,
    seed=None,
):
    """
    各シミュレーションパスについて、毎月:
      1) 前月末残高を月次リターンで運用
      2) 当月の積立額を目標配分どおりに追加
      3) 各資産の構成比が乖離許容幅を超えていればリバランス(目標配分に戻す)
    を繰り返し、月末残高の推移(履歴)を返す。

    戻り値:
      history: shape (n_months+1, n_sims) の合計評価額の推移
      values_final: shape (n_sims, 4) の最終時点の資産別評価額
      rebalance_count: shape (n_sims,) 各パスのリバランス発生回数
    """
    rng = np.random.default_rng(seed)

    n_assets = len(target_weights)

    # 各資産クラスの評価額(円) shape: (n_sims, n_assets)
    values = np.zeros((n_sims, n_assets))

    # 合計評価額の推移を記録(0ヶ月目=運用開始前)
    history = np.zeros((n_months + 1, n_sims))

    rebalance_count = np.zeros(n_sims, dtype=int)

    for t in range(1, n_months + 1):
        # ---- 1) 月次リターンを多変量正規分布からサンプリング ----
        monthly_returns = rng.multivariate_normal(
            mean=monthly_mean, cov=cov_monthly, size=n_sims
        )  # shape (n_sims, n_assets)

        # ---- 2) 前月末残高を運用(既存資産の値上がり・値下がり) ----
        values = values * (1.0 + monthly_returns)
        values = np.maximum(values, 0.0)  # 念のため評価額の下限を0にクリップ

        # ---- 3) 当月の積立額を目標配分どおりに追加 ----
        values += monthly_contribution * target_weights

        # ---- 4) リバランス判定 ----
        total = values.sum(axis=1, keepdims=True)
        total_safe = np.where(total == 0, 1e-12, total)
        weights = values / total_safe

        # 各資産の乖離
        dev_asset = np.abs(weights - target_weights)  # (n_sims, n_assets)
        trigger_asset = np.any(dev_asset > asset_tolerance, axis=1)

        # 債券合計(資産0,1)・株式合計(資産2,3)の乖離
        bond_weight = weights[:, 0] + weights[:, 1]
        stock_weight = weights[:, 2] + weights[:, 3]
        target_bond = target_weights[0] + target_weights[1]
        target_stock = target_weights[2] + target_weights[3]
        trigger_bond = np.abs(bond_weight - target_bond) > bond_tolerance
        trigger_stock = np.abs(stock_weight - target_stock) > stock_tolerance

        trigger = trigger_asset | trigger_bond | trigger_stock

        # トリガーが立ったパスのみ、目標配分に戻す(リバランス)
        if np.any(trigger):
            values[trigger] = total[trigger] * target_weights
            rebalance_count[trigger] += 1

        history[t] = values.sum(axis=1)

    return history, values, rebalance_count


# =====================================================================
# 4. 実行
# =====================================================================

def main():
    monthly_mean, cov_monthly = build_monthly_distribution(
        EXPECTED_RETURNS, ANNUAL_STD, CORR_MATRIX
    )

    print("=" * 60)
    print("年金保険料払込額 運用モンテカルロシミュレーション")
    print("=" * 60)
    print(f"運用期間: {START_YEAR}年 ～ {END_YEAR}年 ({N_YEARS}年間, {N_MONTHS}ヶ月)")
    print(f"毎月積立額: {MONTHLY_CONTRIBUTION:,}円")
    print(f"総積立元本: {MONTHLY_CONTRIBUTION * N_MONTHS:,}円")
    print(f"試行回数: {N_SIMULATIONS:,}回")
    print("-" * 60)
    print("資産配分(基本ポートフォリオ):")
    for name, w, r, s in zip(ASSET_NAMES, TARGET_WEIGHTS, EXPECTED_RETURNS, ANNUAL_STD):
        print(f"  {name}: 配分{w*100:.0f}% / 期待リターン{r*100:.2f}% / リスク{s*100:.2f}%")
    print("-" * 60)

    history, values_final, rebalance_count = run_simulation(
        n_months=N_MONTHS,
        n_sims=N_SIMULATIONS,
        monthly_contribution=MONTHLY_CONTRIBUTION,
        target_weights=TARGET_WEIGHTS,
        monthly_mean=monthly_mean,
        cov_monthly=cov_monthly,
        asset_tolerance=ASSET_TOLERANCE,
        bond_tolerance=BOND_TOLERANCE,
        stock_tolerance=STOCK_TOLERANCE,
        seed=RANDOM_SEED,
    )

    final_values = history[-1]  # 各パスの最終評価額
    principal = MONTHLY_CONTRIBUTION * N_MONTHS

    # ---- 統計サマリー ----
    percentiles = [5, 25, 50, 75, 95]
    stats = {p: np.percentile(final_values, p) for p in percentiles}

    print("\n【最終評価額(運用終了年時点)の分布】")
    print(f"  積立元本          : {principal:>15,.0f} 円")
    print(f"  平均値            : {final_values.mean():>15,.0f} 円")
    for p in percentiles:
        print(f"  {p:>3}パーセンタイル    : {stats[p]:>15,.0f} 円")
    print(f"  元本割れ確率      : {(final_values < principal).mean()*100:>6.1f} %")
    print(f"  平均リバランス回数: {rebalance_count.mean():>6.1f} 回 / {N_MONTHS}ヶ月")

    # ---- 結果をCSVに保存 ----
    summary_df = pd.DataFrame({
        "final_value": final_values,
        "rebalance_count": rebalance_count,
    })
    summary_df.to_csv("/mnt/user-data/outputs/simulation_results_raw.csv", index=False)

    stats_df = pd.DataFrame({
        "指標": ["積立元本", "平均値"] + [f"{p}パーセンタイル" for p in percentiles] + ["元本割れ確率(%)", "平均リバランス回数"],
        "値": [principal, final_values.mean()] + [stats[p] for p in percentiles]
              + [(final_values < principal).mean() * 100, rebalance_count.mean()],
    })
    stats_df.to_csv("/mnt/user-data/outputs/simulation_summary.csv", index=False, encoding="utf-8-sig")

    # =================================================================
    # 5. グラフ作成
    # =================================================================
    years_axis = np.arange(0, N_MONTHS + 1) / 12 + START_YEAR

    # 金額の表示単位:千円(1e7のような指数表記を避けるため)
    UNIT = 1_000
    UNIT_LABEL = "千円"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- (左) 資産推移パーセンタイルバンド ---
    ax = axes[0]
    p5 = np.percentile(history, 5, axis=1) / UNIT
    p25 = np.percentile(history, 25, axis=1) / UNIT
    p50 = np.percentile(history, 50, axis=1) / UNIT
    p75 = np.percentile(history, 75, axis=1) / UNIT
    p95 = np.percentile(history, 95, axis=1) / UNIT
    principal_path = (MONTHLY_CONTRIBUTION * np.arange(0, N_MONTHS + 1)) / UNIT

    ax.fill_between(years_axis, p5, p95, alpha=0.15, color="tab:blue", label="5-95%タイル")
    ax.fill_between(years_axis, p25, p75, alpha=0.30, color="tab:blue", label="25-75%タイル")
    ax.plot(years_axis, p50, color="tab:blue", lw=2, label="中央値")
    ax.plot(years_axis, principal_path, color="gray", lw=1.5, ls="--", label="積立元本(運用なし)")
    ax.set_xlabel("年")
    ax.set_ylabel(f"評価額({UNIT_LABEL})")
    ax.set_title("資産評価額の推移(モンテカルロ)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    # 指数表記(1e7等)を避け、桁区切りのカンマ付き通常表記にする
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:,.0f}"))

    # --- (右) 最終評価額のヒストグラム ---
    ax = axes[1]
    ax.hist(final_values / UNIT, bins=60, color="tab:blue", alpha=0.7)
    ax.axvline(principal / UNIT, color="gray", ls="--", lw=1.5, label="積立元本")
    ax.axvline(np.percentile(final_values, 50) / UNIT, color="tab:orange", lw=1.5, label="中央値")
    ax.set_xlabel(f"最終評価額({UNIT_LABEL})")
    ax.set_ylabel("試行数")
    ax.set_title(f"{END_YEAR}年時点の最終評価額の分布")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:,.0f}"))

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/simulation_chart.png", dpi=150)
    print("\nグラフを simulation_chart.png に保存しました。")
    print("詳細データを simulation_results_raw.csv / simulation_summary.csv に保存しました。")


if __name__ == "__main__":
    main()
