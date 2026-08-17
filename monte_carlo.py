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
from matplotlib.collections import LineCollection
import japanize_matplotlib  # 日本語フォント設定(文字化け防止)
from scipy.optimize import brentq

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

# ---- 診断用:月次で動いていることを可視化するため、先頭N本のパスの
#      資産別構成比の推移を記録する数(0にすると記録しない) ----------
TRACK_SAMPLE_PATHS = 3

# ---- 積立額の影響を除いた「運用のみ」の累積収益率を、全試行分重ねて
#      描画するために、全パスの月次収益率を記録するかどうか ----------
TRACK_ALL_RETURNS = True

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

def compute_annualized_returns(final_values, monthly_contribution, n_months):
    """
    各シミュレーションパスについて、毎月一定額を積み立てた場合の
    money-weighted return(内部収益率, IRR)を年率換算で求める。

    積立の漸化式は V_t = V_{t-1}*(1+r) + c (r:月次IRR) なので、
    n ヶ月後の評価額は年金終価(annuity-immediate)の公式:
        V_n = c * ((1+r)^n - 1) / r   (r ≠ 0の場合)
        V_n = c * n                    (r = 0の場合)
    となる。各パスについて、この式を満たす月次 r を数値的に解き、
    年率 (1+r)^12 - 1 に変換する。

    戻り値: shape (n_sims,) の年率リターン(小数)
    """
    c = monthly_contribution
    n = n_months

    def annuity_value(r):
        if abs(r) < 1e-12:
            return c * n
        return c * ((1.0 + r) ** n - 1.0) / r

    annual_returns = np.empty_like(final_values)
    for i, v in enumerate(final_values):
        f = lambda r: annuity_value(r) - v
        try:
            # 月次IRRの探索範囲: -50%〜+50%(年率にするとかなり広い範囲をカバー)
            r_monthly = brentq(f, -0.5, 0.5, xtol=1e-10)
        except ValueError:
            r_monthly = np.nan
        annual_returns[i] = (1.0 + r_monthly) ** 12 - 1.0 if not np.isnan(r_monthly) else np.nan

    return annual_returns


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
    track_paths=0,
    track_all_returns=False,
):
    """
    各シミュレーションパスについて、毎月:
      1) 前月末残高を月次リターンで運用
      2) 当月の積立額を目標配分どおりに追加
      3) 各資産の構成比が乖離許容幅を超えていればリバランス(目標配分に戻す)
    を繰り返し、月末残高の推移(履歴)を返す。

    引数:
      track_paths: 0より大きい場合、先頭 track_paths 本のパスについて
                   毎月の資産別構成比(weights)の推移を記録して返す
                   (月次で動いていることを可視化するための診断用)
      track_all_returns: Trueの場合、全 n_sims 本について毎月のポートフォリオ
                   収益率(%)を記録して返す(積立額の影響を除いた「運用のみ」
                   の累積収益率を、全試行分重ねて描画するための元データ)

    戻り値:
      history: shape (n_months+1, n_sims) の合計評価額の推移
      values_final: shape (n_sims, 4) の最終時点の資産別評価額
      rebalance_count: shape (n_sims,) 各パスのリバランス発生回数
      weight_history: shape (n_months+1, track_paths, 4) の構成比推移
                       (track_paths=0 の場合は None)
      return_history: shape (n_months+1, track_paths) のポートフォリオ月次収益率
                       (track_paths=0 の場合は None)
      return_history_all: shape (n_months+1, n_sims) の全試行分の月次収益率
                       (track_all_returns=False の場合は None)
    """
    rng = np.random.default_rng(seed)

    n_assets = len(target_weights)

    # 各資産クラスの評価額(円) shape: (n_sims, n_assets)
    values = np.zeros((n_sims, n_assets))

    # 合計評価額の推移を記録(0ヶ月目=運用開始前)
    history = np.zeros((n_months + 1, n_sims))

    rebalance_count = np.zeros(n_sims, dtype=int)

    weight_history = None
    return_history = None
    if track_paths > 0:
        weight_history = np.full((n_months + 1, track_paths, n_assets), np.nan)
        weight_history[0] = target_weights  # 初月は目標配分でスタート
        # ポートフォリオの月次収益率(%)。運用開始月(残高0)はNaNとする。
        return_history = np.full((n_months + 1, track_paths), np.nan)

    return_history_all = None
    if track_all_returns:
        # 全 n_sims 本分の月次収益率(積立額の影響を除いた「運用のみ」の収益率)
        return_history_all = np.full((n_months + 1, n_sims), np.nan)

    for t in range(1, n_months + 1):
        # ---- 0) 【追加】今月の運用リターンを計算する前の構成比を記録 ----
        #      (ポートフォリオ収益率 = 前月末の資産配分 × 今月の各資産リターン)
        if track_paths > 0:
            prev_total = values[:track_paths].sum(axis=1)  # 前月末の合計評価額
        if track_all_returns:
            prev_total_all = values.sum(axis=1)  # 全パスの前月末合計評価額

        # ---- 1) 月次リターンを多変量正規分布からサンプリング ----
        monthly_returns = rng.multivariate_normal(
            mean=monthly_mean, cov=cov_monthly, size=n_sims
        )  # shape (n_sims, n_assets)

        if track_paths > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                weights_prev = np.where(
                    prev_total[:, None] > 0,
                    values[:track_paths] / np.where(prev_total[:, None] > 0, prev_total[:, None], 1.0),
                    target_weights,
                )
            port_ret = (weights_prev * monthly_returns[:track_paths]).sum(axis=1)
            return_history[t] = np.where(prev_total > 0, port_ret, np.nan)

        if track_all_returns:
            with np.errstate(invalid="ignore", divide="ignore"):
                weights_prev_all = np.where(
                    prev_total_all[:, None] > 0,
                    values / np.where(prev_total_all[:, None] > 0, prev_total_all[:, None], 1.0),
                    target_weights,
                )
            port_ret_all = (weights_prev_all * monthly_returns).sum(axis=1)
            return_history_all[t] = np.where(prev_total_all > 0, port_ret_all, np.nan)

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

        # リバランス前の構成比を記録(何%まで乖離してからリバランスされたか分かるように)
        if track_paths > 0:
            weight_history[t, :, :] = weights[:track_paths]

        # トリガーが立ったパスのみ、目標配分に戻す(リバランス)
        if np.any(trigger):
            values[trigger] = total[trigger] * target_weights
            rebalance_count[trigger] += 1

        history[t] = values.sum(axis=1)

    return history, values, rebalance_count, weight_history, return_history, return_history_all


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

    history, values_final, rebalance_count, weight_history, return_history, return_history_all = run_simulation(
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
        track_paths=TRACK_SAMPLE_PATHS,
        track_all_returns=TRACK_ALL_RETURNS,
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

    # =================================================================
    # 6. 【診断用】月次で運用されていることの可視化
    #    サンプルパスの株式合計比率(国内株式+外国株式)の月次推移を描画し、
    #    許容幅(±9%)を超えた月にリバランスが起きていることを確認する。
    # =================================================================
    if weight_history is not None and TRACK_SAMPLE_PATHS > 0:
        stock_idx = [2, 3]  # 国内株式・外国株式
        target_stock = TARGET_WEIGHTS[stock_idx].sum()

        fig2, ax = plt.subplots(figsize=(12, 5))
        for i in range(TRACK_SAMPLE_PATHS):
            stock_weight_path = weight_history[:, i, stock_idx].sum(axis=1) * 100
            ax.plot(years_axis, stock_weight_path, lw=1.2, alpha=0.8,
                     label=f"サンプルパス{i+1}")

        ax.axhline(target_stock * 100, color="black", lw=1, ls="-", label="目標(50%)")
        ax.axhline((target_stock + STOCK_TOLERANCE) * 100, color="red", lw=1, ls="--",
                    label=f"許容上限(±{STOCK_TOLERANCE*100:.0f}%)")
        ax.axhline((target_stock - STOCK_TOLERANCE) * 100, color="red", lw=1, ls="--")

        ax.set_xlabel("年")
        ax.set_ylabel("株式合計の構成比(%)")
        ax.set_title(
            "【診断】株式合計比率の月次推移(赤破線=許容幅→到達でリバランス実行)\n"
            "折れ線がギザギザに毎月動いている点が「月次で運用している」ことの証拠"
        )
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/diagnostic_monthly_check.png", dpi=150)
        print("診断グラフを diagnostic_monthly_check.png に保存しました。")

        # 何ヶ月おきにリバランスが起きているかの目安を出力
        avg_months_between = N_MONTHS / max(rebalance_count.mean(), 1e-9)
        print(f"\n【参考】平均リバランス間隔: 約{avg_months_between:.1f}ヶ月に1回")
        print("(乖離許容幅が広め[±5〜9%]のため、月次運用でも頻度は低めになるのが正常です)")

    # =================================================================
    # 7. 【追加】ポートフォリオの収益率グラフ
    #    (左)月次収益率(%)の推移  (右)月次収益率を複利で積み上げた
    #    累積収益率指数(運用開始時点=100。積立元本の影響を除いた
    #    「運用そのものの成果」を見るためのグラフ)
    # =================================================================
    if return_history is not None and TRACK_SAMPLE_PATHS > 0:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

        # --- (左) 月次収益率(%)の推移 ---
        ax = axes3[0]
        for i in range(TRACK_SAMPLE_PATHS):
            ax.plot(years_axis, return_history[:, i] * 100, lw=0.9, alpha=0.8,
                     label=f"サンプルパス{i+1}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("年")
        ax.set_ylabel("月次収益率(%)")
        ax.set_title("ポートフォリオの月次収益率の推移")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        # --- (右) 累積収益率指数(運用開始時点=100、複利。積立額の影響を除く) ---
        ax = axes3[1]
        monthly_ret_filled = np.nan_to_num(return_history, nan=0.0)  # 未定義月(初月)は0%として扱う
        cum_index = 100.0 * np.cumprod(1.0 + monthly_ret_filled, axis=0)
        for i in range(TRACK_SAMPLE_PATHS):
            ax.plot(years_axis, cum_index[:, i], lw=1.3, label=f"サンプルパス{i+1}")
        ax.axhline(100, color="gray", lw=1, ls="--", label="運用開始時点(=100)")
        ax.set_xlabel("年")
        ax.set_ylabel("累積収益率指数(開始時点=100)")
        ax.set_title("ポートフォリオの累積収益率(複利・積立額の影響を除く)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/portfolio_return_chart.png", dpi=150)
        print("収益率グラフを portfolio_return_chart.png に保存しました。")

    # =================================================================
    # 8. 【追加】全試行(N_SIMULATIONS回)のポートフォリオ・リターン分布
    #    毎月積立を考慮した年率換算のmoney-weighted return(IRR)を
    #    各パスについて計算し、その分布をヒストグラムで示す。
    # =================================================================
    print("\n年率リターン(IRR)を全試行について計算中...")
    annual_returns = compute_annualized_returns(final_values, MONTHLY_CONTRIBUTION, N_MONTHS)
    annual_returns_pct = annual_returns * 100

    ret_percentiles = [5, 25, 50, 75, 95]
    ret_stats = {p: np.percentile(annual_returns_pct, p) for p in ret_percentiles}

    print("【年率リターン(IRR)の分布】")
    print(f"  平均値          : {annual_returns_pct.mean():>6.2f} %")
    for p in ret_percentiles:
        print(f"  {p:>3}パーセンタイル  : {ret_stats[p]:>6.2f} %")
    print(f"  マイナスとなる確率: {(annual_returns_pct < 0).mean()*100:>6.1f} %")

    fig4, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(annual_returns_pct, bins=70, color="tab:blue", alpha=0.75)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(ret_stats[50], color="tab:orange", lw=1.5, label=f"中央値 {ret_stats[50]:.2f}%")
    ax.axvline(ret_stats[5], color="tab:red", lw=1, ls="--", label=f"5%タイル {ret_stats[5]:.2f}%")
    ax.axvline(ret_stats[95], color="tab:green", lw=1, ls="--", label=f"95%タイル {ret_stats[95]:.2f}%")
    ax.set_xlabel("年率リターン(%, money-weighted / IRR)")
    ax.set_ylabel("試行数")
    ax.set_title(f"ポートフォリオの年率リターン分布(全{N_SIMULATIONS:,}試行)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/return_distribution.png", dpi=150)
    print("リターン分布図を return_distribution.png に保存しました。")

    # リターン分布もCSVに保存
    pd.DataFrame({"annualized_return_pct": annual_returns_pct}).to_csv(
        "/mnt/user-data/outputs/return_distribution_raw.csv", index=False
    )

    # =================================================================
    # 9. 【追加】積立額の影響を除いた累積収益率を、全試行分重ねて描画
    #    (スパゲッティプロット)
    #    「毎月100円ずつ足すのではなく、最初の100を複利運用だけしたら
    #    どうなるか」を示す指数(運用開始時点=100)を、全N_SIMULATIONS本
    #    分ぶん重ね描きする。1本ずつax.plotすると遅いので、LineCollection
    #    でまとめて描画する。
    # =================================================================
    if return_history_all is not None:
        print(f"\n全{N_SIMULATIONS:,}試行分の累積収益率指数を描画中...")

        monthly_ret_all_filled = np.nan_to_num(return_history_all, nan=0.0)
        cum_index_all = 100.0 * np.cumprod(1.0 + monthly_ret_all_filled, axis=0)
        # shape: (n_months+1, n_sims)

        fig5, ax = plt.subplots(figsize=(11, 6.5))

        # LineCollection用に (n_sims, n_months+1, 2) の座標配列を作る
        x = years_axis  # shape (n_months+1,)
        y = cum_index_all.T  # shape (n_sims, n_months+1)
        segments = np.stack([np.tile(x, (N_SIMULATIONS, 1)), y], axis=2)

        lc = LineCollection(segments, colors="tab:blue", linewidths=0.25, alpha=0.05)
        ax.add_collection(lc)

        # 中央値・5%/95%タイルは目立たせるため上から重ねて描画
        med = np.percentile(cum_index_all, 50, axis=1)
        p5 = np.percentile(cum_index_all, 5, axis=1)
        p95 = np.percentile(cum_index_all, 95, axis=1)
        ax.plot(x, med, color="black", lw=1.8, label="中央値")
        ax.plot(x, p5, color="tab:red", lw=1.2, ls="--", label="5%タイル")
        ax.plot(x, p95, color="tab:green", lw=1.2, ls="--", label="95%タイル")
        ax.axhline(100, color="gray", lw=1, ls=":", label="運用開始時点(=100)")

        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, np.percentile(cum_index_all, 99.5))
        ax.set_xlabel("年")
        ax.set_ylabel("累積収益率指数(開始時点=100)")
        ax.set_title(
            f"積立額の影響を除いた累積収益率(全{N_SIMULATIONS:,}試行を重ね描き)"
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/return_spaghetti_all_paths.png", dpi=150)
        print("全試行重ね描きグラフを return_spaghetti_all_paths.png に保存しました。")


if __name__ == "__main__":
    main()
