# ibs_bear_backtest

IBS 평균회귀를 약세장(종가가 자기 200일선 아래)에서만 쓰면 비용과 보유를 이기는지, 가설을 만든 적 없는 ETF 6개로 한 번에 판정했다. 지시서는 [SPEC.md](SPEC.md) 다.
엔진은 [krx_backtest_core](https://github.com/destory1984/krx_backtest_core) v0.2.1 을 그대로 쓴다. 주문 코드, 증권사 API 연결은 없다.

## 규칙

- 진입: 신호일 IBS < 0.2 이고 종가 < 자기 200일 단순이동평균이면 다음 날 시가에 산다.
- 청산: 보유 중 IBS > 0.8 이면 다음 날 시가에 판다.
- 손절 없음, 최대 보유일 없음, 한 번에 한 포지션. 비용 편도 0.12%(수수료 0.07% + 슬리피지 0.05%). 가격은 배당 반영.
- 기간 2008-01-02 → 2026-09-25.
- 판정 종목: XLF, XLK, XLE, EEM, EFA, FXI. 가설을 만든 SPY, QQQ, IWM, DIA, SOXX 는 참고로만 봤다.
- 판정 종목의 조건 없는 IBS 효과는 ibs_lev_backtest 가 이미 쟀다. 새로 판정한 것은 "효과가 200일선 아래에 몰리고, 그것만으로 보유를 이기는가"다.

## 결과

**판정: 신호 알림 후보가 아니다.** 미리 정한 기준 6개 중 3개만 만족했다. 이로써 IBS 는 멈춘다.

| 기준 | XLF | XLK | XLE | EEM | EFA | FXI | 판정 |
|---|---|---|---|---|---|---|---|
| 1. 필터 없는 IBS 거래당 평균, 200일선 아래 > 위 (5개 이상) | ○ | ○ | × | ○ | ○ | ○ | 만족 5/6 |
| 2. 약세장 판본 거래 30건 이상 | 219 | 160 | 236 | 269 | 223 | 299 | 만족 |
| 3. 낙폭 맞춘 보유 대비 연 수익률 (4개 이상 이김) | -0.4%p | -8.1%p | -6.1%p | -0.9%p | -3.8%p | +0.4%p | 불만족 1/6 |
| 4. 손익분기 비용 평균 ≥ 편도 20bp | 42.0 | 29.9 | 12.7 | 20.9 | 9.9 | 20.0 | 만족 22.6bp |
| 5. 200일선 아래 무작위 진입 대비 PF 백분위 ≥ 0.95 (4개 이상) | 0.997 | 0.932 | 0.633 | 0.961 | 0.801 | 0.951 | 불만족 3/6 |
| 6. 6개 바구니 Deflated Sharpe ≥ 0.95 (N 1,216) | | | | | | | 불만족 0.060 |

- 방향은 맞다. 6개 중 5개에서 IBS 반등이 200일선 아래에서 더 컸다. 그러나 크기가 가설을 만든 종목보다 작다. 아래 구간 t 값이 판정 종목에서는 -0.15 → 1.67 인데, SPY·IWM·DIA 에서는 2.0 → 2.8 이었다.
- 약세장에서만 사니 돈이 대부분 논다(노출 14 → 28%). 거래당 수익이 커져 손익분기 비용은 평균 22.6bp 로 넘었지만, 같은 낙폭의 보유를 이긴 것은 FXI 하나(+0.4%p)다.
- 6개를 1/6 씩 굴린 바구니는 연 +2.2%, 최대 낙폭 -30% 로, 같은 낙폭의 6개 보유(연 +4.8%)에 2.7%p 진다.
- 참고 종목에서는 약세장 판본이 IWM(+1.8%p), DIA(+0.5%p), SOXX(+0.1%p)에서 보유를 이긴다. 가설을 만든 종목에서만 좋게 나온다는 뜻이다.

자세한 표는 `results/report.md`, 검증은 `results/checks.md` 에 생긴다(저장소에 올리지 않는다).

## 돌리는 법

미국 데이터는 저장소 밖 `../../us_data` 를 같이 쓴다. 기준값 V 와 대조에 `../us_shortterm/results/`, `../ibs_backtest/results/` 의 결과 파일이 필요하다(그 폴더를 먼저 돌린다).

```
pip install -r requirements.txt
python -m src.run        # 거래 목록 → results/trades_<종목>_<판본>.parquet
python -m src.metrics    # 지표와 비교 기준 → metrics.csv, daily.parquet
python -m src.regime     # 국면별 거래 → regime.csv
python -m src.basket     # 판정 6개 바구니 → basket.csv
python -m src.overfit    # Deflated Sharpe → dsr.csv
python -m src.monkey     # 200일선 아래 무작위 진입 → monkey.csv (1분쯤)
python -m src.report     # → results/report.md
python -m src.checks     # → results/checks.md
```

## 파일

| 파일 | 내용 |
|---|---|
| `config.yaml` | 모든 설정과 판정 기준 |
| `src/common.py` | 설정, 봉인한 데이터 불러오기, 규칙 두 판본 |
| `src/equity.py`, `src/benchmark.py` | us_shortterm 에서 복사: 자산곡선과 지표, 비교 기준 |
| `NOTES.md` | 판단과 이유 |
