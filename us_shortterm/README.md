# us_shortterm

일봉으로 돌릴 수 있는 미국 단기 전략 12개를 SPY, QQQ, IWM, DIA 에 같은 조건으로 한 번씩 돌려, 2차 심화로 넘길 전략을 고르는 1차 선별이다. 지시서는 [SPEC.md](SPEC.md) 다.
엔진은 [krx_backtest_core](https://github.com/destory1984/krx_backtest_core) v0.2.1 을 그대로 쓴다. 주문 코드, 증권사 API 연결은 없다.

## 조건

- 선별 기간: 2008-01-02 → 2022-12-30. 2023-01-01 뒤 데이터는 불러온 직후 잘라낸다(2차 심화에서 처음 연다). 200일선 같은 지표의 앞 기간에는 2007년 데이터를 쓴다.
- 가격: 배당까지 반영한 가격(`krxbt.us.ticker_frame`).
- 비용: 편도 0.12% (수수료 0.07% + 슬리피지 0.05%).
- 전액 투입, 한 번에 한 포지션, 손절 없음, 최대 보유일 없음. 쉬는 날 수익은 0 이다.
- 설정: 전략 12개 중 S2 → S7 은 신호일 종가 체결(close)과 다음 날 시가 체결(open)을 둘 다 돌려 18개, ETF 4개를 곱해 72개다. 결과를 보고 규칙이나 문턱값을 바꾸지 않았다.

| # | 전략 | 진입 | 청산 | 판본 |
|---|---|---|---|---|
| S1 | IBS | IBS < 0.2 | IBS > 0.8 | open |
| S2 | RSI(2) | 종가 > 200일선, RSI(2) < 10 | 종가 > 5일선 | close, open |
| S3 | 누적 RSI | 종가 > 200일선, 이틀 RSI(2) 합 < 35 | RSI(2) > 65 | close, open |
| S4 | R3 | 종가 > 200일선, RSI(2) 사흘 연속 하락, RSI(2) < 10 | RSI(2) > 70 | close, open |
| S5 | %b | 종가 > 200일선, %b 사흘 연속 < 0.2 | %b > 0.8 | close, open |
| S6 | Double 7s | 종가 > 200일선, 7일 최저 종가 | 7일 최고 종가 | close, open |
| S7 | 3일 고저 | 종가 > 200일선, 종가 < 5일선, 고가·저가 사흘 연속 하락 | 종가 > 5일선 | close, open |
| S8 | 월말월초 | 그 달 끝에서 둘째 거래일 종가 | 그 달 셋째 거래일 종가 | close |
| S9 | 밤사이 보유 | 매일 종가 | 다음 날 시가 | co |
| S10 | 변동성 돌파 | 고가 ≥ 시가 + 0.5 × 전날 범위 | 다음 날 시가 | bar |
| S11 | NR7 돌파 | 전날이 7일 중 범위가 가장 좁고 고가 > 전날 고가 | 그날 종가 | bar |
| S12 | 갭 하락 되돌림 | 시가 < 전날 저가 | 그날 종가 | bar |

통과 기준(설정마다): 4개 ETF 중 3개 이상에서 비용 포함 연 수익률이 위험(최대 낙폭)을 맞춘 보유보다 높고 그 ETF 의 거래가 각각 50건 이상, 4개 ETF 평균 손익분기 비용이 편도 10bp 이상.

## 결과

18개 설정 중 통과한 설정은 없다. 자세한 표는 `results/screen/report.md`, 검증은 `results/screen/checks.md` 에 있다.

- 가장 가까운 설정은 S1 IBS(open)다. 연 수익률이 SPY +5.1%, QQQ +5.0%, IWM +6.3%, DIA +5.8% 이고, 위험을 맞춘 보유를 SPY(+0.3%p), IWM(+1.1%p) 두 곳에서만 이겼다. 평균 손익분기 비용은 22.6bp 다.
- S5 %b 는 IWM, DIA 에서 위험을 맞춘 보유를 이기고 평균 손익분기 비용이 100bp 를 넘지만, 15년 동안 거래가 ETF 마다 33 → 40건이라 거래 수 기준(50건)에 못 미친다.
- RSI 계열(S2, S3, S4)과 S7 은 ETF 4개 모두에서 위험을 맞춘 보유보다 연 0.2 → 6.8%p 낮다.
- S9 → S12 는 거래가 많아 비용에 진다. S9 는 비용이 없으면 연 +6.2% (SPY) → +10.6% (IWM) 인데, 비용 포함으로는 연 -40% 안팎이다. 손익분기 비용이 편도 1.3 → 2.2bp 다.
- close 판본과 open 판본의 연 수익률 차이는 24칸 중 20칸이 ±0.4%p 안이다. 나머지는 S3 QQQ(close 가 0.9%p 낮음)와 S7 SPY, QQQ, IWM(close 가 0.6 → 1.1%p 높음)이다.

## 돌리는 법

```
pip install -r requirements.txt
python -m src.screen.run       # 72개 설정 → results/screen/trades_*.parquet, metrics.csv, daily.parquet
python -m src.screen.report    # → results/screen/report.md, verdicts.csv
python -m src.screen.checks    # → results/screen/checks.md
python -m pytest -q tests
```

데이터는 모음 저장소 밖 `../../us_data` 를 같이 쓴다(환경변수 `US_DATA_DIR` 로 바꿀 수 있다). 없으면 `python -m krxbt.fetch_us --dir ../../us_data --tickers SPY QQQ IWM DIA` 로 받는다.

## 파일

| 파일 | 내용 |
|---|---|
| `config.yaml` | 모든 설정 |
| `src/common.py` | 설정 읽기, 봉인한 데이터 불러오기 |
| `src/indicators.py` | SMA, RSI(2), IBS, %b, 범위 |
| `src/equity.py` | 거래 목록 → 일별 자산곡선, 지표 |
| `src/benchmark.py` | 보유, 위험을 맞춘 보유, 노출을 맞춘 보유 |
| `src/screen/strategies/` | 전략 하나에 파일 하나 (`s01_ibs.py` → `s12_gap.py`) |
| `src/screen/bar_sim.py` | 장중 가격에 사는 전략(S10 → S12) 시뮬레이터 |
| `src/screen/run.py`, `report.py`, `checks.py` | 실행, 보고서, 검증 |
| `NOTES.md` | 판단과 이유, 돌린 설정 수 |
