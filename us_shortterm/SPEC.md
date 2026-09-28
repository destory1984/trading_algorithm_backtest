# 미국 단기 전략 1차 선별 작업 지시서 (전체판)

작성일 2026.9.28. 이 문서 하나로 저장소 만들기부터 결과 보고서까지 끝낸다. 다른 지시서를 먼저 돌릴 필요가 없다. Claude Code에 그대로 넘긴다.

## 1. 목적과 범위

일봉으로 돌릴 수 있는 미국 단기 전략 12개를 같은 조건에서 한 번씩만 돌려, 2차 심화로 넘길 전략을 고른다.

- 한다: 저장소 구성, 데이터 받기, 전략 12개 구현, 72개 설정 실행, 보고서
- 하지 않는다: 파라미터 조정, 2023년 이후 데이터 사용, 주문 코드, 증권사 API 연결

## 2. 지켜야 할 원칙

**원칙 1. 2023.1.1 이후 데이터는 봉인한다.** 선별 기간은 2008.1.2 → 2022.12.30이다. 2023.1.1 → 현재는 2차 심화에서 처음 연다. 가격 표를 불러온 직후 2022.12.30 뒤를 잘라내고, 잘라낸 표의 마지막 날짜를 단언문으로 확인한다. 지표 계산을 위한 앞 기간(200일선 등)은 2007년 데이터를 써도 된다.

**원칙 2. 규칙은 이 문서에 적힌 그대로, 한 번만 돌린다.** 결과를 보고 문턱값이나 규칙을 바꾸지 않는다. 규칙이 모호해 해석이 필요하면 멈추고 사용자에게 묻는다. 돌린 설정 수를 모두 세어 `NOTES.md`에 적는다(2차 심화의 Deflated Sharpe 계산에 쓴다).

## 3. 저장소 구성

사용자의 기존 저장소 구조(destory1984/bnf_backtest)를 따른다.

```
us_shortterm/                  새 저장소
├── config.yaml                모든 설정은 여기에만
├── requirements.txt
├── NOTES.md                   판단과 이유, 돌린 설정 수
├── README.md                  돌리는 법
├── src/
│   ├── common.py              설정 읽기, 데이터 불러오기, 봉인 검사
│   ├── indicators.py          SMA, RSI(2), IBS, %b 등 지표
│   ├── equity.py              거래 목록 → 일별 자산곡선, 지표 계산
│   ├── benchmark.py           보유, 위험을 맞춘 보유, 노출을 맞춘 보유
│   └── screen/
│       ├── strategies/        전략 하나에 파일 하나 (s01_ibs.py ... s12_gap.py)
│       ├── bar_sim.py         장중 가격 진입 전략용 시뮬레이터 (S10, S11, S12)
│       ├── run.py             72개 설정 실행
│       ├── report.py          report.md 생성
│       └── checks.py          검증 스크립트
├── tests/                     pytest
└── results/                   git 제외
```

`requirements.txt`:

```
krxbt @ git+https://github.com/destory1984/krx_backtest_core@v0.2.1
pandas>=2.2
numpy
pyarrow
PyYAML
pytest
```

이 저장소는 2차 심화에도 계속 쓴다.

## 4. 데이터 받기

1. 미국 데이터 폴더는 기존 `../us_data`를 같이 쓴다. 없으면 새로 만든다.
2. 네 ETF를 받는다. 지수 파일(SPX, NDX, DJI)도 함께 갱신된다.

```
python -m krxbt.fetch_us --dir ../us_data --tickers SPY QQQ IWM DIA
```

3. 받은 뒤 확인: `../us_data/prices/`에 네 파일이 있고, 각 파일의 첫 날짜가 2007-01-03 근처인지, `index_SPX.parquet`의 첫 날짜가 2007-01-03 근처인지.
4. 가격: `open`, `high`, `low`, `close`는 액면분할만 반영, `adj_close`는 배당까지 반영한 값이다. `krxbt.us.ticker_frame`이 OHLC를 `adj_close / close` 비율로 맞춰 배당 반영 가격으로 바꿔 준다. 이 함수를 그대로 쓴다.

## 5. config.yaml

아래를 그대로 쓴다. `krxbt.us.ticker_frame`과 `krxbt.engine.run`이 읽는 키가 모두 들어 있다.

```yaml
data:
  dir: ../krx_data           # 한국 데이터 경로. 이 작업에서는 쓰지 않지만 엔진이 키를 요구한다
  start: "2008-01-02"        # 신호를 세는 첫날
  screen_end: "2022-12-30"   # 봉인선. 이 날짜 뒤 데이터는 불러온 직후 잘라낸다

us:
  dir: ../us_data
  market_index: SPX
  min_avg_value: 2.0e+7      # 20일 평균 거래대금 하한(달러). 네 ETF 모두 넉넉히 넘는다
  costs:
    buy_fee: 0.0007          # 국내 증권사 해외주식 수수료 편도 0.07%
    sell_fee: 0.0007
    sell_tax: 0.0
    slippage: 0.0005         # 편도 0.05%

universe:
  min_listing_days: 25
  avg_value_window: 20

indicators:
  ma_window: 25              # ticker_frame이 요구하는 키. 전략 지표는 src/indicators.py에서 따로 계산한다
  index_ma_window: 120
  index_disparity_crash: 97

rules:                       # 설정마다 run.py가 복사본을 만들어 덮어쓴다
  entry_on_signal_close: false
  target_exit_on_close: false

screen:
  tickers: [SPY, QQQ, IWM, DIA]
  hold_max: 10000            # 최대 보유일 없음
```

엔진은 `cfg["costs"]`를 읽으므로 `run.py`에서 `cfg["costs"] = cfg["us"]["costs"]`로 넣어 준다.

## 6. 공통 지표 정의 (`src/indicators.py`)

모든 지표는 그날 종가까지의 값으로만 계산한다. 결측은 신호 없음으로 본다.

| 이름 | 정의 |
|---|---|
| SMA(n) | 종가 n일 단순 평균. `rolling(n, min_periods=n).mean()` |
| RSI(2) | 와일더 방식. 전일 대비 상승폭 평균 AG, 하락폭 평균 AL을 `ewm(alpha=1/2, adjust=False, min_periods=2)`로 구한다. RSI = 100 - 100 / (1 + AG/AL). AL = 0이고 AG > 0이면 100, 둘 다 0이면 50 |
| IBS | (종가 - 저가) / (고가 - 저가). 고가 = 저가이면 결측 |
| 볼린저 %b | 중심선 SMA(20), 표준편차는 20일 모표준편차(`std(ddof=0)`), 상단 = 중심 + 2σ, 하단 = 중심 - 2σ. %b = (종가 - 하단) / (상단 - 하단) |
| 범위 | 고가 - 저가 |

## 7. 전략 12개

표기: t = 신호일, 모든 조건은 t일 종가 시점에 판단한다.

"종가 체결" 전략은 원전이 신호일 종가에 산다. 한국에서 손으로 주문하면 미국 종가(한국 시간 새벽 5시 또는 6시)에 맞추기 어려우므로, S2에서 S7까지는 **다음 날 시가 체결 판본을 같이 돌린다.** 시가 판본은 진입과 청산을 모두 다음 거래일 시가에 한다.

### 7.1 엔진으로 돌리는 전략 (S1에서 S9)

| # | 이름 | 진입 조건 `cand[t]` | 청산 조건 `target[t]` | 체결 판본 |
|---|---|---|---|---|
| S1 | IBS | IBS[t] < 0.2 | IBS[t] > 0.8 | open |
| S2 | RSI(2) | 종가 > SMA(200) 이고 RSI(2) < 10 | 종가 > SMA(5) | close, open |
| S3 | 누적 RSI | 종가 > SMA(200) 이고 RSI(2)[t] + RSI(2)[t-1] < 35 | RSI(2) > 65 | close, open |
| S4 | R3 | 종가 > SMA(200), RSI(2)[t-2] < 60, RSI(2)[t-2] < RSI(2)[t-3], RSI(2)[t-1] < RSI(2)[t-2], RSI(2)[t] < RSI(2)[t-1], RSI(2)[t] < 10 | RSI(2) > 70 | close, open |
| S5 | %b | 종가 > SMA(200) 이고 %b[t], %b[t-1], %b[t-2] 모두 < 0.2 | %b > 0.8 | close, open |
| S6 | Double 7s | 종가 > SMA(200) 이고 종가[t] = min(종가[t-6] … 종가[t]) | 종가[t] = max(종가[t-6] … 종가[t]) | close, open |
| S7 | 3일 고저 | 종가 > SMA(200), 종가 < SMA(5), 고가[t] < 고가[t-1] < 고가[t-2] < 고가[t-3], 저가[t] < 저가[t-1] < 저가[t-2] < 저가[t-3] | 종가 > SMA(5) | close, open |
| S8 | 월말월초 | t = 그 달의 끝에서 두 번째 거래일 | t = 그 달의 셋째 거래일 | close 전용 |
| S9 | 밤사이 보유 | 매 거래일 | 매 거래일 | 종가 진입, 다음 날 시가 청산 |

모든 `cand`에 `a["eligible"]`을 곱한다.

엔진 설정값:

| 판본 | `entry_on_signal_close` | `target_exit_on_close` | 의미 |
|---|---|---|---|
| close | true | true | 신호일 종가 진입, 청산 조건이 선 날 종가 청산 |
| open | false | false | 다음 날 시가 진입, 청산 조건이 선 다음 날 시가 청산 |
| S9 | true | false | 종가 진입, 다음 날 시가 청산 |

S8과 S9의 달력 계산은 `krxbt.us.load_calendar`(SPX 거래일)로 한다. S8 판본은 close 하나다(월말월초 효과는 종가 기준으로 정의된다).

엔진 동작에서 주의할 점 두 가지:

1. close 판본에서 엔진은 진입 당일(`d == e`)에도 `target[e]`를 본다. 신호일에 청산 조건까지 맞으면 같은 종가에 사고파는 0일 거래가 생긴다. 원전은 다음 날부터 청산을 본다. 그래서 close 판본에는 `target & np.logical_not(cand)`를 넘긴다. 이 처리로 바뀌는 거래 수를 `checks.md`에 적는다.
2. 엔진은 보유 중 새 신호를 무시한다. 청산이 시가(open 판본, S9)이면 청산한 날 종가의 새 신호는 받는다. 원전과 같다.

### 7.2 따로 시뮬레이션하는 전략 (S10에서 S12, `bar_sim.py`)

| # | 이름 | 진입 | 청산 |
|---|---|---|---|
| S10 | 변동성 돌파 | 돌파가 L = 시가[t] + 0.5 × (고가[t-1] - 저가[t-1]). 고가[t] ≥ L이면 진입가 = max(시가[t], L) | 다음 날 시가 |
| S11 | NR7 돌파 | 범위[t-1] < min(범위[t-7] … 범위[t-2]) 이면(전날이 최근 7일 중 가장 좁음, 동률 제외) L = 고가[t-1]. 고가[t] > L이면 진입가 = max(시가[t], L) | 그날 종가 |
| S12 | 갭 하락 되돌림 | 시가[t] < 저가[t-1] 이면 진입가 = 시가[t] | 그날 종가 |

- S10은 청산한 날(다음 날) 다시 돌파 조건이 서면 같은 날 다시 진입할 수 있다. 시가 청산 뒤 장중 재진입이므로 허용한다.
- 비용은 엔진과 같은 식(`krxbt.costs.cost_factors`)으로 진입가와 청산가에 적용한다.
- 거래 목록 형식은 엔진 출력과 같게 맞춘다(`signal_date`, `entry_date`, `exit_date`, `entry_px`, `exit_px`, `ret`, `hold_days`).
- 단위 테스트: 시가가 돌파가 위에서 열린 날(진입가 = 시가), 고가가 돌파가에 못 미친 날(진입 없음), 고가가 정확히 돌파가인 날(S10은 진입, S11은 진입 없음).

### 7.3 설정 수

S1 1개, S2에서 S7까지 6개 × 2판본 = 12개, S8, S9, S10, S11, S12 각 1개. 합계 18개 × 종목 4개 = **72개**다.

## 8. 자산곡선과 지표 (`src/equity.py`)

모든 전략의 거래 목록을 같은 함수로 일별 자산곡선으로 바꾼다.

- 초기자본 1, 전액 투입, 쉬는 날 수익 0
- 거래 하나의 일별 수익: 진입일은 종가 / 진입가 - 1 (종가 진입이면 0), 보유 중간일은 종가 / 전일 종가 - 1, 청산일은 청산가 / 전일 종가 - 1 (같은 날 진입·청산이면 청산가 / 진입가 - 1). 진입일에 매수 비용, 청산일에 매도 비용을 곱한다.
- 검산: 자산곡선 마지막 값 = Π(1 + 거래 수익률)과 소수 여덟째 자리까지 같다.

지표:

| 지표 | 정의 |
|---|---|
| 거래 수, 승률 | 비용 포함 수익률 > 0인 거래 비율 |
| 거래당 기대값 | 비용 포함 거래 수익률 평균 |
| 손익비 | 이익 거래 합 / 손실 거래 합의 절댓값 |
| 연 수익률 | 마지막 값^(365.25 / 달력일수) - 1 |
| 샤프 | 일별 수익률 평균 / 표준편차(ddof=1) × √252. 무위험 수익률 빼지 않음. 쉬는 날 포함 |
| 최대 낙폭 | 일별 자산곡선 기준 |
| 노출 | 포지션을 들고 있던 날의 비율. 같은 날 진입·청산한 거래의 날도 센다 |
| 비용 0 연 수익률 | 비용을 0으로 두고 다시 계산 |
| 손익분기 비용 | 거래당 평균 수익률이 0이 되는 편도 비용(bp). 이분법으로 찾는다 |

## 9. 비교 기준 (`src/benchmark.py`)

같은 기간, 같은 ETF로 세 가지를 만든다.

1. 보유: 전액 보유, 비용 없음
2. 위험을 맞춘 보유: 비중 w로 매일 재조정하는 보유(나머지 현금 수익 0). w는 0에서 1 사이에서 이분법으로 찾아, 이 곡선의 최대 낙폭이 전략의 최대 낙폭과 같아지게 한다. 전략 낙폭이 보유 낙폭보다 크면 w = 1
3. 노출을 맞춘 보유: w = 전략의 노출 비율로 둔 매일 재조정 보유

전략마다 2번 대비 연 수익률 차이, 3번 대비 연 수익률 차이를 낸다.

## 10. 통과 기준

설정(전략 × 판본)마다 판정한다. 아래를 모두 만족하면 2차 심화로 넘긴다.

1. 4개 ETF 가운데 3개 이상에서 비용 포함 연 수익률이 위험을 맞춘 보유보다 높다.
2. 그 3개 이상의 ETF에서 거래가 각각 50건 이상이다.
3. 4개 ETF 평균 손익분기 비용이 편도 10bp 이상이다.

같은 전략의 close 판본과 open 판본은 따로 판정한다. close 판본만 통과하면 "종가 체결 필요"로 표시한다.

## 11. 결과물

`results/screen/`에 저장한다.

- `trades_<전략>_<판본>_<종목>.parquet`: 거래 목록 72개
- `metrics.csv`: 72행, 8절과 9절의 모든 지표
- `report.md`: 아래 순서로. 수치는 모두 `metrics.csv`에서 채운다

report.md 구성:

1. 요약표: 행 = 18개 설정, 열 = 4개 ETF 각각의 "연 수익률 / 위험을 맞춘 보유 대비 차이", 끝 열 = 통과 여부
2. 통과 설정의 상세표(8절 지표 전부)
3. 통과 설정끼리의 일별 수익률 상관표(SPY 기준). 서로 같은 날 들어가는 전략인지 보기 위함이다
4. S2에서 S7까지 close 판본과 open 판본의 연 수익률 차이 표
5. S9의 비용 0 연 수익률(밤사이 수익이 얼마나 되는지의 기준선)
6. 돌린 설정 수(72개)와, 2022.12.30 뒤 데이터를 쓰지 않았다는 확인
7. 한 줄 결론

## 12. 검증 (`src/screen/checks.py` → `results/screen/checks.md`)

1. 봉인: 모든 가격 표의 마지막 날짜가 2022.12.30 이하다.
2. 미래 데이터 금지: SPY의 2022년 마지막 20거래일 가격을 무작위로 바꿔 다시 돌려도, 그 전에 시작한 거래는 하나도 바뀌지 않는다(전략 12개 모두).
3. 손 검산: 전략마다 SPY 거래 3건을 골라 신호일, 진입일, 진입가, 청산일, 청산가를 원래 가격 표와 대조해 적는다.
4. 자산곡선 검산: 8절의 곱셈 검산을 72개 모두 통과한다.
5. 엔진 대조: S1(open)을 엔진이 아닌 단순 반복문으로 다시 짜서 SPY 거래 목록이 같은지 확인한다.
6. close 판본의 `target & np.logical_not(cand)` 처리로 달라진 거래 수
7. 고가 = 저가인 날의 수(종목별)

pytest는 `tests/`에 두고, 지표(RSI(2) 손계산 5일치, %b, IBS 결측), `bar_sim` 세 경우, 자산곡선 곱셈 검산을 넣는다.

## 13. 하지 말 것

- 2022.12.30 뒤 데이터 사용
- 규칙, 문턱값, 지표 설정 변경
- 통과하지 못한 전략을 살리려는 추가 실험
- 주문 코드, 증권사 API 연결
- 계정 생성이나 유료 데이터 사용

## 14. 끝나면

사용자에게 `results/screen/report.md`의 요약표와 통과 목록을 보여 주고 멈춘다. 2차 심화는 통과 설정을 보고 사용자가 정한다.

## 15. 전략 출처

| # | 출처 |
|---|---|
| S1 | https://github.com/toniker10/SPY-IBS-Mean-Reversion-Strategy , https://github.com/CazSyd/IBS-Strategy |
| S2 | 코너스·알바레즈, Short Term Trading Strategies That Work. https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2 |
| S3 | 같은 책 67쪽. https://easycators.com/thinkscript/cumulative-rsi-2-trading-strategy/ |
| S4 | 코너스·알바레즈, High Probability ETF Trading 4장. https://www.whselfinvest.com/en-lu/trading-platform/free-trading-strategies/tradingsystem/81-r3-larry-connors |
| S5 | 같은 책 5장. 밴드 설정은 원전 값을 확인하지 못해 표준값(20일, 2σ)을 쓴다. https://quantifiedstrategies.substack.com/p/larry-connors-b-strategy-bollinger-ab6 |
| S6 | https://thepatternsite.com/Double7sSetup.html |
| S7 | High Probability ETF Trading. 청산(종가 > 5일선)은 널리 인용되는 판본이다. 원전의 물타기 판본은 쓰지 않는다. https://www.edgerater.com/blog/connors-etf-three-day-highlow-method |
| S8 | McConnell·Xu(2008). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=917884 , https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes |
| S9 | https://github.com/gustavosmede/overnight_paper |
| S10 | 래리 윌리엄스, Long-Term Secrets to Short-Term Trading |
| S11 | 토비 크라벨, Day Trading with Short Term Price Patterns and Opening Range Breakout(1990) |
| S12 | 일반형 규칙, 특정 원전 없음 |
| 엔진 | https://github.com/destory1984/krx_backtest_core (v0.2.1) |
