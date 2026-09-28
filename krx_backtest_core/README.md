# krx_backtest_core

코스피·코스닥 일봉 백테스트 엔진이다. 매매 규칙은 없다. 규칙은 알고리즘마다 따로 저장소를 만들어 그쪽에 둔다.
이 저장소는 여러 알고리즘이 같은 데이터와 같은 계산 방식을 쓰게 하려고 [bnf_backtest](https://github.com/destory1984/bnf_backtest) 에서 떼어 냈다.

| 모듈 | 하는 일 |
|---|---|
| `krxbt.fetch` | 종목 목록·지수·일봉 수집 (네이버 `siseJson`, 상장폐지 종목은 KRX KIND). 처음엔 3,243종목에 약 35분 |
| `krxbt.data` | 받은 데이터 읽기, 쓸 수 있는 종목 목록(`universe_tickers`) |
| `krxbt.frame` | 종목별 일봉 표. 거래정지·정리매매·수정 안 된 시세 끊김을 거르는 규칙이 여기 있다 |
| `krxbt.engine` | 신호와 익절 조건(불리언 배열 두 개)을 받아 거래 목록을 만든다. 다음 날 시가 진입, 손절, 최대 보유일, 비용 |
| `krxbt.costs` | 수수료·슬리피지·매도세(해마다 바뀐 세율) |
| `krxbt.portfolio` | 신호 거래 목록으로 동시 보유 N종목 포트폴리오를 돌린다. 폭락일에 자리 늘리기, 손절 뒤 재진입 제한 |

## 설치

알고리즘 저장소의 `requirements.txt` 에 버전을 박아서 설치한다. 엔진을 고쳐도 옛 알고리즘의 결과가 바뀌지 않게 하려는 것이다.

```
krxbt @ git+https://github.com/destory1984/krx_backtest_core@v0.2.1
```

엔진을 같이 고치는 중이면 로컬 폴더를 편집 가능 모드로 설치한다.

```
pip install -e ../krx_backtest_core
```

## 데이터 폴더

시세는 어느 저장소에도 넣지 않는다. 크기가 약 260MB 이고, 네이버 시세를 다시 배포하는 일이 되기 때문이다.
데이터 폴더 하나를 모든 알고리즘이 같이 읽는다. 안에는 `prices/*.parquet`, `index_*.parquet`, `universe.parquet` 가 있다.

폴더 위치는 환경변수 `KRX_DATA_DIR` 가 있으면 그것을 쓰고, 없으면 config 의 `data.dir` 을 config 파일 기준 상대 경로로 읽는다(`../krx_data`).

수집·갱신은 알고리즘 저장소 하나에서 돌리면 된다. 다시 돌리면 새 날짜만 받는다.

```
python -m krxbt.fetch                 # config.yaml 의 data·universe 설정을 읽는다
python -m krxbt.fetch --config path/to/config.yaml
```

## 미국 주식 데이터 (`krxbt.fetch_us`)

야후(`yfinance`)에서 일봉을 받아 `../us_data` 에 한국 데이터와 같은 모양으로 둔다. 아직 수집만 하고, 매매 계산(`frame`·`engine`)은 한국 규칙(가격 제한 ±30% 등)에 맞춰져 있어 미국 데이터에는 쓰지 않는다.

```
python -m krxbt.fetch_us --dir ../us_data     # 처음 받기와 갱신 모두. 129종목에 약 3분
```

- 종목: 지금의 나스닥100(101개, GOOG·GOOGL 둘 다) + 다우30 + `us_data/watchlist.txt` 에 한 줄씩 적은 종목. 목록은 위키백과 표에서 읽는다. 관심 종목 파일은 저장소에 올리지 않는다.
- 지수: `index_NDX`, `index_DJI`, `index_SPX` (2007-01-03 부터).
- 가격: `open`·`high`·`low`·`close` 는 액면분할만 반영한 값이고, `adj_close` 는 배당까지 반영한 값이다.
  특별배당이 큰 종목은 `close` 로 보면 가짜 급락이 나온다. 예: KDP 2018-07-10 에 주당 103달러 특별배당으로 `close` 가 -82% 인데 `adj_close` 로는 +11% 다.
- **생존편향이 있다.** 지금 지수에 든 종목만 받으므로, 그동안 지수에서 빠지거나 망한 종목은 없다. 폭락 매수 같은 규칙은 실제보다 좋게 나온다.
- 2007년부터 있는 종목은 129개 중 90개다. 나머지는 늦게 상장했다(TSLA 2010, META 2012, ARM 2023 등).
- 거래량 0 인 날이 많은 종목: FER 2,478일(2024년 미국 상장 전에는 장외 시세), POET 708일(나스닥 이전 전 장외 시세). 이 기간은 거래정지처럼 걸러야 한다.

## 새 알고리즘 만들기

config.yaml 에 `data`, `universe`, `indicators`, `rules`, `costs` 가 있어야 한다. `bnf_backtest/config.yaml` 을 복사해서 시작하면 된다.
알고리즘이 할 일은 종목별로 "언제 사는가"(`cand`)와 "언제 익절하는가"(`target`)를 불리언 배열로 만드는 것뿐이다.

```python
from krxbt.config import load_config
from krxbt.data import load_calendar, universe_tickers
from krxbt.frame import index_features, ticker_frame
from krxbt import engine

cfg = load_config("config.yaml")
cal, idx = load_calendar(cfg), index_features(cfg)
tk = universe_tickers(cfg)
f = ticker_frame(cfg, "005930", "KOSPI", cal, idx, delisted=False)
a = engine.arrays(f)
cand = a["eligible"] & (a["disp"] <= 85) & engine.market_mask(a, "crash97")
target = a["close"] >= a["ma"]
trades = engine.run(a, cfg, cand, target, stop=-0.08, hold=10)
trades = trades[~trades["data_break"]]   # 시세 끊김을 끼고 들고 있던 거래는 뺀다
```

## 거르는 규칙 (`krxbt.frame`)

- 거래량 0 인 날은 거래정지로 보고 표에 남기되 사고팔지 않는다. 보유일 수에는 센다.
- 상장폐지 종목의 마지막 7거래일(정리매매)에는 신호를 막는다. 가격 제한이 없어 하루 -50% 가 나온다.
- 하루 하락이 가격 제한(2015-06-15 전 -16%, 이후 -30%)을 넘거나, 3거래일 넘는 거래정지가 풀리면 그날부터 이동평균 기간(25거래일) 동안 신호를 막는다.
- 그런 끊긴 날을 끼고 들고 있던 거래는 `data_break` 로 표시한다. 결과에서 빼는 것은 알고리즘 몫이다.
- 20일 평균 거래대금(종가 × 거래량 어림) 10억 원 미만, 상장 25거래일 미만 구간은 신호를 막는다(`eligible`).

## 버전

- v0.2.1 (2026-09-28): 설치 목록에 `yfinance`·`lxml`(미국 수집에 필요) 추가.
- v0.2.0 (2026-09-27): 미국 주식 수집(`krxbt.fetch_us`)과 미국용 종목표(`krxbt.us`, 배당 반영 가격, 가격 제한 없음). 한국 쪽 계산은 그대로다.
- v0.1.0 (2026-09-27): bnf_backtest 에서 떼어 냄. 익절 조건을 "종가 ≥ 25일선" 고정에서 알고리즘이 넘기는 배열로 바꿨다. 결과 숫자는 떼기 전과 같다.
