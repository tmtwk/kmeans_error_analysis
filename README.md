## 실행 방법

```bash
pip install -r requirements.txt
python data_generator.py
python main.py
```

## 면접을 위해 반드시 이해해야 할 핵심

### 보고서 전체를 외우기보다 다음 다섯 가지를 자기 말로 설명할 수 있으면 충분하다

```bash
왜 균일 난수가 아니라 make_blobs를 사용했는가?
실제 군집이 없는 균일 데이터에서는 K-Means가 공간을 강제로 나누는 결과만 만들기 때문이다.

왜 MinMaxScaler와 StandardScaler를 모두 사용했는가?
MinMaxScaler는 축을 1~100으로 만들어 해석하기 쉽게 하고, StandardScaler는 군집화 전에 두 특성의 평균과 분산을 맞추기 위해 사용했다.

K-Means와 DBSCAN의 가장 큰 차이는 무엇인가?
K-Means는 모든 데이터를 정해진 수의 군집에 포함하고, DBSCAN은 밀도가 낮은 데이터를 노이즈로 분리한다.

DBSCAN의 실루엣 계수가 더 높은데 왜 항상 더 좋다고 할 수 없는가?
DBSCAN은 노이즈를 제외하고 실루엣 계수를 계산했기 때문에 평가 대상 자체가 다르기 때문이다.

K-Means++는 무엇을 개선하는가?
초기 중심점이 한쪽에 몰리는 문제를 줄여 결과의 안정성과 수렴 속도를 개선한다.
```
