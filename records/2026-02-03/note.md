## Web UI 제작

지금까지 구현한 배치/탐색 로직을 사용자가 웹에서 실시간으로 상호작용하며 실행하고, 탐색 과정(후보 선택, 평가, 배치 진행 상황 등)을 시각적으로 확인할 수 있도록 Web UI를 제작했습니다. 사용자는 입력 파라미터를 조정하고, 배치 결과와 중간 과정을 단계별로 탐색하며, 주요 지표(비용, 제약 위반 여부, 진행률 등)를 함께 확인할 수 있습니다.

기본적인 레이아웃은 다음과 같습니다.

| 구분 | 이미지 |
|---|---|
| Light mode | ![Light mode](light_mode.png) |
| Dark mode | ![Dark mode](dark_mode.png) |

첫 세션을 시작하면 화면 좌측에서 env, agent, wrapper, search mode를 선택할 수 있습니다. 현재 env는 서버에 등록된 환경만 선택 가능하며, env 파일(JSON 형식) 업로드 기능은 추후 필요 시 추가 구현할 예정입니다.

| 항목 | 이미지 |
|---|---|
| Env 선택 | ![select env](select_env.png) |
| Agent 선택 | ![select agent](select_agent.png) |

선택한 옵션에 대한 세부 설정(parameter)은 화면 우측 설정 창에서 조정할 수 있습니다.

| 항목 | 이미지 |
|---|---|
| Parameter 설정 | ![select parameter](select_param.png) |

이와 같은 설정을 마친 뒤 Session Start를 누르면, 선택한 설정을 기준으로 세션이 시작됩니다.


### Search 기능

Search 알고리즘(MCTS, Beam Search)을 선택하면 탐색 기능을 사용할 수 있습니다. Run Search 버튼을 누르면 탐색이 시작되고, 레이아웃 화면과 Candidate 목록에 실시간으로 결과가 업데이트됩니다.

**레이아웃 화면:**
- 탐색 전: Prior(P) 기반 색상 표시 (모든 유효 후보)
- 탐색 후: Q-value 기반 색상 표시 (탐색된 후보만)
- 초록색에 가까울수록 유망한 좌표

**Candidate 목록:**

| 컬럼 | 설명 |
|------|------|
| Q | Q-value. 탐색을 통해 평가된 기대 가치. 높을수록 좋은 배치 |
| P | Prior. 탐색 전 휴리스틱 기반 초기 점수 |
| N | Visits. 해당 후보가 탐색에서 방문된 횟수. 많이 방문될수록 신뢰도 높음 |

컬럼 헤더를 클릭하면 해당 기준으로 정렬할 수 있으며, 값이 없는 항목(N/A)은 항상 맨 아래에 표시됩니다.

![candidate](candidate.png)

![example](ex.gif)