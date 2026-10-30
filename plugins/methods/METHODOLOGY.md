# plugins/methods/ — 방법론 정리

각 baseline이 **어떤 기법**을 쓰고, 이 저장소가 데이터셋에 쓰는 것과 같은
축(`long`/`persona`/`multiparty`, `plugins/datasets/*/meta.json`의 `axes` 필드
참고)으로 볼 때 **어떤 문제를 겨냥해 설계됐는지** 한눈에 보기 위한 문서다.
"완료됐는지/버그가 있는지"는 `README.md`, "왜 이걸 다음에 할지"는
`BASELINE_METRIC_PLAN.md`가 원본이고, 이 문서는 순수하게 기법/축 분류만 다룬다.

축은 **방법론 자체의 설계 의도** 기준이다 — 예를 들어 `dialsim_bm25`는
multi-party 데이터셋(dialsim)을 대상으로 하지만, 검색 로직 자체엔 "누가 누구한테
말했는지"를 특별히 다루는 메커니즘이 없어서 `multiparty`를 약하게만 표시했다.
반대로 `groupmembench_bm25`는 프롬프트에 `asking_user_id`(질문자가 누구인지)를
직접 꽂아 넣어서 답이 달라질 수 있게 만들었으니 `multiparty`를 강하게 표시한다.

## Family A — Full-context / 검색 기반 (비교 기준선)

| 방법론 | 기법 | long | persona | multiparty |
|---|---|:---:|:---:|:---:|
| `full_context` | 대화 전체를 프롬프트에 그대로 욱여넣고 질문 | ●(약점 노출용) | – | – |
| `bm25_rag` | 턴 단위로 쪼개서 BM25(어휘 검색)로 top-k만 추림 | ● | – | – |
| `dense_rag` | 턴 단위로 쪼개서 임베딩 코사인 유사도로 top-k만 추림 | ● | – | – |

세 개 다 "메모리 프레임워크가 이것보다 나은가?"를 판가름하는 비교 기준선.
persona/multiparty에 특화된 메커니즘은 없고, 어떤 축의 데이터셋에 돌려도
동일하게 동작한다 (turn에 화자 라벨이 있으면 그대로 프롬프트에 실릴 뿐).

## Family B — 요약 기반 메모리

| 방법론 | 기법 | long | persona | multiparty |
|---|---|:---:|:---:|:---:|
| `running_summary` | 세션(또는 고정 길이 턴 윈도우)마다 LLM 호출 1번으로 누적 요약을 갱신, 최종 요약만으로 답변 | ●● | – | – |
| `raptor` | 대화를 재귀적으로 클러스터링·요약해 다층 트리를 만들고, 모든 레이어를 통틀어 임베딩 유사도로 top-k 검색 | ●● | – | – |

둘 다 "컨텍스트 윈도우보다 긴 대화"라는 `long` 축 문제 하나에 집중한 설계다.

## Family C — 오픈소스 메모리 에이전트 / 그래프-RAG 프레임워크

| 방법론 | 기법 | long | persona | multiparty |
|---|---|:---:|:---:|:---:|
| `mem0` | LLM이 대화에서 사실을 추출/통합(add/update/delete)해서 벡터 스토어(Qdrant)에 저장, 질문마다 검색 | ●● | – | – |
| `langmem` | LangGraph 기반 메모리 스토어에 LLM tool-calling으로 삽입/갱신 결정, 시맨틱 검색으로 응답 | ●● | – | – |
| `a_mem` | Zettelkasten 스타일 — 청크마다 키워드/맥락/태그 붙은 노트를 만들고 ChromaDB에 저장, 주기적으로 노트끼리 연결·갱신(evolution) | ●● | – | – |
| `memoryos` | 단기/중기/장기 3단 계층 메모리 — 개인 비서용으로 설계됨(LoCoMo로 자체 평가) | ●● | ○ | – |
| `zep` | LLM이 엔티티/관계를 추출해 temporal knowledge graph(그래프 DB)에 저장, 하이브리드(벡터+텍스트) 검색 | ●● | – | – |
| `hipporag` | LLM 기반 open information extraction(개체명+triple 추출)으로 지식그래프 구축, 임베딩 유사도 + Personalized PageRank로 검색 | ●● | – | – |
| `hypermem` | 3계층 하이퍼그래프(topic→episode→fact), 계층마다 BM25+dense RRF 융합으로 위에서 아래로 좁혀가며 검색 | ●● | ○ | – |
| `graphrag` | LLM 기반 엔티티/관계 추출로 지식그래프 구축, Leiden 커뮤니티 탐지로 계층적 클러스터링, 커뮤니티마다 LLM이 요약 리포트 작성 — 엔티티 중심 검색(local) 또는 전체 리포트 map-reduce(global) | ●● | – | – |
| `lightrag` | LLM 기반 엔티티/관계 추출로 지식그래프 구축(커뮤니티 계층 없음), local(엔티티 중심)/global(관계 중심)/hybrid/naive(순수 벡터)/mix(그래프+벡터, 기본값) 중 검색 모드 선택 | ●● | – | – |
| `hypergraphrag` | LLM이 관계를 이항 엣지가 아니라 **여러 엔티티를 한 번에 묶는 hyperedge**(완결성 점수가 붙은 지식 세그먼트 문장)로 추출, 엔티티+hyperedge에 대한 하이브리드 벡터 검색 | ●● | – | – |
| `pathrag` | LLM 기반 엔티티/관계 추출로 지식그래프 구축(이항 엣지, graphrag/lightrag와 동일), 검색 시 1-hop 이웃이 아니라 질의 관련 엔티티 사이의 **관계 경로(path)를 찾아 가지치기**해서 멀티홉 추론에 사용 | ●● | – | – |
| `lightmem` | LLM이 사실을 추출해 임베딩 인덱스에 저장한다는 점은 mem0/a_mem과 같지만, 실제 기여는 효율성 — attention 기반 인코더로 연속된 턴을 하나의 토픽으로 먼저 묶은 뒤에만 추출을 트리거하고(턴마다 LLM 호출 안 함), 쓰기마다 정리하는 대신 주기적인 "offline" 업데이트로 유사 항목을 병합/정리 | ●● | – | – |
| `memalpha` | 다른 열두 개와 근본적으로 다름 — LLM 프롬프트로 "항상 추출해서 저장"하게 시키는 게 아니라, core/semantic/episodic 메모리에 어떤 연산(삽입/수정/삭제)을 언제 호출할지를 **GRPO 강화학습으로 직접 학습시킨 작은 정책 모델**(Qwen3-4B)이 결정. 실제 학습된 체크포인트가 공개돼 있어서 RL 학습 없이 바로 추론 사용 | ●● | – | – |
| `rfmem` | 나머지 전부와 다르게 **LLM 기반 사실 추출이 아예 없음** — 순수 검색 전략 자체가 기여점. 질문마다 저렴한 probe 검색으로 확신도(평균 유사도/엔트로피)를 재서 확신 높으면 빠른 직접 top-k 검색, 낮으면 KMeans 재클러스터링+쿼리 임베딩 갱신을 반복하는 beam search로 전환(인간 기억의 Familiarity/Recollection 이중과정 이론에서 착안). 검색 자체가 LLM 호출 0회라 이 세션에서 API 키 없이 가장 강하게 검증됨 | ●● | – | – |
| `anchormem` | LLM이 청크마다 원자적 "Fact"를 추출하는 건 mem0/a_mem/lightmem과 같지만, 유일하게 **2차 LLM 요약 레이어**가 있음 — 임베딩 유사도로 여러 청크에 걸친 관련 Fact들을 클러스터링한 뒤 LLM이 클러스터를 상위 "Event"로 요약, 질문 시 Fact와 Event 양쪽 다 매칭하는 dual-path 검색 | ●● | – | – |

전부 "긴 대화에서 뭘 기억하고 뭘 버릴지"가 핵심 설계 문제라 `long`이 확실한 1순위
축이다. `memoryos`/`hypermem`은 논문에서 각각 개인화된 사용자 프로필/화자별
스타일 벡터 개념을 일부 다루긴 하지만, persona 자체를 모델링하는 게 목적은
아니라서 `persona`는 약하게(○)만 표시했다. multi-party 전용 메커니즘(발화자
attribution, addressee 추론 등)은 열다섯 개 다 없음 — turn에 화자 라벨만 실릴 뿐.
`graphrag`/`lightrag`/`hipporag`/`hypergraphrag`/`pathrag` 다섯 다 기법이 가장
비슷하다(모두 LLM이 그래프를 추출 → 그래프 기반 검색) — 차이는 hipporag가
Personalized PageRank로 그래프를 직접 훑고, graphrag는 그래프를 커뮤니티
단위로 미리 요약해두고 그 요약을 검색·map-reduce하는 계층적 접근인 반면,
lightrag는 커뮤니티 계층 없이 그래프 자체(엔티티/관계)와 원본 텍스트 청크를
직접 섞어 검색(mix)하고, hypergraphrag는 애초에 관계 표현 자체가 다르다 —
이항 엣지가 아니라 임의 개수의 엔티티를 동시에 묶는 hyperedge라서, "A와 B와
C가 함께 등장하는 하나의 사실"을 A-B, B-C, A-C 세 개의 별도 엣지로 쪼개지
않고 하나로 표현할 수 있다는 게 논문의 핵심 주장이며, pathrag는 그래프
추출(이항 엣지) 자체는 graphrag/lightrag와 같지만 검색 시 1-hop 이웃만 보는 게
아니라 엔티티 사이의 관계 경로 자체를 찾아 가지치기해서 명시적인 멀티홉 추론
체인을 제공하는 게 차별점. graphrag가 다섯 중 가장 무겁고(추가 요약 단계),
lightrag/hypergraphrag/pathrag가 더 가볍다(셋 다 lightrag의 예전 버전과
유사하거나 같은 코드 계보라 별도 이벤트 루프 브릿징도 필요 없었음 — pathrag는
그 대신 라이브러리 자체의 진짜 버그 두 개를 method.py에서 우회해야 했다:
`insert()`/`query()`가 스스로 깨진 async 래퍼라 직접 못 쓰고, `QueryParam`의
기본 검색 모드 값 자체가 라이브러리가 처리 못 하는 모드라 항상 명시적으로
override해야 함).

## Family D — 데이터셋 네이티브 wrapper

| 방법론 | 기법 | long | persona | multiparty |
|---|---|:---:|:---:|:---:|
| `groupmembench_bm25` | groupmembench 원본 BM25 baseline 그대로 이식 — 토크나이저/패시지 포맷/에이전트 프롬프트까지 동일, **질문자(`asking_user_id`)를 검색 쿼리와 프롬프트에 직접 반영** | ● | ○ | ●● |
| `evermembench_llm` | evermembench 원본 "LLM system"(메모리 없이 전체 대화 그대로) baseline 이식 — 화자/채널 태그 포함한 대화 포맷, MC/open-ended 프롬프트까지 원본 그대로 | ● | ○ | ● |
| `dialsim_bm25` | DialSim 원본 BM25 baseline 이식 — 세션 전체 단위 검색, `(A)~(E)` 강제 프롬프트, 쇼별 챗봇 페르소나(Ross/Sheldon/Michael) | ● | ○ | ○ |

Family D는 전부 "그 데이터셋 논문이 보고한 숫자를 그대로 재현"하는 게 목적이라
데이터셋 자체의 축(`plugins/datasets/*/meta.json`)을 그대로 물려받는다.
`groupmembench_bm25`만 유일하게 "누가 묻는지"를 검색/프롬프트 로직에 실제로
반영해서 `multiparty`를 강하게(●●) 줬고, 나머지는 화자 라벨 수준의 약한 반영이라
○ 처리했다.

## Family E — 학습된 persona 조건부 생성 모델

| 방법론 | 기법 | long | persona | multiparty |
|---|---|:---:|:---:|:---:|
| `paa` | GPT2 인코더-디코더에 persona/context를 게이트된 cross-attention으로 융합 — Persona-Adaptive Attention(AAAI 2023), ConvAI2로 학습 필요 | – | ●● | – |
| `ltkm` | 임의의 대화 모델에 적용하는 2단계 학습 레시피(모델 아키텍처 아님) — coarse 단계는 DNLI 기반 관계추출로 persona-aware 질문을 만들어 학습 augment, fine 단계는 DNLI 기반 엔테일먼트 스코어로 찾은 핵심 persona 단어를 MLM으로 바꿔치기한 negative sample에 대해 contrastive loss 추가. Learning to Know Myself(AAAI 2023), 공식 코드 없어서 논문만 보고 새로 구현 | – | ●● | – |
| `orig` | DialoGPT 학습 시 persona 문장 순서를 두 가지(원본/무작위 셔플)로 각각 돌려서 두 출력 분포 사이에 KL divergence 정규화 loss를 추가 — persona 순서를 바꿔도 답이 안 바뀌게 만드는 게 목적. Order-Insensitive Representation Regularization(ACL 2023 Findings) | – | ●● | – |
| `sbys` | (persona, context, response) 대신 (persona, context, response, response의-persona-일치도-점수) 4-튜플로 학습, 생성 시엔 마지막 User 턴 뒤에 텍스트로 "Score: 1.0"(최대 점수)을 넣어서 가장 일치하는 답을 유도. Score Before You Speak(ECAI 2025) | – | ●● | – |
| `bob` | BERT 인코더 하나 + BERT 디코더 두 개(decoder1: 평범한 persona+query→response 생성, decoder2: decoder1의 hidden state와 raw persona 토큰 둘 다에 별도로 cross-attention + MNLI/DNLI unlikelihood 학습으로 persona 일관성 강화, 추론 시 최종 답변은 decoder2 출력) — BERT Over BERT(ACL 2021) | – | ●● | – |
| `lmedr` | BART 인코더-디코더에 학습 가능한 "메모리" 코드북 두 개(memory1: 엔테일먼트 관계 프로토타입, persona 문장만 인코딩해서 읽음; memory2: 담화관계 프로토타입, persona+history+query 전체를 인코딩해서 읽음)를 추가 — 각각 softmax 가중합으로 읽은 뒤 더해서 디코더 첫 hidden state에 직접 주입, 별도 토큰/cross-attention 없이 엔테일먼트·담화구조를 조건으로 생성. Learning to Memorize Entailment and Discourse Relations(AAAI 2023) | – | ●● | – |

지금까지 만든 것 중 **`persona`가 핵심 설계 축인 baseline은 이 여섯뿐**.
메모리 프레임워크가 아니라 fine-tuned 생성 모델이라 `long`/`multiparty`는
해당 없음(대상 데이터셋도 msc/personachat, 둘 다 길지 않고 2인 대화).
`paa`/`orig`/`sbys`/`bob`/`lmedr`는 각각 완결된 아키텍처(+ 학습 레시피) 하나인
반면 `ltkm`은 "어떤 생성 모델에든 얹을 수 있는 학습 방법"이라 방법론 자체는
모델 아키텍처가 아니라 학습 레시피. 여섯의 관계: `paa`가 페르소나를 "잘
반영하는가"(persona relevance)에 집중한다면, `orig`는 그 위에 "반영하는
방식이 입력 순서에 안정적인가"(robustness), `sbys`는 "얼마나 잘 반영했는지를
점수로 직접 지정해서 요청할 수 있는가"(controllability), `bob`은 "관계 추출
없이도 raw persona 텍스트에 별도 cross-attention을 둬서 반영 자체를
강화"(architectural reinforcement), `lmedr`은 "엔테일먼트/담화관계라는 두
축을 학습 가능한 코드북으로 명시적으로 분리해서 각각 읽어낸 다음 디코더에
직접 주입"(disentangled relational memory)를 다룬다는 점이 각각 다르다 —
다만 실제로 돌려볼 수 있는지는 다르다 — `sbys`는 upstream이 HuggingFace Hub에
공개한 체크포인트로 바로 실행 가능하고, `ltkm`은 이 저장소에서 실제
ConvAI2/DNLI 전체로 직접 학습을 돌려서(공식 코드가 없어 학습 스크립트 자체를
새로 작성해야 했던 게 오히려 이 학습을 여기서 직접 돌릴 수 있게 만든 이유)
체크포인트를 확보했다. 나머지 넷(`paa`/`orig`/`bob`/`lmedr`)은 아직 실제
학습을 안(못) 돌려서 현재 실행이 불가능하다.

## 요약: 축별로 보면

- **`long` 전담** (Family A/B/C 대부분): `full_context`/`bm25_rag`/`dense_rag`
  (비교 기준선) → `running_summary`/`raptor`(요약 압축) →
  `mem0`/`langmem`/`a_mem`/`memoryos`/`zep`/`hipporag`/`hypermem`/`graphrag`/
  `lightrag`/`hypergraphrag`/`pathrag` (메모리 에이전트/그래프). 서로 다른
  방식으로 "긴 대화를 어떻게 압축/색인해서 기억할지"를 푸는 게 공통 목표.
- **`persona` 전담**: `paa`/`ltkm`/`orig`/`sbys`/`bob`/`lmedr` 여섯 — 나머지는 부수적으로만 다룸.
- **`multiparty`를 실제로 반영**: `groupmembench_bm25`(질문자 ID를 검색/프롬프트에
  직접 사용)가 제일 강하고, `evermembench_llm`/`dialsim_bm25`는 화자/채널 태그
  정도. 순수 multi-party 메커니즘(addressee 해석, 화자 간 관계 모델링)을 갖춘
  baseline은 아직 없음 — `ubuntu_irc`(addressee recognition), `mpchat`(persona
  grounding, 근데 구조적으로 막힘) 데이터셋용으로 언젠가 필요해질 부분.
