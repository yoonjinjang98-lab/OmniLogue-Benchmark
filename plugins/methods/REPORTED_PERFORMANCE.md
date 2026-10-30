# plugins/methods/** — 논문에 보고된 성능 레퍼런스

각 baseline이 wrap/재현하는 원본 논문(또는 그 논문이 인용하는 자기 baseline)이
**자체적으로 보고한** 데이터셋 + 지표 + 수치만 정리한 것입니다. 이 저장소에서
직접 재현/재측정한 값이 아니라 **원본 논문/README/paperswithcode에 적힌 값**이라는
점에 주의 — 우리 환경(모델, 프롬프트, 데이터 분할)에서 같은 수치가 나온다는 보장은
없습니다. 링크는 각 방법론의 `meta.json`/`method.py`에 있는 것과 동일합니다.

값은 웹 리서치로 수집했고, 출처가 PDF 직접 확인인지 AI 요약 기반 fetch인지
에이전트가 구분해서 보고한 경우 그 신뢰도 메모도 같이 남겨뒀습니다. 수치가
안 나오거나 불확실한 경우 추측하지 않고 "확인 안 됨"으로 명시했습니다.

## Family A — Full-context / 검색 기반 (비교 기준선)

`full_context`/`bm25_rag`/`dense_rag`는 특정 논문을 wrap한 게 아니라 이 저장소
자체 구현(비교 기준선 역할)이라 "논문에 보고된 성능"이 해당 없음.

## Family B — 요약 기반 메모리

`running_summary`는 MemoryBank(Zhong et al., AAAI 2024)의 아이디어를 참고한
자체 구현이지 그 논문의 코드/체크포인트를 wrap한 게 아니라서 논문 수치를 직접
적용할 근거가 약함 — 생략.

`raptor` — **"RAPTOR: Recursive Abstractive Processing for Tree-Organized
Retrieval"**, Sarthi et al., ICLR 2024, arXiv:2401.18059.

| 데이터셋 | 지표 | 값 |
|---|---|---|
| QASPER | F1 Match (RAPTOR+GPT-4) | 55.7% (SOTA) |
| QASPER | F1 Match (RAPTOR+GPT-3) | 53.1% |
| QASPER | F1 Match (RAPTOR+UnifiedQA) | 36.6% |
| QuALITY (test) | Accuracy (RAPTOR+GPT-4) | 82.6% (SOTA) |
| QuALITY (hard subset) | Accuracy (RAPTOR+GPT-4) | 76.2% (SOTA) |
| QuALITY (통제 비교) | Accuracy (RAPTOR+GPT-3 / +UnifiedQA) | 62.4% / 56.6% |
| NarrativeQA | ROUGE-L / BLEU-1 / BLEU-4 / METEOR (RAPTOR+UnifiedQA) | 30.8 / 23.5 / 6.4 / 19.1 (METEOR SOTA) |

## Family C — 오픈소스 메모리 에이전트 / 그래프-RAG 프레임워크

`lightmem` — **"LightMem: Lightweight and Efficient Memory-Augmented
Generation"**, ICLR 2026, arXiv:2510.18866. **LoCoMo**(GitHub
`experiments/locomo/readme.md`에 직접 실려있는 표, 여러 하이퍼파라미터
설정 중 대표값) + **LongMemEval**(논문 본문, GitHub에는 표가 없어서
신뢰도가 LoCoMo 표보다 낮음).

| 데이터셋 | 백본 | Accuracy | F1 | BLEU-1 |
|---|---|---|---|---|
| LoCoMo | GPT-4o-mini (최고 설정) | 72.99% | 47.77 | 36.57 |
| LoCoMo | Qwen2.5-32B | 72.40~74.35% | – | – |
| LoCoMo | Qwen3-30B | 60.84~72.60% | 44.78 | 37.06 |
| LongMemEval-S | GPT-4o-mini | 64.29~68.64% | – | – |
| LongMemEval-S | Qwen3-30B | 61.95~73.20% | – | – |

논문 abstract의 헤드라인 수치(가장 신뢰도 높음, 가장 강한 baseline 대비):
QA 정확도 +7.7%(LongMemEval)/+29.3%(LoCoMo), 총 토큰 사용량 최대
38배(LongMemEval)/20.9배(LoCoMo) 감소, API 호출 수 최대
30배(LongMemEval)/55.5배(LoCoMo) 감소. 논문 본문 기준 가장 강한 비교
baseline은 A-MEM으로, LoCoMo에서 +6.10~29.29%p, LongMemEval에서
+2.09~7.67%p 앞선다고 보고.

`memalpha` — **"Mem-alpha: Learning Memory Construction via Reinforcement
Learning"**, arXiv:2509.25911. GitHub README엔 수치 표가 없고 논문 본문의
표를 직접 확인. 두 벤치마크에서 평가.

| 벤치마크 | 세부 항목 | 값 |
|---|---|---|
| Memalpha(자체 검증셋: SQuAD/HotpotQA/PerLTQA/TREC-C/NLU/Pubmed/BookSum) | 항목별 F1/Accuracy | SQuAD 0.786, HotpotQA 0.832, PerLTQA 0.659, TREC-C 0.666, NLU 0.658, Pubmed 0.545, BookSum 0.187 |
| Memalpha | **평균** | **0.642** |
| MemoryAgentBench | Single-Doc/Multi-Doc/LME(S)/TREC-C/NLU/TREC-F/Clinic/Banking77/InfBench | 0.740/0.680/0.520/0.710/0.710/0.410/0.730/0.700/0.129 |
| MemoryAgentBench | **평균** | **0.592** |

β(압축 보상)/γ(내용 보상) 헤드라인 설정은 β=0.05, γ=0.1(두 벤치마크 다 최고
점수: Memalpha 0.642, MemoryAgentBench 0.592) — GitHub README도 이 설정을
"Main Model / optimal hyperparameters"로 명시. 초록의 "30k 토큰만 학습했는데
400k+ 토큰(학습 길이의 13배 이상)까지 일반화"라는 주장은 정성적 서술일 뿐,
구간별 정확도 표는 논문에 없음 — 가장 가까운 정량적 근거는 MemoryAgentBench의
Multi-Doc 항목(메모리 약 323K 토큰 사용, 0.680점, RAG-Top2는 474K 토큰
써도 0.450).

`mem0` — **"Mem0: Building Production-Ready AI Agents with Scalable
Long-Term Memory"**, arXiv:2504.19413. **LoCoMo에서만 평가**.

| 변형 | 카테고리 | F1 | BLEU-1 | LLM-Judge(%) |
|---|---|---|---|---|
| Mem0 | Single-Hop | 38.72 | 27.13 | 67.13±0.65 |
| Mem0 | Multi-Hop | 28.64 | 21.58 | 51.15±0.31 |
| Mem0 | Open-Domain | 47.65 | 38.72 | 72.93±0.11 |
| Mem0 | Temporal | 48.93 | 40.51 | 55.51±0.34 |
| Mem0 | **전체** | – | – | **66.88±0.15** |
| Mem0ᵍ(그래프 변형) | 전체 | – | – | **68.44±0.17** |

`langmem` — langchain-ai/langmem GitHub README에 **학술 논문/기술 보고서/보고된
벤치마크 수치가 전혀 없음** — 순수 소프트웨어 라이브러리 문서만 존재. 해당 없음.

`a_mem` — **"A-MEM: Agentic Memory for LLM Agents"**, NeurIPS 2025,
arXiv:2502.12110.

| 데이터셋 | 카테고리 (백본: GPT-4o-mini) | F1 | BLEU-1 |
|---|---|---|---|
| LoCoMo | Single-hop | 27.02 | 20.09 |
| LoCoMo | Multi-hop | 45.85 | 36.67 |
| LoCoMo | Temporal | 12.14 | 12.00 |
| LoCoMo | Open-domain | 44.65 | 37.06 |
| LoCoMo | Adversarial | 50.03 | 49.47 |
| DialSim (전이 평가, 부록) | – | F1 3.45, BLEU-1 3.37, ROUGE-L 3.54, ROUGE-2 3.60, METEOR 2.05, SBERT-sim 19.51 | |

(논문 Table 1엔 GPT-4o/Qwen2.5-1.5B/3B/Llama-3.2-1B/3B 등 6개 백본 전부 보고 —
위는 그중 하나만 발췌.)

`memoryos` — **"Memory OS of AI Agent"**, EMNLP 2025 Oral, arXiv:2506.06326.

| 데이터셋 | 카테고리 (백본: GPT-4o-mini) | F1 | BLEU-1 |
|---|---|---|---|
| LoCoMo | Single-Hop | 35.27 | 25.22 |
| LoCoMo | Multi-Hop | 41.15 | 30.76 |
| LoCoMo | Temporal | 20.02 | 16.52 |
| LoCoMo | Open-Domain | 48.62 | 42.99 |

(Qwen2.5-3B 백본은 전반적으로 더 낮음 — 논문 자체 주장: GPT-4o-mini 기준
baseline 대비 F1 평균 +49.11%, BLEU-1 평균 +46.18% 개선.)

`zep` — **"Zep: A Temporal Knowledge Graph Architecture for Agent Memory"**,
arXiv:2501.13956.

| 벤치마크 | 조건 | 값 |
|---|---|---|
| DMR | Zep, gpt-4-turbo | 94.8% (vs MemGPT 93.4%, full-context 94.4%) |
| DMR | Zep, gpt-4o-mini | 98.2% (vs full-context 98.0%) |
| LongMemEval_s | Zep, gpt-4o-mini | Accuracy 63.8% (vs full-context 55.4%), latency 3.20s (vs 31.3s) |
| LongMemEval_s | Zep, gpt-4o | Accuracy 71.2% (vs full-context 60.2%), latency 2.58s (vs 28.9s) |

`hipporag` — GitHub `main` 브랜치가 실제로 구현하는 건 **HippoRAG 2**
("From RAG to Memory: Non-Parametric Continual Learning for LLMs", ICML 2025,
arXiv:2502.14802) — 최초 NeurIPS 2024 논문(arXiv:2405.14831)과는 다른 버전.

| 데이터셋 | 지표 (HippoRAG 2, reader: Llama-3.3-70B-Instruct) |
|---|---|
| NaturalQuestions | F1 63.3 |
| PopQA | F1 56.2 |
| MuSiQue | F1 48.6 |
| 2Wiki | F1 71.0 |
| HotpotQA | F1 75.5 |
| LV-Eval | F1 12.9 |
| NarrativeQA | F1 25.9 |
| **전체 평균** | **F1 59.8** |

LoCoMo 결과는 이 논문에 없음.

`hypermem` — **"HyperMem: Hypergraph Memory for Long-Term Conversations"**,
ACL 2026 (accepted), arXiv:2604.08256. **LoCoMo에서만 평가**, 지표는
GPT-4o-mini judge 기준 정확도(%).

| 카테고리 | 값 |
|---|---|
| Single-hop | 96.08 |
| Multi-hop | 93.62 |
| Temporal reasoning | 89.72 |
| Open Domain | 70.83 |
| **전체** | **92.73** |

(논문 자체 주장: 가장 강한 RAG 계열 baseline인 HyperGraphRAG 86.49% 대비
+6.24pt. 가장 강한 메모리-시스템 baseline과의 비교 수치는 출처마다 상이해서
—확인 안 됨으로 남김.)

`graphrag` — **"From Local to Global: A Graph RAG Approach to Query-Focused
Summarization"**, Microsoft Research, arXiv:2404.16130. 절대 점수가 아니라
**LLM-judge pairwise 승률(%)**, vector-RAG(SS) 대비, Podcast/News 두 코퍼스.

| 코퍼스 | 지표 | 승률 |
|---|---|---|
| Podcast | Comprehensiveness (최고, C2) | 78% |
| Podcast | Diversity (최고, C3) | 81% |
| News | Comprehensiveness (최고, C1) | 75% |
| News | Diversity (최고, C2) | 71% |

(Directness는 vector-RAG가 더 나음 — 논문도 이건 의도된 트레이드오프라고 설명.)

`lightrag` — **"LightRAG: Simple and Fast Retrieval-Augmented Generation"**,
arXiv:2410.05779 / EMNLP 2025 Findings. 절대 점수가 아니라 **LLM-judge
Overall 승률(%)**, Agriculture/CS/Legal/Mix 4개 코퍼스, baseline별로 다름.

| 비교 대상 | Agriculture | CS | Legal | Mix |
|---|---|---|---|---|
| vs NaiveRAG | 66.70 | 65.24 | 82.54 | 62.40 |
| vs GraphRAG | 56.38 | 54.02 | 54.30 | 48.14 |

(vs RQ-RAG, vs HyDE 수치도 논문에 있음 — 전부 50% 이상이 LightRAG 선호.)

`hypergraphrag` — **"HyperGraphRAG: Retrieval-Augmented Generation via
Hypergraph-Structured Knowledge Representation"**, NeurIPS 2025,
arXiv:2503.21322.

| 도메인 | F1 | R-S(검색 유사도) | G-E(생성 평가) |
|---|---|---|---|
| Medicine | 35.35 | 70.19 | 59.35 |
| Agriculture | 33.89 | 62.27 | 59.79 |
| Computer Science | 31.30 | 60.09 | 57.94 |
| Legal | 43.81 | 60.47 | 63.61 |
| Mix | 48.71 | 68.21 | 66.90 |

`pathrag` — **"PathRAG: Pruning Graph-based Retrieval Augmented Generation
with Relational Paths"**, arXiv:2502.14902. Legal/History/Biology/Mix/
SQuALITY/SummScreen, 6개 baseline 대비 **승률(%) 평균**.

| 지표 | 6개 baseline 평균 승률 |
|---|---|
| Comprehensiveness | 62.52% |
| Diversity | 65.37% |
| Logicality | 60.68% |
| Relevance | 59.92% |
| Coherence | 59.43% |

SQuALITY에서 사람이 쓴 요약과 직접 비교(Table 4): BLEU-1 35.98, BLEU-2 13.81,
ROUGE-1-F1 15.35, ROUGE-2-F1 3.95, METEOR 18.53 (baseline 중 최고).

## Family D — 데이터셋 네이티브 wrapper

`groupmembench_bm25` — **"GroupMemBench: Benchmarking LLM Agent Memory in
Multi-Party Conversations"**, UCSB-NLP-Chang, arXiv:2605.14498. 우리가 wrap한
건 논문 자체의 **BM25 baseline**(제안 방법론이 아님), 카테고리별 정확도(%).

| 카테고리 | 값 |
|---|---|
| Multi-Hop Reasoning | 40.11 |
| Knowledge Update | 25.23 |
| Term Ambiguity | 14.15 |
| User-Implicit Reasoning | 40.82 |
| Temporal Reasoning | 54.94 |
| Abstention | 77.98 |
| **평균** | **43.22** |

(논문 자체 프레이밍: BM25가 이 벤치마크에서 정교한 에이전트 메모리 시스템들보다
더 강한 baseline으로 보고됨.)

`evermembench_llm` — EverMemBench 논문(EverMind-AI, KDD 2026 Oral,
arXiv:2602.01313)의 **"full dialogue context, 메모리 시스템 없음" baseline**
수치. 백본별 평균 정확도(%).

| 백본 | 평균 |
|---|---|
| Gemini-3-Flash (full context) | 72.61 |
| GPT-4.1-mini (full context) | 37.44 |
| LLaMA-4-Scout-17B (full context) | 40.18 |

(카테고리별 세부값 — Single-hop/Multi-hop/Temporal/Constraint/Proactivity/
Update/Style/Skill/Role — 은 논문 Table 4 참고.)

`dialsim_bm25` — DialSim 논문("DialSim: A Real-Time Simulator for Evaluating
Long-Term Multi-Party Dialogue Understanding of Conversational Agents",
arXiv:2406.13144)의 **BM25-RAG baseline** 자체 정확도(%, ±표준편차).

| 쇼 | 정확도 |
|---|---|
| Friends | 52.11 (±0.52) |
| The Big Bang Theory | 52.54 (±1.65) |
| The Office | 60.92 (±0.40) |

## Family E — 학습된 persona 조건부 생성 모델

`paa` — **"Personalized Dialogue Generation with Persona-Adaptive
Attention"**, Huang et al., AAAI 2023, arXiv:2210.15088. GPT2-SMALL 백본,
**ConvAI2**.

| 지표 | 값 |
|---|---|
| PPL | 14.03 |
| F1 | 17.36 |
| BLEU-1 | 20.50 |
| BLEU-2 | 4.17 |
| Distinct-1 / Distinct-2 | 1.31 / 5.21 |
| Hits@1 | 93.9% |

(저자원 실험: 학습 데이터 20~30%만 써도 100% 데이터로 학습한 GPT2-MEDIUM/LARGE
PPL 17.7/16.9와 비슷하거나 더 나음.)

`ltkm` — **"Learning to Know Myself: A Coarse-to-Fine Persona-Aware Training
Framework"**, Li et al., AAAI 2023 (DOI 10.1609/aaai.v37i11.26545). 학습
프레임워크(K=coarse, A=fine)를 4개 백본에 적용, **ConvAI2 / Persona-Chat /
DuLeMon** 세 데이터셋. Consistency score(C)가 헤드라인 지표.

| 데이터셋 | 최고 조합 | PPL | BLEU-1 | BLEU-4 | Dist-2 | **C** |
|---|---|---|---|---|---|---|
| ConvAI2 (val) | DialoGPT+K+A | 13.19 | 19.70 | 3.52 | 11.14 | **63.90** |
| Persona-Chat (test) | DialoGPT+K+A | 13.44 | 20.59 | 3.95 | 10.85 | **61.97** |
| DuLeMon (전체) | GPT2+K+A | 5.58 | – | 6.94 | 13.09 | **69.57** |

`orig` — **"Towards Robust Personalized Dialogue Generation via
Order-Insensitive Representation Regularization"**, Chen et al., ACL 2023
Findings, arXiv:2305.12782. **Persona-Chat** (원본 persona, 두 백본).

| 백본 | BLEU-1 | BLEU-2 | ROUGE | Entropy | CIDEr | C-score |
|---|---|---|---|---|---|---|
| GPT2-ORIG | 14.61 | 7.43 | 14.94 | 6.54 | 10.60 | 0.733 |
| BART-ORIG | 14.64 | 7.90 | 15.20 | 6.41 | 13.27 | 0.446 |

`sbys` — **"Score Before You Speak"**, Saggar et al., ECAI 2025,
arXiv:2508.06886. **PERSONA-CHAT(test)** / **ConvAI2(val)**, 두 모델 규모.

| 모델 | 데이터셋 | PPL | BLEU-1 | BLEU-4 | Dist-2 | C |
|---|---|---|---|---|---|---|
| DialoGPT 117M | PERSONA-CHAT | 11.92 | 23.65 | 4.71 | 13.96 | 66.64% |
| DialoGPT 117M | ConvAI2 | 14.02 | 22.11 | 4.21 | 16.52 | 70.60% |
| Llama 3.1 8B | PERSONA-CHAT | 6.88 | 26.27 | 5.85 | 19.50 | 64.47% |
| Llama 3.1 8B | ConvAI2 | 6.98 | 25.22 | 4.99 | 23.09 | 73.11% |

`bob` — **"BoB: BERT Over BERT..."**, Song et al., ACL 2021.
**PersonaChat(전체 데이터)**:

| 지표 | 값 |
|---|---|
| PPL | 7.8 |
| Distinct-1 / Distinct-2 | 8.40 / 36.08 |
| p.Ent / p.Ctd | 7.3 / 83.4 |
| ΔPPL | 76.1 |
| C.Score | 17.18 |
| Fluency / Informativeness / Relevance | 4.12 / 4.03 / 4.09 |
| Per.C. | 0.60 |

PersonalDialog(biased testset)에서도 평가: PPL 19.5, C.Score 12.76, KvPI
ΔP 85.40 (random testset에서는 논문도 "persona 신호가 너무 희박해 어떤
방법도 일관되게 이기지 못함"이라고 명시).

`lmedr` — **"Learning to Memorize Entailment and Discourse Relations for
Persona-Consistent Dialogues"**, Chen et al., AAAI 2023, arXiv:2301.04871.
**PersonaChat(원본/revised)** + **DSTC7-AVSD**.

| 데이터셋 | Hits@1 | PPL | F1 |
|---|---|---|---|
| PersonaChat, 원본 persona | 89.5 | 10.99 | 21.99 |
| PersonaChat, revised persona | 85.0 | 13.42 | 19.54 |

| DSTC7-AVSD | 값 |
|---|---|
| BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4 | 0.801 / 0.680 / 0.576 / 0.488 |
| METEOR | 0.316 |
| ROUGE-L | 0.631 |
| CIDEr | 1.403 |

## 신뢰도 메모

리서치 에이전트가 PDF를 직접 읽고 확인한 값(bob/lmedr/ltkm/mem0/memoryos/
orig/paa/pathrag/raptor/sbys/zep/hipporag/hypergraphrag/lightrag)과, HTML
fetch 기반 AI 요약으로 확인한 값(dialsim_bm25, evermembench_llm 일부,
groupmembench_bm25, hypermem 일부)이 섞여 있습니다. 인용/발표용으로 쓰기
전에는 후자 쪽 수치를 원본 PDF 표와 직접 대조하는 걸 권장합니다. hypermem의
"가장 강한 baseline 대비 개선폭" 비교 수치는 출처 간 불일치가 있어 이 문서엔
싣지 않았습니다.
