# dialsim_bm25 — 이게 뭐고 왜 있는지

## 요약

`dialsim_bm25`는 새로운 검색 기법이 아니다. **BM25는 학습되는 모델이 아니라
TF-IDF 계열의 통계 공식**이라, `prepare()`에서 그 대화의 문서들만 갖고 그 자리에서
점수를 계산하는 것뿐이고 gradient descent 같은 학습 과정 자체가 없다. 이 저장소의
베이스라인들 중 진짜 "학습"이 필요한 건 `paa` 하나뿐이고 (원본이 사전학습 체크포인트를
안 줘서 아직 실행 불가), `bm25_rag`/`dialsim_bm25`/`dense_rag`/메모리 프레임워크들
전부 학습 없이 그 자리에서 인덱싱하거나 이미 학습된 API를 호출하는 것뿐이다.

## `bm25_rag`(범용)랑 뭐가 다른가

**알고리즘 자체는 완전히 동일하다.** "질문이랑 겹치는 단어 많은 문서를 BM25 점수로
뽑아서 프롬프트에 넣는다"는 같은 로직. 차이는 세부 구현 3가지뿐:

1. 토크나이저 — 원본은 `nltk.word_tokenize`(소문자화), `bm25_rag`는 정규식(`[A-Za-z0-9_]+`)
2. 검색 단위 — 원본은 세션(씬) 전체를 문서 하나로, `bm25_rag`는 턴 단위로 쪼갬
3. 프롬프트 — 원본은 `RAG_qa_prompt_multi_choice_structured.txt` 그대로
   (Ross/Sheldon/Michael 페르소나, "(A)~(E) 중 하나로만 답하라, 모르면 (E)" 강제),
   `bm25_rag`는 우리 식 문구

## 그럼 왜 만들었나

**새 기능이 아니라 "DialSim 논문이 보고한 숫자랑 우리 결과를 1:1로 비교하고
싶을 때"만 의미가 있다.** 이 저장소의 baseline family 중 "D. 데이터셋 네이티브
wrapper"(`groupmembench_bm25`, `evermembench_llm`도 같은 부류)의 성격이 다 이렇다 —
새로운 걸 만드는 게 아니라 그 데이터셋 논문 저자가 자기 실험에서 쓴 코드를
토씨 하나 안 바꾸고 그대로 이식해서, 우리 harness 안에서 재현 가능하게 만드는 것뿐.

**논문 재현이 목적이 아니라면 이거 없어도 상관없다.** `plugins/datasets/dialsim/`
adapter가 `.sessions`/`.qa` 형태라서, 이미 있던 범용 `bm25_rag`/`full_context`/
`dense_rag`/`running_summary`가 dialsim에서 별도 작업 없이 그냥 잘 돌아간다
(메모리 프레임워크들끼리 비교하는 용도로는 그걸로 충분).

## 폴더 구조 관련 헷갈렸던 부분

- `plugins/datasets/dialsim/` = 데이터셋 어댑터 (이게 있어야 "dialsim"이 로드됨)
- `plugins/methods/dialsim_bm25/` = 베이스라인 하나 (여기)

`plugins/methods/` 밑엔 데이터셋 이름 그대로인 폴더가 없다 — 메서드는 "어떤
알고리즘이냐"로 이름 붙이지 "어떤 데이터셋용이냐"로 안 붙인다 (`full_context`,
`bm25_rag`, `dense_rag`처럼). `dialsim_bm25`는 "dialsim 데이터셋의 **자체**
BM25 baseline을 재현한 것"이라 `<데이터셋>_<방법>` 네이밍(Family D 공통 패턴)을
쓴 것뿐이다.

## 코드가 실제로 하는 일 (`method.py`)

1. `prepare(conversation)`: 세션(씬)마다 `[Date: ..., Session #N]\n{전체 대사}`
   문서를 만들고, 전체 문서로 `BM25Okapi` 인덱스를 그 자리에서 구축 (학습 아님, 인덱싱)
2. `answer(item)`: 질문을 같은 토크나이저로 토큰화해서 BM25 인덱스에 쿼리 →
   점수 높은 세션 상위 10개(`top_k`) 검색
3. 검색된 세션들을 `[Retrieved Dialog History]`에 넣고 LLM한테 `(A)~(E)` 중
   하나로만 답하게 강제 → 응답에서 어느 letter가 나왔는지 찾아서 그 옵션의
   실제 텍스트로 변환해 반환 (채점은 여기서 안 함 — `qa_em`/`qa_f1` 같은
   기존 Metric plugin이 담당, 이 저장소 전체의 "Method는 답만, Metric은 채점만"
   원칙 그대로)

원본 대비 문서화된 차이 하나: 원본은 질문마다 6가지 캐릭터별 문구 중 랜덤으로
골라서 물어보는데, `plugins/datasets/dialsim/adapter.py`는 그중 중립적인
"default" 문구만 남겨둬서, 이 method도 그걸 쓴다.
