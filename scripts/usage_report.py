#!/usr/bin/env python3
"""Summarize the paid-API usage ledger (DATA_ROOT/usage/*.jsonl, written by
src/dialogue_benchmarks/usage.py) for a billing period -- the "what was this
spend used for" evidence that goes next to the provider's own invoice and
usage dashboard for a research-grant settlement.

    .venv/bin/python scripts/usage_report.py --from 2026-10-01 --to 2026-10-31
    .venv/bin/python scripts/usage_report.py --from 2026-10-01 --to 2026-10-31 --provider openai --out report.html

Prints a plain-text summary and writes a self-contained HTML page (default:
DATA_ROOT/usage/report_<from>_<to>.html) meant to be opened in a browser and
screen-captured: totals, per model (matches the dashboard's per-model view),
per purpose, per experiment, per day, per worktree.
"""
import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dialogue_benchmarks._paths import DATA_ROOT  # noqa: E402

USAGE_DIR = DATA_ROOT / "usage"
PURPOSE_KO = {"method": "베이스라인 답변 생성", "judge": "평가(LLM 채점)", "embed": "임베딩(검색)", "other": "기타"}


def load(start: date, end: date, provider) -> list:
    rows = []
    for path in sorted(USAGE_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                day = datetime.fromisoformat(r["ts"]).date()
                if start <= day <= end and (provider is None or r["provider"] == provider):
                    r["day"] = day.isoformat()
                    rows.append(r)
    return rows


def group(rows: list, key) -> list:
    acc = defaultdict(lambda: {"calls": 0, "tokens_in": 0, "cached_tokens": 0, "tokens_out": 0, "cost_usd": 0.0})
    for r in rows:
        a = acc[key(r)]
        a["calls"] += 1
        for k in ("tokens_in", "cached_tokens", "tokens_out"):
            a[k] += r.get(k, 0)
        a["cost_usd"] += r["cost_usd"]
    return sorted(acc.items(), key=lambda kv: -kv[1]["cost_usd"])


def _table(title: str, header: list, groups: list, label) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    head += "<th>호출 수</th><th>입력 토큰</th><th>(캐시)</th><th>출력 토큰</th><th>비용 (USD)</th>"
    body = []
    for k, a in groups:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in label(k))
        body.append(
            f"<tr>{cells}<td class=n>{a['calls']:,}</td><td class=n>{a['tokens_in']:,}</td>"
            f"<td class=n>{a['cached_tokens']:,}</td><td class=n>{a['tokens_out']:,}</td>"
            f"<td class=n>${a['cost_usd']:.4f}</td></tr>"
        )
    return f"<h2>{html.escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(rows: list, start: date, end: date, provider) -> str:
    total = sum(r["cost_usd"] for r in rows)
    models = sorted({r["model"] for r in rows})
    sections = [
        _table("모델별 사용량 (OpenAI 사용량 화면의 모델별 보기와 대조)", ["모델"], group(rows, lambda r: r["model"]), lambda k: [k]),
        _table("용도별 사용량", ["용도"], group(rows, lambda r: r["purpose"]), lambda k: [PURPOSE_KO.get(k, k)]),
        _table(
            "실험별 사용량 (데이터셋 × 베이스라인 × 평가지표)",
            ["데이터셋", "베이스라인", "평가지표", "모델"],
            group(rows, lambda r: (r.get("dataset", "-"), r.get("method", "-"), r.get("metric", "-"), r["model"])),
            lambda k: list(k),
        ),
        _table("일자별 사용량 (UTC)", ["날짜"], sorted(group(rows, lambda r: r["day"])), lambda k: [k]),
        _table("작업 공간별 사용량", ["워크트리"], group(rows, lambda r: r["worktree"]), lambda k: [k]),
    ]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>API 사용 내역</title>
<style>
  :root {{ --fg:#1d1d1f; --muted:#6b6b70; --line:#d9d9de; --head:#f3f3f5; --accent:#2b5cab; }}
  body {{ font-family: -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
         color:var(--fg); background:#fff; max-width: 1080px; margin: 32px auto; padding: 0 16px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin: 16px 0 8px; }}
  .kpi {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:150px; }}
  .kpi b {{ display:block; font-size:20px; color:var(--accent); }}
  .kpi span {{ font-size:12px; color:var(--muted); }}
  h2 {{ font-size: 16px; margin: 28px 0 8px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid var(--line); padding: 6px 8px; text-align: left; }}
  th {{ background: var(--head); font-weight: 600; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .note {{ color: var(--muted); font-size: 12px; margin-top: 24px; line-height: 1.6; }}
</style></head><body>
<h1>외부 LLM API 사용 내역 — 대화 벤치마크 통합 평가 프로그램(dialogue_benchmarks) 연구 실험</h1>
<div class="sub">기간 {start} ~ {end} (UTC) · 제공사 {html.escape(provider or "전체")} · 생성 {generated}</div>
<div class="kpis">
  <div class="kpi"><span>총 비용</span><b>${total:.4f}</b></div>
  <div class="kpi"><span>총 호출 수</span><b>{len(rows):,}</b></div>
  <div class="kpi"><span>사용 모델</span><b style="font-size:14px">{html.escape(", ".join(models) or "-")}</b></div>
</div>
{''.join(sections)}
<div class="note">
  · 집계 대상: 실제로 제공사 API에 도달한 과금 호출만 (디스크 캐시로 재사용된 호출은 비용 0, 미포함).<br>
  · 비용은 호출마다 응답의 토큰 수 × 공식 단가(표준 요금, 자동 프롬프트 캐시 할인 반영)로 계산한 값이며,
    제공사 청구서와는 반올림·환율 차이 정도만 날 수 있습니다.<br>
  · 원자료: {html.escape(str(USAGE_DIR))}/*.jsonl (호출 1건 = 1줄, 시각·모델·토큰·비용·실험명 기록).
</div>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", required=True, type=date.fromisoformat)
    p.add_argument("--to", dest="end", required=True, type=date.fromisoformat)
    p.add_argument("--provider", default=None, help="openai / anthropic (default: all)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = load(args.start, args.end, args.provider)
    out = args.out or USAGE_DIR / f"report_{args.start}_{args.end}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(rows, args.start, args.end, args.provider), encoding="utf-8")

    print(f"{len(rows)} billable call(s), ${sum(r['cost_usd'] for r in rows):.4f} total, {args.start}..{args.end}")
    for model, a in group(rows, lambda r: r["model"]):
        print(f"  {model:<28} {a['calls']:>6} calls  ${a['cost_usd']:.4f}")
    print(out)


if __name__ == "__main__":
    main()
