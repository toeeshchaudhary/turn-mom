# ChadGPT — working notes

## ⚠️ BIGGER ISSUE FIRST (TBD)
There's a larger problem to address before the voice-quality retrain. Owner to describe.
> _(placeholder — fill in from next message)_

---

## Model-improvement plan (voice quality) — PARKED until the bigger issue is handled

The model's format/behavior is good (9/9 eval). Remaining problems are **voice quality baked
into the teacher labels**, not prompt/tuning issues → must fix the data + retrain.

Ordered steps:
1. **Diagnose current labels** (`data/interim/labeled.jsonl`, ~10 min, no GPU): measure
   - % suggestions opening with acknowledgement filler ("Got it/Alright/Great/Glad to hear")
   - % ineligible replies that NAME the reason ("because of the bankruptcy")
   - % using "primary residence"/"investment property" verbatim
   - is it worse in transcript-sourced vs bonzo-sourced examples?
   → tells us if a prompt-only fix suffices or if the 14B teacher is the ceiling.
2. **Finalize ALL teacher rules in one pass** so a single relabel fixes everything:
   - ✅ backchannel rule — PREPPED (`label_with_teacher.py` NO_BACKCHANNEL + `audit_gate.py` reject 2+ filler)
   - ⬜ don't name the ineligibility reason (ineligible stage)
   - ⬜ never say "primary residence"/"investment property" verbatim (property_use)
3. **Relabel → retrain → swap**: `run.sh` (relabel) → `unsloth_train.py` → merge → replace `out/merged`.

### PENDING DECISION — teacher model for the relabel
The **14B teacher is likely the source of the bland, acknowledgement-heavy voice.** Options:
- **Qwen2.5-32B** (local, stronger, free on GPU, ~1.5-2x slower) — best quality/effort local.
- **Frontier API (Claude/GPT)** — best voice, fastest setup, costs $ at 34k scale (or label a subset).
- **Stay on 14B, prompt-only** — cheapest; do the diagnostic first to see if it's enough.
_Decision: NOT MADE YET — revisit after the bigger issue._

---

## Known quality issues (owner feedback)
- Call-style **acknowledgement fillers** everywhere (main complaint).
- Ineligible replies **name the reason** (prompt says don't reveal).
- Says **"primary residence"** verbatim (prompt says rephrase).

## Where we are (2026-07-17)
- Model: Mistral-24B LoRA, serving as `naf` on :8000. 9/9 eval.
- UI (`ui/`): Assist (3 cards) + Chat toggle, **orchestrator** (screening engine + LLM
  answer-detector) so context advances / collects — stateless model unchanged.
- Owner roadmap (context, not now): 3-rec CSS engine → LO engine → 1 rec → supervised → autonomous.
  Autonomous needs tool-calling, context mgmt, product knowledge (RAG), robustness/red-teaming,
  empathy + escalation judgment. Build so we don't paint into a corner.
