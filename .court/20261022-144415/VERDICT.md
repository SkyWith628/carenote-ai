# The verdict: GUILTY, 5 to 1

**The case:** Case: 구현 경진대회(2026-12 중순 예상, 공고 미공개, 약 8주)에서 수상하기 위한 "케어노트AI" 설계도(Blueprint v1)는 수상 가능한 설계인가?

**The charge:** The case fails as stated: it should not go ahead the way it is described.

## How the jury voted

| Juror | Vote | Why |
|---|---|---|
| The Retired Judge | GUILTY | The plan states a "최소 수상 라인" and a fixed 8-week schedule as if they were established, yet the record holds no scoring criteria, no team size, and no measured numbers, so the plan rests on hope rather than proof and should be re-based before it proceeds as written. |
| The Small-Business Owner | GUILTY | I would not open a shop on a plan that lists six agents, a review UI, an on-device STT and a self-improving loop for eight weeks without saying who will actually do the work on Monday morning, and the "slow month" answer (drop B and C) leaves the headline pitch hollow. |
| The Statistician | GUILTY | Plans that stack three research-grade innovations, six agents and a UI into 8 weeks with an unknown team slip far more often than they land, and the headline claim of 0% unsupported values rests on only 60 self-made synthetic cases, where even zero errors only bounds the true rate at about 5% (rule of three). |
| The Teenager | GUILTY | Honestly it's a pile of agents and an empty ablation table, and the only part anyone would remember, a live demo where you talk and the form fills in with the source highlighted, doesn't exist until week 5 and has no team behind it. |
| The Parent | NOT GUILTY | As a parent weighing time, money and safety, I see eight weeks of effort with no real patient data, no payments and a plan that downgrades itself to "A deep, B/C partial" when it slips, so the downside is bounded and the plan is worth starting. |
| The Accountant | GUILTY | As an accountant I see eight weeks of build effort committed with no headcount, no model or API budget, and no per-visit cost estimate, so nobody can say who pays or how much, and the plan's own cost column is blank. |

## Charges that stuck
1. Team size is unknown, yet an 8-week schedule with six agents, a UI, on-device STT and a self-improving loop is already fixed (5 jurors).
2. The "minimum winning line" is asserted although the contest criteria and scoring are unpublished, so nobody can say what actually wins (3 jurors).
3. The pitch rests on three innovation pillars, but B and C may be cut, and the ablation table is empty, so the headline claim is unproven (3 jurors).

## What the defense got right
- The plan has a fallback: if B and C slip, it shrinks to "design + partial demo" and goes deeper on A. The downside is bounded, since the data is synthetic and nothing harmful can happen.
- Golden set, Eval Runner and ablation table are useful whatever the final scoring turns out to be, and putting the LLM in an extract-only role with code rules and human sign-off answers the main hallucination worry.

## Conditions for acquittal
1. Write down the actual team and an owner for A, the UI and the demo, with person-weeks per week-block. Cost and latency per visit for the multi-model pipeline should be estimated too. Doing this takes one afternoon.
2. Reframe the pitch as "A plus a live demo". List B and C as stretch goals, and delete the "minimum winning line" wording until the announcement is out. Then re-check the plan against the criteria the day they appear.
3. Move a thin end-to-end demo (voice in, evidence-linked draft out) to week 2 or 3, and add the Verifier and the other agents after it.

## The sentence
The court finds the plan ambitious but not hopeless. Do not throw it away; trim it. Fix the team facts this week, make the demo plus the ablation-backed core the pitch, and treat everything else as optional. Come back when you can say who builds what.

_The full trial is in /Users/sktwith/Library/Mobile Documents/com~apple~CloudDocs/WorkSpace_M/projects/carenote-ai/.court/20261022-144415: the opening statements, the rebuttals and every juror's vote._

_Made with [/court](https://github.com/alexyc9381/court-skill), a free Claude Code skill by Alex Chen ([@nocodealex](https://instagram.com/nocodealex))._
