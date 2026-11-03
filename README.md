# omlx-patches

**Independent deep-dive bug reports & runnable fix patches for OMLX.**

This repository is an **independent** study of [OMLX](https://github.com/jundot/omlx) —
the remarkable MLX-accelerated LLM inference server by **Jun Kim (jundot)**.

We are heavy, happy users of OMLX. Jun's work — bringing modern speculative
decoding (Lightning MTP) and MLX acceleration to Apple Silicon — is genuinely
exceptional engineering that benefits the entire ML community. This repo is our
way of giving back: we push OMLX to extreme edge cases (ultra-long contexts,
multi-request concurrency, large-RAM workstations) that ordinary 2K-token dev
prompts never reach, and when we find something, we reproduce it, root-cause it,
fix it, and share the patch here.

> **Independence.** Everything here was discovered, reproduced, and fixed
> independently. This repository has **zero affiliation** with the upstream
> `jundot/omlx` repository or its author. We file these publicly as our own
> findings; whether upstream adopts them is entirely their decision.

---

## Bug Reports

| # | Bug | Severity | Upstream status | Patch |
|---|-----|----------|-----------------|-------|
| 01 | [DSpark RMSNorm dimension collision on ultra-long-context MTP handoff](bug01_mtp_rmsnorm/REPORT.md) | **Fatal** (crash → 1.5 tok/s + abort) | **Unfixed** in `main` | [patch](bug01_mtp_rmsnorm/0001-fix-mtp-rmsnorm-handoff.patch) |
| 02 | [StatisticsError crash — empty baseline must never park](bug02_statistics_error/REPORT.md) | High (engine-step crash under concurrency) | **Unfixed** in `main` | [patch](bug02_statistics_error/0001-fix-empty-baseline-park-crash.patch) |
| 03 | [process_memory_enforcer over-aggressive self-kill on large-RAM machines](bug03_memory_guard/REPORT.md) | **Fatal** (crash loop, 8–23/day) | **Unfixed** in `main` | [patch](bug03_memory_guard/0001-fix-memory-guard-overshoot.patch) |

Each report contains: **full technical analysis** (version, symptom, root cause,
fix), a **ready-to-apply `.patch`**, the **original & fixed source** for
side-by-side comparison, and **sanitized reproduction logs**.

---

## License

All reports and patches in this repository are released under the
**[Apache-2.0](LICENSE)** license, consistent with OMLX's own license.

OMLX itself remains the work of its original author, **Jun Kim (jundot)**,
licensed under Apache-2.0.