# OMLX Lightning MTP: StatisticsError Crash — Empty Baseline Must Never Park

> **Independent deep-dive report & runnable fix patch**
> Target: OMLX (github.com/jundot/omlx), Apache-2.0
> Discovered, reproduced, and fixed **independently**. Zero affiliation with the upstream repository or its author.

---

## 〇、致谢 (Acknowledgement)

A sincere thank-you to **Jun Kim (jundot)** and the OMLX project once again. The MTP batch-policy / parking machinery here is genuinely intricate — the depth search, pooled acceptance, and cross-cohort `ParkMemory` backoff are the kind of subtle systems-level design that most projects never attempt. We are sharing this edge case in the hope it makes an already-excellent decoder even more robust under concurrency.

---

## 一、概述 (Summary)

Under **multi-request / batched Lightning MTP**, a freshly rebuilt cohort policy can reach its park decision with an **empty baseline** deque and crash the engine step with:

```
statistics.StatisticsError: no median for empty data
```

The crash appears in the `_run_verify_cycle_batched` path. Our fix is a one-line guard in `BatchPolicy.should_park()`: **never park without a calibrated `standard` baseline**. It is small, surgical, and preserves all existing backoff/retirement behavior.

---

## 二、受影响版本 (Affected Versions)

| Version | Status |
|---------|--------|
| 0.7.1.dev1 | ✅ Bug present (no empty-baseline guard) |
| 0.7.1.dev2 | ✅ Reproduced (bug present) |
| 0.7.2 | ✅ Bug present in upstream main |
| 0.7.3 | ✅ Bug present in upstream `main` (`BatchPolicy.should_park` still returns `not self.fixed and (self.losing >= 16 or self.clear_losing >= 4)`, no empty-baseline guard) |

---

## 三、根因 (Root Cause)

`BatchPolicy` keeps a rolling `standard` deque (maxlen 8) that stores measured ordinary-decode timings:

```python
self.standard = deque(maxlen=8)
self.standard_warmup = 2
```

`score(0)` (the ordinary-decode baseline) is:

```python
def score(self, depth):
    if depth == 0:
        return len(self.uids) / median(self.standard) if self.standard else 0.0
    ...
```

When `self.standard` is empty, `score(0)` returns `0.0`. On a **freshly rebuilt cohort** (e.g. a new request joins, or a policy is recreated after a park), the deque can be empty at the moment a decision is evaluated. Then:

- `self.score(best) <= self.score(0) * 1.03` compares against `0.0`, so **any** MTP depth score (≥0) is treated as "losing".
- `self.losing` and `self.clear_losing` **accumulate blindly** on every decision.
- When the counters reach `16` (`losing`) or `4` (`clear_losing`), `should_park()` returns `True`.

The crash then happens **in the batched verify loop** (`batch_generator._run_verify_cycle_batched`), where a parked policy is logged by calling `median(policy.standard)`:

```python
# batch_generator.py, _run_verify_cycle_batched, after policy.should_park()
median(policy.standard),   # StatisticsError: no median for empty data
```

This `median(...)` call has **no empty-deque guard**. So the sequence is: empty `standard` → `score(0)==0.0` → `losing` climbs blindly → `should_park()` returns `True` → the batched verify loop logs the park with `median(policy.standard)` over the empty deque → `statistics.StatisticsError: no median for empty data` → the whole engine step dies.

The root cause is that **parking is allowed when there is no calibrated baseline to park against**. The counters are meaningless until a baseline exists.

---

## 四、修复方案 (The Fix)

Require a non-empty `standard` baseline before a policy is allowed to park:

```python
def should_park(self):
    return (
        not self.fixed
        and bool(self.standard)
        and (self.losing >= 16 or self.clear_losing >= 4)
    )
```

We deliberately keep everything else intact: once a baseline exists, the original `losing >= 16 or clear_losing >= 4` logic — and the `ParkMemory` cross-cohort backoff — behave exactly as designed.

**A ready-to-apply patch is provided in this directory:**
`0001-fix-empty-baseline-park-crash.patch`

Apply with:

```bash
cd <omlx-source>
git apply /path/to/0001-fix-empty-baseline-park-crash.patch
```

(or `patch -p1 < 0001-fix-empty-baseline-park-crash.patch`)

---

## 五、验证 (Verification)

- The guard is **purely additive**: it only changes the outcome when `self.standard` is empty, a state in which parking was already invalid.
- Existing park/retire/backoff unit scenarios still pass — the `bool(self.standard)` check short-circuits to `False` only before calibration, and to the original expression after.
- Reproduced crash (`StatisticsError: no median for empty data`) is eliminated under the same multi-request batched setup.

---

## 六、结语 (Closing note)

This is a classic "empty-collection" crash that only shows up under real multi-request concurrency — the kind of edge case single-stream dev prompts never reach. We ship a minimal, safe guard that makes the parking machinery robust regardless of cohort lifecycle.

**License:** This report and the accompanying patch are released under **Apache-2.0**, consistent with OMLX's own license. OMLX remains the work of its original author (Jun Kim / jundot).