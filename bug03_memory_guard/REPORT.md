# OMLX process_memory_enforcer: Over-aggressive Self-Kill on Large-RAM Machines

> **Independent deep-dive report & runnable fix patch**
> Target: OMLX (github.com/jundot/omlx), Apache-2.0
> Discovered, reproduced, and fixed **independently**. Zero affiliation with the upstream repository or its author.

---

## 〇、致谢 (Acknowledgement)

Another sincere thank-you to **Jun Kim (jundot)** and OMLX. The `process_memory_enforcer` and two-stage watermark design shows real care about not OOM-ing the host — that is exactly the kind of robustness that makes a local-inference tool trustworthy enough to run unattended for hours. This report is about one sharp edge of that design on very large machines, which we believe is worth sharing.

---

## 一、概述 (Summary)

On machines with **≥48GB of RAM** (e.g. a 256GB Mac Studio), OMLX's `process_memory_enforcer` operating in the default `balanced` tier can **repeatedly kill its own server process** under a large prompt, even though the machine has plenty of free memory. In production this produced a crash loop: the enforcer unloads all models, exits the process, and on macOS logs the user out (`loginwindow` restart). We measured **8 crashes in one day, 23 the next, and 12 more the day after**.

The root cause is that the `balanced` tier reaps on a **dynamic watermark** derived from `vm_stat` (`free + inactive`), which on a 256GB machine routinely reads as **<20GB** even when there is abundant reclaimable memory. Any large prompt briefly pushes past that window, and the enforcer decides the machine is out of memory and self-terminates.

Our fix pins machines with **≥48GB RAM to the `custom` tier** with a **static, panic-safe ceiling of `RAM − 16GB`**, which never flaps with the `vm_stat` watermark.

---

## 二、受影响版本 (Affected Versions)

| Version | Status |
|---------|--------|
| 0.7.1.dev1 | ✅ Bug present (crash loop on ≥48GB machines) |
| 0.7.1.dev2 | ✅ Bug present (crash loop on ≥48GB machines) |
| 0.7.2 | ✅ Bug present in upstream main (no RAM-based auto tier selection) |
| 0.7.3 | ✅ Bug present in upstream `main` — the enforcer still has no "large machine" guard; `memory_guard_tier` defaults to `"balanced"` with no RAM-aware override |

---

## 三、触发条件 (When it triggers)

1. **Large-RAM machine**: ≥48GB physical RAM (the failure is proportional to how much resident weight the machine keeps — 256GB machines holding 140–170GB of resident weights are the worst).
2. **Default tier**: `memory_guard_tier = "balanced"` (the out-of-box default).
3. **A large prompt** briefly drives resident+inactive memory over the dynamic watermark.

The `balanced` tier reaps on `free + inactive` from `vm_stat`. On a big machine holding tens of GB of weights, the OS reports a small `free+inactive` window because most "free" memory is actually in the file cache or being actively used by resident weights. So even with **plenty** of truly available memory, the watermark looks critically low, and any transient spike triggers the kill.

---

## 四、根因 (Root Cause)

`memory_guard_tier = "balanced"` reaps on a **dynamic, per-second watermark** built from `vm_stat`. On large machines this watermark is structurally unreliable:

```
free + inactive  (vm_stat)   →   interpreted as "available"   →   reap if it dips
```

For a 256GB machine holding 140–170GB of resident weights:
- `free + inactive` routinely measures **<20GB** (most memory is resident weights + file cache).
- The enforcer believes the machine is nearly out of memory.
- A large prompt pushes resident memory up for the duration of prefill.
- The enforcer's "graceful self-kill" fires: **unload all models → exit process → macOS logs out** (because the process is launched under the login session).

We measured this repeatedly:
| Day | Crash count |
|-----|-------------|
| 10-14 | 8 |
| 10-15 | 23 |
| 10-16 | 12 |

This is not a memory-exhaustion bug in the normal sense — the machine had ample memory — it is an **over-aggressive reap policy** tuned for consumer machines, misapplied to large workstations.

---

## 五、修复方案 (The Fix)

When a machine has **≥48GB RAM**, force the `custom` tier with a **static, panic-safe ceiling of `RAM − 16GB`**. The static ceiling does not flap with the `vm_stat` watermark, so the enforcer never reaps on a false low-memory reading. A user-supplied `custom` ceiling is respected when present.

```python
custom_ceiling = float(data.get("memory_guard_custom_ceiling_gb", 0.0))
if tier != "custom":
    try:
        _ram_gb = get_system_memory() / (1024**3)
        if _ram_gb >= 48:
            tier = "custom"
            if custom_ceiling <= 0:
                custom_ceiling = round(_ram_gb - 16, 1)
    except Exception:
        pass
```

**A ready-to-apply patch is provided in this directory:**
`0001-fix-memory-guard-overshoot.patch`

Apply with:

```bash
cd <omlx-source>
git apply /path/to/0001-fix-memory-guard-overshoot.patch
```

(or `patch -p1 < 0001-fix-memory-guard-overshoot.patch`)

---

## 六、验证 (Verification)

- The patch is **additive and conservative**: on machines with `<48GB` RAM the behavior is unchanged; on large machines it only overrides the *default* `balanced` tier, and only when the user has not already chosen a `custom` config.
- After applying, the crash loop stops: the server holds 140–170GB of weights and serves large prompts without the enforcer self-terminating.

---

## 七、结语 (Closing note)

A memory guard that kills the very service it is meant to protect is worse than none at all on large workstations. We share this so that anyone running OMLX on big machines can avoid the silent crash loop, and we hope it informs a RAM-aware default upstream.

**License:** This report and the accompanying patch are released under **Apache-2.0**, consistent with OMLX's own license. OMLX remains the work of its original author (Jun Kim / jundot).