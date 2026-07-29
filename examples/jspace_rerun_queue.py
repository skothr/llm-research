#!/usr/bin/env python3
"""Sequential, VRAM-gated, pausable runner for the arc-04 lens refits.

The RTX 2080 (8 GB) fits exactly one lens at a time and the fits are long
(2-16 h), so the re-run is a queue rather than a fan-out. This script is that
queue: it waits for the GPU, runs each job to completion, and refuses to start
the next one on a failure rather than silently producing a partial set.

Why a committed script and not a shell one-liner: the 2026-07-29 C4-redaction
re-run (`plans/2026-07-29-c4-redaction-rerun.md`) needs three fits in a fixed
order, each with model-specific batch/offload knobs established by measurement
(`observations/2026-07-18-fit-cost-calibration.md`). Hand-launching them across
sessions is how the knobs drift.

PAUSING (the GPU is also the machine's gaming GPU)
--------------------------------------------------
    python examples/jspace_rerun_queue.py --pause     # yield the GPU
    python examples/jspace_rerun_queue.py --status
    python examples/jspace_rerun_queue.py --resume    # pick up where it left off

`--pause` signals the running queue, which stops the in-flight fit *at once* —
it does NOT wait for the next checkpoint, so the GPU frees within seconds.
Everything already checkpointed is kept and the rest is redone on resume, so a
pause costs at most one checkpoint interval (<=19 min at 1.5B, <=47 min at 7B —
see `checkpoint_every` per job). `--resume` clears the flag; the queue re-waits
for VRAM and restarts the fit from its last checkpoint.

Resumability is three layers: this script SKIPS a job whose final artifact
exists, restarts a paused job from its checkpoint, and `jspace_fit_lens.py`
itself resumes from that checkpoint. Re-invoking after any interruption —
pause, crash, reboot — is always safe and never redoes finished work.

usage:
    python examples/jspace_rerun_queue.py [--dry-run] [--only TAG ...]
    python examples/jspace_rerun_queue.py --pause | --resume | --status
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARC = REPO / "research" / "arcs" / "04_jspace"
CACHE = ARC / "data" / "cache"
LOGS = CACHE / "logs"
PAUSE_FLAG = CACHE / "PAUSE"

# Two consecutive reads above the threshold before starting — a single read can
# catch the instant between one job releasing VRAM and a desktop app taking it.
VRAM_POLL_SECONDS = 60
VRAM_CONSECUTIVE_READS = 2
# How often to notice a --pause while a fit is running.
PAUSE_POLL_SECONDS = 10
# Grace period for the fit to die on SIGTERM before SIGKILL.
TERM_GRACE_SECONDS = 30


@dataclass(frozen=True)
class Job:
    """One lens fit. `free_mib` is measured peak + headroom, not a guess."""

    tag: str
    model: str
    mode: str
    dim_batch: int
    prompts: str
    corpus_tag: str | None
    free_mib: int
    hours: float
    checkpoint_every: int
    extra: list[str] = field(default_factory=list)

    @property
    def out(self) -> Path:
        model_slug = "qwen2.5-7b" if "7B" in self.model else "qwen2.5-1.5b"
        suffix = f"_{self.corpus_tag}" if self.corpus_tag else ""
        return CACHE / f"jlens_{model_slug}_{self.mode}_n100{suffix}.pt"

    @property
    def max_pause_loss_minutes(self) -> float:
        """Worst-case rework if paused right before a checkpoint write."""
        return self.hours * 60 / 100 * self.checkpoint_every

    def argv(self) -> list[str]:
        argv = [
            sys.executable,
            str(REPO / "examples" / "jspace_fit_lens.py"),
            "--model", self.model,
            "--mode", self.mode,
            "--dim-batch", str(self.dim_batch),
            "--n-prompts", "100",
            "--checkpoint-every", str(self.checkpoint_every),
            "--prompts", str(ARC / "data" / self.prompts),
        ]  # fmt: skip
        if self.corpus_tag:
            argv += ["--corpus-tag", self.corpus_tag]
        return argv + self.extra


# Order matters: cheapest first, so a knob mistake surfaces in 2 h, not 16 h.
JOBS: list[Job] = [
    Job(
        tag="c4en-1.5b",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        mode="bf16",
        dim_batch=8,
        prompts="fitting_prompts_c4en_n1000.json",
        corpus_tag="c4en",
        free_mib=5900,
        # 3.2 h from calibration (115 s/prompt x 100); observed 3.12 h on
        # 2026-07-29. The plan originally said 2.2 h — a transcription error.
        hours=3.2,
        checkpoint_every=10,
    ),
    Job(
        tag="wikitext-1.5b",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        mode="bf16",
        dim_batch=8,
        prompts="fitting_prompts_wikitext103_n1000.json",
        corpus_tag=None,
        free_mib=5900,
        hours=3.2,
        checkpoint_every=10,
    ),
    Job(
        # 7B checkpoints more often than the 1.5B jobs despite each write being
        # ~1.4 GB: at 559 s/prompt, checkpoint_every=10 would put ~1.5 h of
        # rework behind a single pause.
        tag="wikitext-7b",
        model="Qwen/Qwen2.5-7B-Instruct",
        mode="nf4",
        dim_batch=2,
        prompts="fitting_prompts_wikitext103_n1000.json",
        corpus_tag=None,
        free_mib=6600,
        hours=16.3,
        checkpoint_every=5,
        extra=["--offload-lm-head"],
    ),
]


def free_vram_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def paused() -> bool:
    return PAUSE_FLAG.exists()


def wait_while_paused(label: str) -> None:
    if not paused():
        return
    print(f"[queue] PAUSED at {label} — waiting for --resume", flush=True)
    while paused():
        time.sleep(PAUSE_POLL_SECONDS)
    print(f"[queue] resumed at {label}", flush=True)


def wait_for_vram(need_mib: int, label: str) -> None:
    """Block until VRAM is free on N consecutive reads.

    Also covers a fit launched outside this queue: it holds VRAM, so the gate
    waits it out without inspecting PIDs (which a bare `pgrep -f` would get
    wrong by matching itself).
    """
    streak = 0
    while streak < VRAM_CONSECUTIVE_READS:
        wait_while_paused(label)
        free = free_vram_mib()
        if free >= need_mib:
            streak += 1
            print(
                f"[queue] {label}: {free} MiB free ({streak}/{VRAM_CONSECUTIVE_READS})",
                flush=True,
            )
        else:
            if streak:
                print(
                    f"[queue] {label}: dropped to {free} MiB, streak reset", flush=True
                )
            streak = 0
        if streak < VRAM_CONSECUTIVE_READS:
            time.sleep(VRAM_POLL_SECONDS)


def _stop(proc: subprocess.Popen[bytes], label: str) -> None:
    """SIGTERM, then SIGKILL on grace expiry. Releases the GPU either way."""
    print(f"[queue] stopping {label} (pause requested)", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        print(
            f"[queue] {label} ignored SIGTERM after {TERM_GRACE_SECONDS}s, killing",
            flush=True,
        )
        proc.kill()
        proc.wait()


def run_once(job: Job) -> int | None:
    """Run the fit to completion. Returns exit code, or None if paused out."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"fit_{job.tag}.log"
    # Append: a paused-and-resumed job writes several segments to one log, and
    # truncating would discard the record of what ran before the pause.
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== segment start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.flush()
        proc = subprocess.Popen(job.argv(), stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                return proc.wait(timeout=PAUSE_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if paused():
                    _stop(proc, job.tag)
                    return None


def run(job: Job, dry_run: bool) -> bool:
    if job.out.exists():
        print(f"[queue] SKIP {job.tag}: {job.out.name} already present", flush=True)
        return True
    if dry_run:
        print(
            f"[queue] DRY {job.tag} (~{job.hours} h, pause costs "
            f"<={job.max_pause_loss_minutes:.0f} min): {' '.join(job.argv())}",
            flush=True,
        )
        return True

    started = time.time()
    while True:
        wait_for_vram(job.free_mib, job.tag)
        print(
            f"[queue] START {job.tag} (~{job.hours} h, checkpoint every "
            f"{job.checkpoint_every} prompts) -> {LOGS / f'fit_{job.tag}.log'}",
            flush=True,
        )
        rc = run_once(job)
        if rc is None:
            # Paused mid-fit. Block here, then loop to re-wait for VRAM and
            # restart from the checkpoint.
            wait_while_paused(job.tag)
            continue
        break

    elapsed = (time.time() - started) / 3600
    if rc != 0:
        print(
            f"[queue] FAIL {job.tag}: exit {rc} after {elapsed:.2f} h "
            f"(incl. any pause) — see {LOGS / f'fit_{job.tag}.log'}",
            flush=True,
        )
        return False
    if not job.out.exists():
        # Exit 0 without the artifact means the fit's own success path is
        # broken; treat it as failure rather than letting the queue advance.
        print(f"[queue] FAIL {job.tag}: exit 0 but {job.out.name} missing", flush=True)
        return False
    print(f"[queue] DONE {job.tag} in {elapsed:.2f} h (est {job.hours})", flush=True)
    return True


def cmd_status() -> int:
    print(
        f"pause flag : {PAUSE_FLAG} {'SET (paused)' if paused() else 'clear (running)'}"
    )
    try:
        print(f"free VRAM  : {free_vram_mib()} MiB")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"free VRAM  : unavailable ({e})")
    for job in JOBS:
        state = "done" if job.out.exists() else "pending"
        ckpt = CACHE / f"{job.out.stem}.ckpt.pt"
        if state == "pending" and ckpt.exists():
            state = f"pending (checkpoint {time.strftime('%H:%M', time.localtime(ckpt.stat().st_mtime))})"
        print(f"  {job.tag:<16} {state:<32} ~{job.hours} h")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--pause",
        action="store_true",
        help="yield the GPU after the current checkpoint",
    )
    mode.add_argument("--resume", action="store_true", help="clear the pause flag")
    mode.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", metavar="TAG", help="run only these job tags")
    args = ap.parse_args()

    if args.status:
        return cmd_status()
    if args.pause:
        CACHE.mkdir(parents=True, exist_ok=True)
        PAUSE_FLAG.touch()
        print(f"[queue] pause requested ({PAUSE_FLAG}).")
        print(
            "        The in-flight fit stops immediately and VRAM frees within "
            "seconds; work since its\n        last checkpoint is redone on "
            "--resume."
        )
        return 0
    if args.resume:
        PAUSE_FLAG.unlink(missing_ok=True)
        print("[queue] pause cleared — the waiting queue picks up from its checkpoint.")
        return 0

    jobs = [j for j in JOBS if not args.only or j.tag in args.only]
    if args.only:
        unknown = set(args.only) - {j.tag for j in JOBS}
        if unknown:
            ap.error(f"unknown job tag(s): {sorted(unknown)}")

    total = sum(j.hours for j in jobs if not j.out.exists())
    print(f"[queue] {len(jobs)} job(s), ~{total:.1f} h of unfinished work", flush=True)
    print(f"[queue] pause any time: {sys.argv[0]} --pause", flush=True)

    for job in jobs:
        if not run(job, args.dry_run):
            print("[queue] ABORT — not starting later jobs", flush=True)
            return 1
    print("[queue] all jobs complete", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
