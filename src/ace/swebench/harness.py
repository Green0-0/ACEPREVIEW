"""Sandboxed SWE-bench Verified evaluation with Apptainer (no docker needed).

``run_instance(instance, patch, log_dir=...)`` applies ``patch`` to /testbed in
the instance's eval image, runs the swebench-generated eval script and grades
the captured log with swebench's own grader (swebench 5.0.2:
``swebench.harness.utils.make_test_spec`` + ``swebench.harness.grading
.get_eval_report``). Instances come from the new-format HF dataset
``SWE-bench/SWE-bench_Verified`` (same 500 ids as princeton-nlp/..., plus the
``eval_script``/``log_parser``/``eval_type`` columns that swebench 5 needs);
``load_verified()`` caches it at /usr/project/xtmp/kg349/ace/data/verified.jsonl.

Images: one SIF per instance at ``$ACE_SWEBENCH_IMAGES/<instance_id>.sif``
(default /usr/project/xtmp/kg349/ace/images), converted from Epoch AI's
optimized images ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>
(see slurm/pull_images.sbatch). SIFs are mounted read-only via squashfuse, so
start-up is ~1-2 s and nothing is copied per run.

Isolation (what the evaluated code can and cannot do):
  * ``apptainer exec --containall --cleanenv --no-home``: private PID/IPC
    namespaces, no host env vars (only the few we --env explicitly), no $HOME,
    minimal /dev, no automatic host binds (no /home, /usr/project, cwd, ...).
  * ``--net --network none``: a fresh network namespace with only loopback, so
    no outbound sockets (verified: connect() -> ENETUNREACH).
  * ``--fakeroot`` (root-mapped user namespace, no subuids on this cluster):
    the eval scripts need root inside (locale-gen, writes under /testbed);
    outside, the processes are still just our uid with no extra capabilities.
  * Root filesystem: SIF (read-only) + a fresh per-run ``--overlay`` upper
    directory on node-local scratch, so /testbed and the rest of / are
    writable without the 64 MB --writable-tmpfs session limit. ``--workdir``
    points /tmp and /var/tmp at the same per-run scratch dir.
  * The only host path bound in is ``<run dir>/in`` (read-only: patch, eval
    script, driver script, a localhost-only /etc/hosts). Nothing the container
    writes is read back except its stdout/stderr, captured through a pipe
    (capped at 64 MB) into the host log file, which the host-side swebench
    grader parses. Patch success/failure is signalled with a per-run random
    nonce, printed before any repository code runs.
  * Wall-clock timeout is enforced from outside: the apptainer process runs in
    its own session; on timeout (and after every run) every process of that
    session is SIGTERMed then SIGKILLed -- this includes apptainer's
    squashfuse/fuse-overlayfs daemons, which reparent to init and would
    otherwise linger. The container's PID namespace dies with its init.
    The run dir is always deleted (verified: no leftover processes/dirs).
  * CPU/memory are bounded by the Slurm allocation's cgroup (apptainer
    --memory/--cpus need cgroup delegation, unavailable unprivileged here).

Deviation from the docker harness: with no network, ``pip install -e .`` steps
in eval scripts run with ``PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=0`` (pip's
inverted env semantics: this means --no-build-isolation). The packages are
already installed editable in the image, so pure-Python reinstalls succeed
offline; projects whose build needs packages absent from the env (e.g. astropy's
pinned setuptools/extension-helpers) fail fast and keep the image's existing
editable install. Consequence: a patch that edits C/Cython extension sources is
not recompiled (the docker harness would recompile). Gold validation measures
the overall effect.

Known environment gap (open): Ghostscript 9.55 in the matplotlib images fails
with /invalidfileaccess ("Permission denied") on PDF/EPS files under /testbed or
/ (but not under /tmp or $HOME) inside this sandbox, so matplotlib tests with
[pdf]/[eps] image comparisons fail even for the gold patch (19 instances; see
runs/gold_validation/summary). Both agents are affected identically.

Network-dependent tests (psf/requests hitting httpbin.org, sphinx linkcheck)
fail offline even with the gold patch. ``run_instance(..., network=True)`` /
``swebench_eval.py --network-repos`` keeps the host network for chosen repos;
it is OFF by default, weakens isolation, and was not used for any results here.

Known limits: the graded log is still produced by the evaluated code (as in
upstream SWE-bench), so a patch could in principle print fake test results;
swebench 5's exit-code cross-check catches the simple forgeries. The images'
own /etc/hosts is empty (docker injects one), so we bind a localhost-only one.

Timing (measured on compsci nodes, 8 parallel workers on 16 CPUs): container
start + overlay setup ~1-2 s; whole run_instance wall time is typically 5-15 s
(requests, flask, pytest, pylint, sympy, sklearn, sphinx, astropy, most
django), 30-60 s (xarray, seaborn, matplotlib test_axes), up to ~5 min for
instances that run the full django suite (django__django-10097). Parallel
workers did not slow individual runs noticeably.

Validation: gold patch resolved 16/16 diverse instances (12 repos); empty patch
unresolved 15/16 -- django__django-10097 is "resolved" by the empty patch
(dataset quirk: its FAIL_TO_PASS list holds 438 tests that already pass at the
base commit). Full 500-instance gold run: runs/gold_validation/.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ACE_XTMP = Path(os.environ.get("ACE_XTMP", "/usr/project/xtmp/kg349/ace"))
IMAGES_DIR = Path(os.environ.get("ACE_SWEBENCH_IMAGES", ACE_XTMP / "images"))
VERIFIED_JSONL = ACE_XTMP / "data" / "verified.jsonl"
APPTAINER = os.environ.get("ACE_APPTAINER", shutil.which("apptainer") or "/bin/apptainer")
MAX_LOG_BYTES = 64 * 1024 * 1024

# Same fallback chain as swebench.harness.run_evaluation.GIT_APPLY_CMDS.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]

def load_verified(path: Path = VERIFIED_JSONL) -> list[dict]:
    """All 500 SWE-bench Verified instances (new-format dataset), cached as JSONL."""
    path = Path(path)
    if not path.exists():
        from datasets import load_dataset

        rows = list(load_dataset("SWE-bench/SWE-bench_Verified", split="test"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sif_path(instance_id: str) -> Path:
    return IMAGES_DIR / f"{instance_id}.sif"


def _scratch_root() -> str:
    for cand in (os.environ.get("ACE_SWEBENCH_SCRATCH"), os.environ.get("TMPDIR"), "/tmp"):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return tempfile.gettempdir()


def _driver_script(nonce: str, has_patch: bool) -> str:
    lines = ["#!/bin/bash", "cd /testbed || exit 97"]
    if has_patch:
        chain = " || ".join(f"{c} /ace_in/patch.diff" for c in GIT_APPLY_CMDS)
        lines += [
            f"if ( {chain} ) > /tmp/ace_apply.log 2>&1; then",
            f"  echo ACE_PATCH_APPLIED_{nonce}",
            "else",
            "  cat /tmp/ace_apply.log",
            f"  echo ACE_PATCH_FAILED_{nonce}",
            "  exit 98",
            "fi",
        ]
    else:
        lines.append(f"echo ACE_PATCH_EMPTY_{nonce}")
    lines.append("exec /bin/bash /ace_in/eval.sh")
    return "\n".join(lines) + "\n"


def _apptainer_cmd(sif: Path, run_dir: Path, repo: str = "", network: bool = False) -> list[str]:
    env = {
        "PIP_NO_INDEX": "1",
        "PIP_NO_BUILD_ISOLATION": "0",  # pip quirk: "0" == --no-build-isolation
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HOME": "/root",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    cmd = [
        APPTAINER, "-s", "exec",
        "--containall", "--cleanenv", "--no-home",
        "--fakeroot", "--ignore-fakeroot-command",
        *([] if network else ["--net", "--network", "none"]),
        "--no-mount", "home,cwd,bind-paths,hostfs",
        "--overlay", str(run_dir / "overlay"),
        "--workdir", str(run_dir / "work"),
        "--bind", f"{run_dir / 'in'}:/ace_in:ro",
        # Images ship an empty /etc/hosts (docker injects one at run time) and
        # --no-mount bind-paths drops apptainer's host copy: provide localhost.
        "--bind", f"{run_dir / 'in' / 'hosts'}:/etc/hosts:ro",
        "--pwd", "/",
    ]
    if network:
        cmd += ["--bind", "/etc/resolv.conf:/etc/resolv.conf:ro"]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    return cmd + [str(sif), "/bin/bash", "/ace_in/run.sh"]


def _pump(stream, out_path: Path) -> None:
    written = 0
    with open(out_path, "wb") as out:
        while chunk := stream.read(65536):
            if written < MAX_LOG_BYTES:
                out.write(chunk[: MAX_LOG_BYTES - written])
                written += len(chunk)
        if written >= MAX_LOG_BYTES:
            out.write(b"\n[ace: log truncated]\n")


def run_instance(
    instance: dict,
    patch: str,
    *,
    timeout: int = 1800,
    log_dir: Path,
    sif: Path | None = None,
    tag: str = "",
    network: bool = False,
) -> dict:
    """Evaluate ``patch`` on one SWE-bench instance inside an isolated container.

    ``network=True`` (off by default) keeps the host network namespace, for the
    few instances whose tests need the internet (psf/requests -> httpbin.org,
    sphinx linkcheck); everything else stays isolated.
    """
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    iid = instance["instance_id"]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{iid}{('.' + tag) if tag else ''}.log"
    result = {"instance_id": iid, "resolved": False, "status": "error",
              "tests_status": {}, "duration_s": 0.0, "log_path": str(log_path)}
    t0 = time.monotonic()
    sif = Path(sif) if sif else sif_path(iid)
    if not sif.exists():
        result["error"] = f"missing image {sif}"
        log_path.write_text(result["error"] + "\n")
        return result

    patch = patch or ""
    has_patch = bool(patch.strip())
    if has_patch and not patch.endswith("\n"):
        patch += "\n"
    spec = make_test_spec(instance)
    nonce = secrets.token_hex(16)
    run_dir = Path(tempfile.mkdtemp(prefix=f"ace-swe-{iid}-", dir=_scratch_root()))
    proc = None
    try:
        for sub in ("in", "overlay", "work"):
            (run_dir / sub).mkdir()
        (run_dir / "in" / "patch.diff").write_text(patch, encoding="utf-8", errors="surrogateescape")
        (run_dir / "in" / "eval.sh").write_text(spec.eval_script)
        (run_dir / "in" / "run.sh").write_text(_driver_script(nonce, has_patch))
        (run_dir / "in" / "hosts").write_text(
            "127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n")
        for f in (run_dir / "in").iterdir():
            f.chmod(0o444)
        (run_dir / "in").chmod(0o555)

        raw_log = run_dir / "container.log"
        proc = subprocess.Popen(
            _apptainer_cmd(sif, run_dir, instance.get("repo", ""), network), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(run_dir),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "APPTAINER_CACHEDIR": str(run_dir / "work"),
                 "APPTAINER_TMPDIR": str(run_dir / "work")},
        )
        pump = threading.Thread(target=_pump, args=(proc.stdout, raw_log), daemon=True)
        pump.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_session(proc)
        _kill_session(proc)  # reap leftovers (e.g. orphaned fuse daemons) in any case
        pump.join(timeout=60)
        text = raw_log.read_bytes().decode("utf-8", errors="replace")
        with open(log_path, "w") as f:
            f.write(text)
            f.write(f"\n[ace: exit_code={proc.returncode} timed_out={timed_out}]\n")
        result["exit_code"] = proc.returncode
        result["duration_s"] = round(time.monotonic() - t0, 1)

        if timed_out:
            result["status"] = "timeout"
            return result
        if f"ACE_PATCH_FAILED_{nonce}" in text:
            result["status"] = "patch_failed"
            return result
        if f"ACE_PATCH_APPLIED_{nonce}" not in text and f"ACE_PATCH_EMPTY_{nonce}" not in text:
            result["error"] = "container did not start the driver (see log)"
            return result
        # Grade only the eval-script part of the log, as swebench does.
        marker = f"ACE_PATCH_APPLIED_{nonce}" if has_patch else f"ACE_PATCH_EMPTY_{nonce}"
        eval_part = text.split(marker, 1)[1]
        eval_log = run_dir / "eval_output.log"
        eval_log.write_text(eval_part)
        pred = {"instance_id": iid, "model_patch": patch, "model_name_or_path": "ace"}
        report = get_eval_report(spec, pred, str(eval_log), include_tests_status=True)[iid]
        result["tests_status"] = report.get("tests_status", {})
        result["resolved"] = bool(report.get("resolved"))
        # The patch applied, so anything short of resolved is "unresolved" (a
        # patch that breaks the test run is the patch's fault). Unparseable
        # output is flagged for triage, as swebench does (advisory only).
        result["status"] = "resolved" if result["resolved"] else "unresolved"
        if not report.get("patch_successfully_applied"):
            result["no_test_output"] = True
            result["infra_failure"] = bool(report.get("infra_failure"))
            if report.get("infra_failure_reason"):
                result["infra_failure_reason"] = report["infra_failure_reason"]
        return result
    except Exception as e:  # noqa: BLE001 - harness errors are reported, not raised
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    finally:
        if proc is not None:
            _kill_session(proc)
        result["duration_s"] = round(time.monotonic() - t0, 1)
        _rmtree(run_dir)


def _session_pids(sid: int) -> list[int]:
    """Our processes whose session id is ``sid`` (apptainer's fuse daemons get
    reparented to init but keep the session, so killpg alone misses them)."""
    uid, pids = os.getuid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = os.stat(f"/proc/{d}")
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if st.st_uid == uid and int(fields[3]) == sid:  # fields: state ppid pgrp session
            pids.append(int(d))
    return pids


def _kill_session(proc: subprocess.Popen, grace: float = 5.0) -> None:
    """SIGTERM the whole session started for ``proc``, then SIGKILL survivors."""
    sid = proc.pid
    pids = _session_pids(sid)
    if proc.poll() is None or pids:
        for pid in pids or [sid]:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and (proc.poll() is None or _session_pids(sid)):
            time.sleep(0.2)
        for pid in _session_pids(sid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _rmtree(path: Path) -> None:
    def onerror(func, p, exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onerror=onerror)
