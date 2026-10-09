# Repo layout for v1 and v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the finished v1 capstone into `v1/`, give v2 a home in `v2/`, and keep only shared files at the repo root, with a script that proves no file changed or was lost.

**Architecture:** Three local commits on `main`, then one normal push. Commit 1 is `git mv` renames only. Commit 2 edits the shared files (router README, split `.gitignore`, `AGENTS.md`, `FLOW.md`, `DECISIONS.md`). Commit 3 adds the v2 README and the dataset survey. A stdlib Python verifier records a baseline of every file's git hash before the move and checks it after each stage.

**Tech Stack:** git, Python 3 standard library (the verifier and `unittest`). No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-09-repo-v1-v2-layout-design.md`

## Global Constraints

Copied from the spec and from `AGENTS.md`. Every task includes them.

- No change to any v1 code, path, number or result. v1 is frozen as it is, including its stale paths and PSC account strings.
- No removal of tracked data, and no change to the PPMI files except being renamed along with their folder.
- No history rewrite, force-push, branch deletion, tag, or visibility change.
- No v2 code, environment, data download or compute.
- Stage explicit paths. Never `git add -A`. Never amend, reset or skip hooks. Undo with `git revert`.
- Plain English. No em dashes and no double hyphens as punctuation in any file or commit message. Number ranges use a plain hyphen.
- No personal paths, account IDs, emails or tokens in any tracked file.
- The verifier lives outside the repo, in `$WORK`. It is not committed.
- Run every command from the repo root unless a step says otherwise.
- End each commit message with the attribution lines your session was told to use.

## Differences from the spec

The spec is the source of truth. These are the places where this plan adds to it, so the reviewer can accept or reject each one:

1. `AGENTS.md` gets one sentence beyond the spec's test-command change: `v1/` is frozen, so do not edit it unless asked. Without it, nothing in the repo tells an agent that v1 is frozen.
2. The verifier has two checks the spec does not list. It confirms the baseline commit is still an ancestor of HEAD (history was not rewritten). It also confirms that no v1 script finds its folders by climbing above `v1/`.
3. The history check compares non-merge commits only, because `git log --follow` simplifies merge commits differently from a plain log. The ancestor check covers merges.
4. The plan was rehearsed on a throwaway local clone before it was written (see "Rehearsal evidence").

## Rehearsal evidence

On 2026-10-09 the full sequence below was run on a throwaway local clone of `main` at the commit that added the spec, with four ignored local files added to exercise the move check. Nothing in the real repo was touched.

- Before the move, the verifier failed as intended, so its checks are not vacuous.
- 746 tracked files moved. Commit 1 contained 746 entries, all `R100`.
- All checks passed after commit 1 and after commit 3 (the final stage reported 13 of 13 on the last run).
- The 6 existing tests passed from `v1/tests` with no edits. `scripts/verify_setup.py` printed identical output before and after the move.
- Planted bad lines (an em dash, a home path, an email, an account-ID-shaped string, a script climbing above `v1/`) were each caught, and the files were restored afterwards.
- 18 v1 scripts locate their folders with `Path(__file__)` anchors. All 18 stay inside `v1/` after the move.
- Reverting all three commits with one `git revert` command restored a tree whose hash was identical to the baseline tree.

## Review Focus

Failure modes the spec implies but no ordinary test would exercise. Each one is pinned by a check named in the owning task.

1. **Ignored local files move with their folders.** Data a person downloaded, notebook checkpoints and generated outputs must not be left behind at the old path or lost. Pinned by the verifier check "ignored local paths moved with their folders" (Task 2 step 3, Task 5 step 1).
2. **No v1 script climbs above `v1/`.** A script that finds `data/` or `outputs/` by walking up too far would look in the repo root and find nothing. Pinned by the check "no v1 script anchors its paths above v1/" (Task 2 step 3, Task 5 step 1).
3. **Ignore rules behave the same at the new paths.** The data allowlist, the `.npz` exception for the fusion embeddings and the generated handwriting artifacts must still match. Pinned by the 12 sample paths in the verifier (Task 2 step 3, Task 3 step 6, Task 5 step 1).
4. **The new shared files are clean.** No leftover `BASELINE_SHA`, no personal paths, account-ID shapes, emails or dash punctuation. Pinned by the grep in Task 3 step 1 and the verifier scans (Task 5 step 1).
5. **Links resolve and `DECISIONS.md` only grows.** The root and v2 READMEs link to files that exist, and the old `DECISIONS.md` text is an exact prefix of the new one. Pinned by two verifier checks (Task 5 step 1).

## File Structure

Moved by Task 2 (pure renames, 746 files in all):

- Folders into `v1/`: `catboost_info`, `data`, `notebooks`, `outputs`, `scripts`, `src`, `tests`.
- Files into `v1/`: `README.md`, `CONTRIBUTING.md`, `requirements.txt`, `.gitignore`.
- Children of `docs/` into `v1/docs/`: `guidelines`, `literature`, `planning`, `roadmap`, `setup`.

Changed by Task 3:

- Create `README.md` (router), `.gitignore` (generic rules), `FLOW.md`.
- Replace `v1/.gitignore` with the v1-specific rules.
- Modify `AGENTS.md` (two sentences) and `DECISIONS.md` (two entries appended).

Created by Task 4: `v2/README.md`, `v2/docs/dataset-survey.md`.

Untouched: `CLAUDE.md`, `docs/superpowers/` and everything inside the moved folders.

## Setup used by every task

Each task starts from the repo root. Choose a scratch folder outside the repo that will still exist for all five tasks, and export it:

```bash
cd "$(git rev-parse --show-toplevel)"
export REPO="$PWD"
export WORK=/path/to/a/scratch/folder   # outside the repo; holds the verifier and baseline
mkdir -p "$WORK"
```

---

### Task 1: Verifier and baseline (no repo change)

**Files:**
- Create (outside the repo): `$WORK/verify_layout.py`
- Create (outside the repo): `$WORK/baseline.json`, `$WORK/verify_setup.before.txt`

**Interfaces:**
- Produces: `python3 -I "$WORK/verify_layout.py" record --repo REPO --out FILE`, which writes the baseline JSON. Its keys are `head`, `moved`, `kept`, `file_count`, `ignored_local`, `ignore_old` and `follow`.
- Produces: `python3 -I "$WORK/verify_layout.py" check --repo REPO --baseline FILE --stage moved|final`, which prints `PASS`/`FAIL` lines and exits 0 only if all pass. Tasks 2 and 5 call it.

- [ ] **Step 1: Confirm the starting state**

```bash
cd "$REPO"
git fetch origin
git status -sb | head -5
git log --oneline -3
```

Expected: the first line is `## main...origin/main`, there are no modified tracked files, and the newest commit is the one that added this plan. If not, stop and report.

- [ ] **Step 2: Write the verifier**

Create `$WORK/verify_layout.py` with exactly this content:

````python
#!/usr/bin/env python3
"""Prove that the v1/v2 layout change lost nothing.

Standard library only. Run it with python3 -I.

  verify_layout.py record --repo REPO --out BASELINE.json
      Before the move: check the repo is clean and on main, then save a baseline
      (the git hash of every file, the ignored local files, and ignore-rule samples).

  verify_layout.py check --repo REPO --baseline BASELINE.json --stage moved
      After commit 1 (the pure move): every file is identical at its new v1/ path.

  verify_layout.py check --repo REPO --baseline BASELINE.json --stage final
      After commit 3: the same, plus the new root layout, the edited shared files,
      README links, and the style and privacy scans on new or changed files.

Exit code 0 means every check passed. Any failure prints FAIL lines and exits 1.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

KEEP_AT_ROOT = ["AGENTS.md", "CLAUDE.md", "DECISIONS.md"]
MOVE_TOP = [
    "catboost_info", "data", "notebooks", "outputs", "scripts", "src", "tests",
    "README.md", "CONTRIBUTING.md", "requirements.txt", ".gitignore",
]
MOVE_DOCS_CHILDREN = ["guidelines", "literature", "planning", "roadmap", "setup"]
KEEP_DOCS_CHILD = "superpowers"
FINAL_ROOT = sorted([
    ".gitignore", "AGENTS.md", "CLAUDE.md", "DECISIONS.md", "FLOW.md",
    "README.md", "docs", "v1", "v2",
])
# Files whose content may differ from the baseline once the shared files are edited.
EDITED_V1 = {"v1/.gitignore"}
EDITED_ROOT = {"AGENTS.md", "DECISIONS.md"}
# Root files that are created fresh after the old ones move (the router README and the generic ignore rules).
NEW_ROOT_FILES = {"README.md", ".gitignore"}

# Paths that need not exist. Used to compare ignore rules before and after.
IGNORE_SAMPLES = [
    "data/gait/other_raw.csv",
    "data/handwriting/new_file.csv",
    "data/handwriting/processed/merged_handwriting_timeseries.csv",
    "src/multimodal_fusion/embeddings/new_embeddings.npz",
    "src/unimodal/speech/new_weights.npz",
    "src/unimodal/speech/scripts/catboost_info/learn/x",
    "outputs/unimodal_handwriting/svm_embeddings/x.csv",
    "outputs/unimodal_handwriting/model_benchmark/x.csv",
    "notebooks/eda/.ipynb_checkpoints/x",
    "notes.local.env",
    "run.log",
    "weights.pt",
]
# Paths that must be ignored in the final layout.
FINAL_MUST_IGNORE = ["notes.local.env", "v2/data/x.csv", ".DS_Store", "weights.pt"]

STYLE_PATTERNS = [("em or en dash", re.compile("[" + chr(0x2014) + chr(0x2013) + "]")),
                  ("double hyphen used as punctuation", re.compile(r" -- "))]
PRIVACY_PATTERNS = [
    ("absolute user path", re.compile(r"/Users/|/home/|/ocean/")),
    ("home-relative path", re.compile(r"(^|[\s`(])~/")),
    ("PSC-style account id", re.compile(r"\b[a-z]{3}\d{6}p\b")),
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("token shape", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
]


def git(repo, *args, check=True, text=True):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)
    if check and result.returncode != 0:
        err = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise SystemExit(f"git {' '.join(args)} failed: {err.strip()}")
    return result


def tree_map(repo, rev):
    """Map every tracked path at rev to (mode, blob hash)."""
    out = git(repo, "ls-tree", "-r", "-z", rev, text=False).stdout
    mapping = {}
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, _kind, sha = meta.decode().split()
        mapping[path.decode("utf-8", "surrogateescape")] = (mode, sha)
    return mapping


def is_moved(path):
    parts = path.split("/")
    if parts[0] in MOVE_TOP:
        return True
    return parts[0] == "docs" and len(parts) > 2 and parts[1] in MOVE_DOCS_CHILDREN


def is_ignored(repo, path):
    code = git(repo, "check-ignore", "-q", "--no-index", path, check=False).returncode
    if code not in (0, 1):
        raise SystemExit(f"git check-ignore failed for {path} (exit {code})")
    return code == 0


def anchor_depths(text):
    """Parent hops used by `Path(__file__)...` anchors: parents[N] gives N, each .parent hop gives hops-1."""
    found = [int(m.group(1)) for m in re.finditer(r"Path\(__file__\)(?:\.resolve\(\))?\.parents\[(\d+)\]", text)]
    for m in re.finditer(r"Path\(__file__\)(?:\.resolve\(\))?((?:\.parent(?!s))+)", text):
        found.append(m.group(1).count(".parent") - 1)
    return found


def ignored_local_paths(repo):
    out = git(repo, "status", "--porcelain=v1", "-z", "--ignored=matching", text=False).stdout
    paths = []
    for entry in out.split(b"\0"):
        if entry.startswith(b"!! "):
            paths.append(entry[3:].decode("utf-8", "surrogateescape"))
    return paths


def expect_layout(tm):
    """Stop on any surprise in the top-level layout instead of guessing."""
    top = {p.split("/")[0] for p in tm}
    want_top = set(MOVE_TOP) | set(KEEP_AT_ROOT) | {"docs"}
    problems = []
    if top != want_top:
        problems.append(f"top-level entries differ: extra={sorted(top - want_top)} missing={sorted(want_top - top)}")
    children = {p.split("/")[1] for p in tm if p.startswith("docs/")}
    want_children = set(MOVE_DOCS_CHILDREN) | {KEEP_DOCS_CHILD}
    if children != want_children:
        problems.append(f"docs children differ: extra={sorted(children - want_children)} missing={sorted(want_children - children)}")
    return problems


def cmd_record(args):
    repo = Path(args.repo)
    if git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise SystemExit("FAIL: tracked files have uncommitted changes")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        raise SystemExit(f"FAIL: branch is {branch}, expected main")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    origin = git(repo, "rev-parse", "origin/main").stdout.strip()
    if head != origin:
        raise SystemExit("FAIL: HEAD is not equal to origin/main")
    tm = tree_map(repo, "HEAD")
    problems = expect_layout(tm)
    if problems:
        raise SystemExit("FAIL: unexpected layout:\n  " + "\n  ".join(problems))
    moved = {p: list(v) for p, v in tm.items() if is_moved(p)}
    kept = {p: list(v) for p, v in tm.items() if not is_moved(p)}
    ignored = [p for p in ignored_local_paths(repo) if is_moved(p.rstrip("/") + "/x")]
    ignore_old = {p: is_ignored(repo, p) for p in IGNORE_SAMPLES}
    first_in = {}
    for p in sorted(moved):
        parts = p.split("/")
        key = "/".join(parts[:2]) if parts[0] == "docs" else parts[0]
        first_in.setdefault(key, p)
    follow = {p: git(repo, "log", "--format=%H", head, "--", p).stdout.split() for p in first_in.values()}
    baseline = {
        "head": head, "moved": moved, "kept": kept, "file_count": len(tm),
        "ignored_local": ignored, "ignore_old": ignore_old, "follow": follow,
    }
    Path(args.out).write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    print(f"recorded {len(moved)} files to move, {len(kept)} to keep, {len(ignored)} ignored local paths")
    print(f"baseline HEAD {head[:10]}, wrote {args.out}")


class Report:
    def __init__(self):
        self.failures = []
        self.passed = []

    def ok(self, label, cond, detail=""):
        if cond:
            self.passed.append(label)
            print(f"PASS  {label}")
        else:
            self.failures.append(label)
            print(f"FAIL  {label}" + (f": {detail}" if detail else ""))


def cmd_check(args):
    repo = Path(args.repo)
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    final = args.stage == "final"
    tm = tree_map(repo, "HEAD")
    rep = Report()

    # 1. Every moved file is identical at its new path, and gone from the old path.
    bad = []
    for p, (mode, sha) in base["moved"].items():
        n = "v1/" + p
        if tm.get(n) != (mode, sha):
            if final and n in EDITED_V1:
                continue
            bad.append(f"changed or missing: {n}")
        if p in tm and not (final and p in NEW_ROOT_FILES):
            bad.append(f"still at old path: {p}")
    rep.ok(f"all {len(base['moved'])} moved files identical at v1/ and gone from old paths", not bad, "; ".join(bad[:5]))

    # 2. Files that stay at the root.
    bad = []
    for p, (mode, sha) in base["kept"].items():
        if final and p in EDITED_ROOT:
            if p not in tm:
                bad.append(f"missing: {p}")
            continue
        if tm.get(p) != (mode, sha):
            bad.append(f"changed or missing: {p}")
    rep.ok(f"all {len(base['kept'])} kept root files {'present (edited ones may differ)' if final else 'unchanged'}", not bad, "; ".join(bad[:5]))

    # 3. No stray top-level entries.
    top = sorted({p.split("/")[0] for p in tm})
    if final:
        rep.ok("final root layout is exactly the expected entries", top == FINAL_ROOT, f"got {top}")
    else:
        want = sorted(set(KEEP_AT_ROOT) | {"docs", "v1"})
        rep.ok("root holds only kept files, docs and v1/", top == want, f"got {top}")

    if not final:
        # 4. Commit 1 is one pure-rename commit on top of the baseline.
        parent = git(repo, "rev-parse", "HEAD~1").stdout.strip()
        rep.ok("HEAD is one commit above the baseline", parent == base["head"], f"parent {parent[:10]}")
        lines = git(repo, "diff", "--name-status", "-M", "HEAD~1", "HEAD").stdout.splitlines()
        non_r100 = [ln for ln in lines if not ln.startswith("R100")]
        rep.ok(f"commit 1 has only 100% renames ({len(lines)} entries)", not non_r100 and len(lines) == len(base["moved"]),
               f"{len(non_r100)} non-R100 entries; {len(lines)} entries vs {len(base['moved'])} moved files; first: {non_r100[:3]}")
        rep.ok("tracked file count unchanged", len(tm) == base["file_count"], f"{len(tm)} vs {base['file_count']}")
    else:
        # DECISIONS.md is append-only.
        old = git(repo, "show", f"{base['head']}:DECISIONS.md", text=False).stdout
        new = git(repo, "show", "HEAD:DECISIONS.md", text=False).stdout
        rep.ok("DECISIONS.md only grew (old text is an exact prefix)", new.startswith(old) and len(new) > len(old))

    # 5. Ignored local files moved with their folders.
    bad = []
    for p in base["ignored_local"]:
        old_path, new_path = repo / p, repo / "v1" / p
        if old_path.exists():
            bad.append(f"left behind: {p}")
        if not new_path.exists():
            bad.append(f"not found under v1/: {p}")
    rep.ok(f"{len(base['ignored_local'])} ignored local paths moved with their folders", not bad, "; ".join(bad[:5]))

    # 6. Ignore rules behave the same at the new paths.
    bad = []
    for p, was in base["ignore_old"].items():
        now = is_ignored(repo, "v1/" + p)
        if now != was:
            bad.append(f"{p}: was {'ignored' if was else 'tracked'}, now {'ignored' if now else 'tracked'}")
    rep.ok(f"ignore rules unchanged for {len(base['ignore_old'])} sample paths", not bad, "; ".join(bad[:5]))

    # 7. History is intact. The baseline commit must still be an ancestor of HEAD (nothing rewritten),
    #    and git log --follow must reach every original non-merge commit of each sample file.
    #    Merge commits are skipped because --follow simplifies merges differently from a plain log.
    ancestor = git(repo, "merge-base", "--is-ancestor", base["head"], "HEAD", check=False).returncode == 0
    rep.ok("baseline commit is still an ancestor of HEAD (history not rewritten)", ancestor)
    bad = []
    for old in base["follow"]:
        want = set(git(repo, "log", "--no-merges", "--format=%H", base["head"], "--", old).stdout.split())
        got = set(git(repo, "log", "--follow", "--no-merges", "--format=%H", "--", "v1/" + old).stdout.split())
        if not want <= got:
            bad.append(f"{old} ({len(want - got)} missing)")
    rep.ok(f"git log --follow reaches the original history for {len(base['follow'])} sample files", not bad, "; ".join(bad[:5]))

    # 7b. No v1 script finds its folders by climbing above v1/ (it would look in the repo root and find nothing).
    bad, scanned = [], 0
    for p in sorted(tm):
        if p.startswith("v1/") and p.endswith(".py"):
            text = (repo / p).read_text(encoding="utf-8", errors="replace")
            depths = anchor_depths(text)
            scanned += bool(depths)
            inside = p.count("/") - 1  # folders between v1/ and the file
            if any(n > inside for n in depths):
                bad.append(f"{p}: climbs {max(depths)} levels from a file {inside} folders below v1/")
    rep.ok(f"no v1 script anchors its paths above v1/ ({scanned} scripts use Path(__file__) anchors)", not bad, "; ".join(bad[:5]))

    # 8. Working tree is clean.
    flag = [] if final else ["--untracked-files=no"]
    dirty = git(repo, "status", "--porcelain", *flag).stdout.strip()
    rep.ok("git status is clean" + ("" if final else " (tracked files)"), not dirty, dirty[:200])

    if final:
        bad = [p for p in FINAL_MUST_IGNORE if not is_ignored(repo, p)]
        rep.ok("final ignore rules cover local-account files, v2/data and weights", not bad, f"not ignored: {bad}")
        # README links resolve.
        bad = []
        for rel in ("README.md", "v2/README.md"):
            text = (repo / rel).read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
                if "://" in target or target.startswith("mailto:"):
                    continue
                if not (repo / rel).parent.joinpath(target).exists():
                    bad.append(f"{rel} -> {target}")
        rep.ok("relative links in the root and v2 READMEs resolve", not bad, "; ".join(bad[:5]))
        # Style and privacy scans on files added or edited since the baseline (outside the moved v1 files).
        names = git(repo, "diff", "--name-only", "-z", "--diff-filter=AM", base["head"], "HEAD", text=False).stdout.split(b"\0")
        files = [n.decode() for n in names if n and not (n.decode().startswith("v1/") and n.decode() not in EDITED_V1)]
        bad = []
        for rel in files:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
            for label, pattern in STYLE_PATTERNS + PRIVACY_PATTERNS:
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        bad.append(f"{rel}:{lineno} {label}")
        rep.ok(f"style and privacy scans clean on {len(files)} new or edited files", not bad, "; ".join(bad[:6]))

    print()
    if rep.failures:
        print(f"{len(rep.failures)} check(s) FAILED, {len(rep.passed)} passed")
        sys.exit(1)
    print(f"all {len(rep.passed)} checks passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--repo", required=True)
    rec.add_argument("--out", required=True)
    chk = sub.add_parser("check")
    chk.add_argument("--repo", required=True)
    chk.add_argument("--baseline", required=True)
    chk.add_argument("--stage", choices=["moved", "final"], required=True)
    args = ap.parse_args()
    {"record": cmd_record, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
````

- [ ] **Step 3: Capture v1's own checks before the move**

```bash
python3 -m unittest discover -s tests 2>&1 | tail -3
python3 scripts/verify_setup.py > "$WORK/verify_setup.before.txt" 2>&1; echo "verify_setup exit: $?"
```

Expected: `Ran 6 tests` and `OK`, then `verify_setup exit: 0`.

- [ ] **Step 4: Record the baseline**

```bash
python3 -I "$WORK/verify_layout.py" record --repo "$REPO" --out "$WORK/baseline.json"
```

Expected: `recorded 746 files to move, 5 to keep, N ignored local paths` followed by `baseline HEAD <hash>`. The five kept files are `AGENTS.md`, `CLAUDE.md`, `DECISIONS.md`, the spec and this plan. If the counts differ, stop and find out why before moving anything. `N` depends on what ignored local files exist on this machine.

- [ ] **Step 5: Confirm the check fails before the move**

```bash
python3 -I "$WORK/verify_layout.py" check --repo "$REPO" --baseline "$WORK/baseline.json" --stage moved; echo "exit: $?"
```

Expected: several `FAIL` lines (for example "all 746 moved files identical at v1/") and `exit: 1`. A check that cannot fail would prove nothing.

Nothing to commit in this task.

---

### Task 2: Commit 1, the pure move

**Files:**
- Move: every path listed under "File Structure" above, with `git mv` only.

**Interfaces:**
- Consumes: `$WORK/baseline.json` and `$WORK/verify_layout.py` from Task 1.
- Produces: one commit whose subject is `Move the v1 capstone into v1/`, with 746 renames at 100% similarity.

- [ ] **Step 1: Move the files**

```bash
cd "$REPO"
(
  set -e
  mkdir -p v1/docs
  for p in catboost_info data notebooks outputs scripts src tests README.md CONTRIBUTING.md requirements.txt .gitignore; do
    git mv "$p" "v1/$p"
  done
  for c in guidelines literature planning roadmap setup; do
    git mv "docs/$c" "v1/docs/$c"
  done
)
git status --short | awk '{print $1}' | sort | uniq -c
```

Expected: exactly `746 R`. Any other letter means something other than a rename happened: stop. Do not run `git add -A`: renames are already staged, and stray untracked files (for example `.DS_Store`) must not be added.

- [ ] **Step 2: Commit**

```bash
git commit -q -F - <<'EOF'
Move the v1 capstone into v1/

v1 is the finished capstone and stays frozen. This commit only renames:
every tracked file at the old root moves to the same path under v1/, with
no content edits, so git records 100% renames and keeps each file's history.
The shared files (AGENTS.md, CLAUDE.md, DECISIONS.md and docs/superpowers/)
stay at the root.

Design: docs/superpowers/specs/2026-10-09-repo-v1-v2-layout-design.md
EOF
git log -1 --format='%h %s'
```

Add the attribution lines your session was told to use at the end of the message. Expected: one new commit with the subject above.

- [ ] **Step 3: Run the verifier for the moved stage**

```bash
python3 -I "$WORK/verify_layout.py" check --repo "$REPO" --baseline "$WORK/baseline.json" --stage moved; echo "exit: $?"
```

Expected: every line starts with `PASS`, the last line says `all 12 checks passed`, and `exit: 0`. If any line says `FAIL`, stop. Do not change the verifier to make it pass. Report the `FAIL` lines. To undo, run `git revert --no-edit HEAD`.

- [ ] **Step 4: Run v1's own checks from the new location**

```bash
python3 -m unittest discover -s v1/tests -v 2>&1 | tail -4
( cd v1 && python3 scripts/verify_setup.py > "$WORK/verify_setup.after.txt" 2>&1; echo "verify_setup exit: $?" )
diff <(sed "s#$REPO/v1#ROOT#g; s#$REPO#ROOT#g" "$WORK/verify_setup.after.txt") <(sed "s#$REPO#ROOT#g" "$WORK/verify_setup.before.txt") && echo "verify_setup output identical before and after"
```

Expected: `Ran 6 tests` and `OK`, then `verify_setup exit: 0`, then `verify_setup output identical before and after`.

Do not push yet.

---

### Task 3: Commit 2, shared files

**Files:**
- Create: `README.md`, `.gitignore`, `FLOW.md`
- Modify: `v1/.gitignore` (replace), `AGENTS.md`, `DECISIONS.md` (append)

**Interfaces:**
- Consumes: the Task 2 commit, and `$WORK/baseline.json` (for the baseline hash).
- Produces: the root `README.md` that links to `v1/README.md` and `v2/README.md`. Task 4 creates the `v2/README.md` link target, and Task 5 checks the links.

- [ ] **Step 1: Create the router README**

Create `README.md` at the repo root with exactly this content:

````markdown
# Multimodal Parkinson's Disease Classification and Explainability

Two versions of one project live in this repo.

| | What it is | Where |
|---|---|---|
| **v1** | The finished CMU PBAI capstone: gait, handwriting and speech models, late fusion, and SHAP explainability. Frozen as it was. | [`v1/`](v1/README.md) |
| **v2** | A redo on datasets where the same people have more than one modality, so fusion can be tested on real people and not on randomly paired ones. In planning. | [`v2/`](v2/README.md) |

## Why v2

v1 used separate public datasets for gait, handwriting and speech, with different people in each. Its fusion scores therefore come from randomly pairing people across cohorts. That is a simulation, not a multimodal test. v2 starts from datasets with the same participants in every modality, and tests fusion on them.

## Shared files

- [`AGENTS.md`](AGENTS.md): the working rules for people and AI agents in this repo.
- [`DECISIONS.md`](DECISIONS.md): a dated record of structural decisions.
- [`FLOW.md`](FLOW.md): entry points and execution order for each version.
- `docs/superpowers/`: design specs and implementation plans.

## About v1's paths

v1 is kept exactly as it was, including the paths and cluster settings inside its scripts, which belong to the original course environment. Run v1 commands from inside `v1/`. The last commit before this layout is `BASELINE_SHA`: use `git show BASELINE_SHA:README.md` to see the original README, or check that commit out to see the old layout.
````

The text `BASELINE_SHA` stands for the last commit before the layout change. Replace it with the 7-character hash saved in the baseline:

```bash
SHA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["head"][:7])' "$WORK/baseline.json")
sed -i '' "s/BASELINE_SHA/$SHA/g" README.md   # macOS sed. With GNU sed, drop the empty quotes after -i.
grep -c BASELINE_SHA README.md                # Expected: 0
grep -n "$SHA" README.md | head -3            # Expected: one line (the "About v1's paths" paragraph), showing the hash twice
```

- [ ] **Step 2: Split the ignore rules**

Create the root `.gitignore` with exactly this content:

````text
.DS_Store

# Python cache/build
__pycache__/
*.py[cod]
*.so
*.egg-info/
build/
dist/

# Local environments (account and allocation values go in *.local.* files)
.env
*.local.*
.venv/
venv/
.conda/

# Editor metadata
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# Runtime logs
*.log
*.out
*.err

# Training artifacts
*.pt
*.pth
*.pkl
*.joblib
*.npy
*.npz

# Large compressed packages
*.zip
*.tar
*.tar.gz
*.7z

# v2 data lives outside the repo (see PD_DATA_ROOT). Safety net only.
v2/data/
````

Replace the whole content of `v1/.gitignore` (the old file that moved in Task 2) with exactly this:

````text
# v1-specific rules. Generic rules (caches, envs, weights, archives) are in the root .gitignore.
# Paths here are relative to this folder.

# Keep the committed fusion embeddings (the root ignores *.npz)
!src/multimodal_fusion/embeddings/*.npz

# Speech model run folders
src/unimodal/speech/scripts/catboost_info/

# Handwriting generated artifacts
outputs/unimodal_handwriting/svm_embeddings/
outputs/unimodal_handwriting/model_benchmark/
data/handwriting/processed/merged_handwriting_timeseries.csv

# Only track handwriting data, ignore all other data
data/*
!data/handwriting/
!data/handwriting/**
````

- [ ] **Step 3: Edit AGENTS.md**

Make two replacements in `AGENTS.md`. Each old string occurs exactly once.

Replace

```
It began as a team capstone for a CMU course (PBAI) and is public. `README.md` describes the current layout. `DECISIONS.md` records why things are the way they are.
```

with

```
It began as a team capstone for a CMU course (PBAI) and is public. The repo holds two versions. `v1/` is the finished capstone and is frozen: do not edit anything under `v1/` unless asked. `v2/` is the redo on datasets where the same people have several modalities. `README.md` describes the layout, `DECISIONS.md` records why things are the way they are, and `FLOW.md` lists the entry points.
```

Replace

```
Run the tests from the repo root: `python3 -m unittest discover -s tests -v`. Tests use the standard library `unittest`.
```

with

```
Run the v1 tests from the repo root: `python3 -m unittest discover -s v1/tests -v`. Tests use the standard library `unittest`. v2 tests are added with v2 code.
```

- [ ] **Step 4: Create FLOW.md**

Create `FLOW.md` at the repo root with exactly this content:

````markdown
# Flow

How the code runs, one section per version. Entry points and execution order only. Git history records what changed. Update this file only when the flow itself changes.

Every item below was read from the code on 2026-10-09. None of the commands were run to write this file. Where the repo does not record something, this file says so.

## v1 (in `v1/`, frozen)

Run v1 commands from inside `v1/`. Each script finds `data/` and `outputs/` relative to its own location (`Path(__file__)` plus a fixed number of parent folders), so the folder can sit anywhere. The cluster launchers (`*.slurm`) also contain absolute paths from the original course environment and will not run elsewhere unchanged.

### Late fusion and explainability

1. `python -m src.multimodal_fusion.evaluate [--strategy equal|auc_weighted|softmax_auc_weighted|confidence_weighted] [--no-save]`. The default strategy is `auc_weighted`.
2. `evaluate.py` calls `loaders.load_all_from_embeddings()`, which reads from `src/multimodal_fusion/embeddings/`. Which file it reads: handwriting is pinned to `handwriting_embeddings_0406.csv`; gait is the last file by name among `gait_embeddings_*.npz` and `*.csv`; speech is the last match for `speech_embeddings_*.npz`. Adding a file with a later name changes the gait or speech input.
3. It fits `LateFusionModel(strategy, calibrate=True)` from `fusion.py`, computes a metrics table with bootstrap confidence intervals, and writes `outputs/multimodal_fusion/fusion_model.json` and `metrics_table.csv`.
4. The three embedding sets come from different people. The fusion scores are estimated by randomly pairing subjects across modalities and labeling each pair by majority vote (`fusion.py`: the bootstrap evaluation of `LateFusionModel` and `StackingFusionModel`). They are simulated results, not measurements on people who have every modality.
5. `explainability.py` defines `FusionExplainer` and `GaitEmbeddingExplainer` (SHAP). No script, notebook or launcher in the repo calls them. How the SHAP outputs were produced is not recorded.
6. The repo does not record which script wrote each committed file in `src/multimodal_fusion/embeddings/`.

### Gait: figshare IMU data (severity among PD patients)

- `python src/unimodal/gait/train_gait.py` takes no command-line flags. A temporal convolutional network (TCN) separates Mild PD (Hoehn and Yahr 2.0 or lower) from Moderate/Severe PD (above 2.0), using freezing-of-gait patients only. This is a severity task, not PD against healthy controls. It reads `data/gait/figshare/IMU/` and `PDFEinfo.csv`, uses `StratifiedGroupKFold` with 5 folds and seed 42, and writes `outputs/unimodal_gait/PDFE_Severity_Classification/` (`cv_results.csv`, `predictions.npz`, `scaler.pkl`, `summary.json`, one folder per fold).
- `python src/unimodal/gait/train_gait_rf.py` is a random forest baseline on engineered features for the same severity task.

### Gait: WearGait-PD

1. `python src/unimodal/gait/prepare_weargait_index.py [--data-root PATH] [--out-csv PATH]` builds an index of the WearGait files, with group 1 for paths under `PD PARTICIPANTS` and 0 otherwise. Default output: `outputs/unimodal_gait/weargait_index.csv`.
2. `python src/unimodal/gait/train_weargait_embeddings.py --tasks <task>` trains a model for one walking task and writes subject embeddings and predictions under `outputs/unimodal_gait/weargait_dl_embeddings/<task>/` (`weargait_subject_embeddings.npz`, `predictions.npz`, `cv_metrics.csv`, `summary.json`, one model file per fold). The tasks are SelfPace, HurriedPace and TUG.
3. `python src/unimodal/gait/gait_ensemble_orchestrator.py --tasks <pdfe,weargait,rf or all> [--force]` runs, in order: `train_gait.py`, step 2 for each walking task, `concat_weargait_task_embeddings.py`, then `train_gait_rf.py`. It writes `outputs/unimodal_gait/ensemble_summary.json`. **`concat_weargait_task_embeddings.py` is not tracked on `main`.** It was deleted on 2026-03-26 in `8207708` and can be restored from `be97d0b`. Without it, the WearGait branch of the orchestrator fails at that step.
4. `python src/unimodal/gait/ensemble_fusion.py --strategy all` combines the saved prediction files from the three gait tasks (weighted average, stacking, calibrated late fusion, voting) where their labels align.
5. Experiments: `src/unimodal/gait/experiments/weargait_representation`, `weargait_multimodal_compare` and `weargait_update3_ablation`. Each has a `run_experiment.py` and a Slurm launcher, and works under `outputs/unimodal_gait/`. `benchmark_weargait_representations.py` also reads there. These were not examined in detail.
6. Cluster launchers: `src/unimodal/gait/slurm/` runs the orchestrator (`--tasks all --force`, or `--tasks rf --force` on CPU) and then `ensemble_fusion.py --strategy all`. `train_gait.slurm` and `train_weargait_embeddings.slurm` (which runs steps 1 and 2) sit beside the scripts.

### Handwriting

Scripts in `src/unimodal/handwriting/`, each with a Slurm launcher in `slurm/`:

- `train_handwriting_svm_embeddings.py` writes out-of-fold predictions, cross-validation metrics and a feature table. The README describes it as an SVM that exports drawing-level embeddings.
- `benchmark_handwriting_models.py [--n-splits N] [--embedding-dim N] [--seed N]` cross-validates several classifiers. It writes per-model metrics and out-of-fold embeddings, a leaderboard, and `handwriting_summary_features_labeled.csv`.
- `finalize_handwriting_model.py [--outer-splits N] [--inner-splits N] [--primary-metric M] [--target-recall X] [--seed N]` runs nested cross-validation, ranks the models, and writes `all_models_oof_predictions.csv` and `final_model_input_table.csv`.
- The `in_air_*` pipelines and `eda_non_linear_relationships.py` were not examined.

Default output folders are computed from each script's location. The input paths, and which script wrote each committed `handwriting_embeddings_*.csv`, are not recorded.

### Speech

`src/unimodal/speech/scripts/TrainSpeechBasedModel_v2.0.ipynb` trains a CNN on mel-spectrograms and a CatBoost model, and calls `np.savez`. It reads precomputed mel-spectrogram files from absolute paths under a home directory on another machine, and nothing in it computes mel-spectrograms. The code that made those files is not in the repo, so speech cannot be rerun from here. The notebook never uses the name `speech_embeddings`, so which code wrote the committed `speech_embeddings_0325.npz` is not recorded. Utility scripts in the same folder: `sort_by_diagnosis.py`, `sort_by_task.py`, `sort_static.py`, `findStaticOutliers.py`, `tryTrainForStaticFeatures.py`. An older notebook is at `notebooks/modelTraining/TrainSpeechBasedModel.ipynb`.

### Other

- `notebooks/eda/` holds exploratory notebooks for gait, handwriting and WearGait-PD, and speech quality-control outputs.
- `scripts/verify_setup.py` checks that a set of folders and files exists. Its list describes an earlier layout of the project.
- `tests/test_download_weargait.py` (standard library `unittest`) checks the WearGait download script and manifests. Run it from the repo root with `python3 -m unittest discover -s v1/tests -v`.

## v2 (in `v2/`)

No code yet. See `v2/README.md` for the plan. Add its entry points and execution order here when they exist.
````

- [ ] **Step 5: Append two entries to DECISIONS.md**

Append this to the end of `DECISIONS.md`. The first line of the block is empty, which leaves one blank line between the old last entry and the new first one. Do not edit any existing line.

````markdown

## 2026-10-08: v2 is a redo on paired-modality data, in this repo

**Decision:** Build a second version of the project ("v2") in this repo. v2 uses whichever two or three modalities have a real dataset in which the same participants appear in every modality. The plan is approach A: mPower (voice, finger tapping and walking from the same phone users) as the main paired fusion test, and PADS (smartwatch motion on 11 tasks plus a symptom questionnaire) as a parallel, fully open movement track. The mPower target is self-reported PD and is called that everywhere. Nothing from v1 may be destroyed.
**Options considered:** Keep the three v1 modalities (gait, handwriting, speech) from separate datasets and keep scoring fusion on randomly paired people. Use only fully open data (PADS and WearGait-PD). Put v2 in a new repo.
**Why:** v1's fusion scores come from randomly pairing people across separate cohorts, so they are a simulation and not a multimodal test. No open dataset was found with the same people in gait plus speech or handwriting, so the data decides the modalities. mPower is the largest set found with three modalities from the same people (2,729 people with all three, in the published analysis). A new repo was considered and not chosen. The dataset survey is in `v2/docs/dataset-survey.md`.

## 2026-10-09: v1 moves into v1/ and v2 gets v2/ (layout 2)

**Decision:** v1 moves unchanged into `v1/` with `git mv`, so every file keeps its history. v2 lives in `v2/`. The repo root holds only shared files: the router `README.md`, `AGENTS.md`, `CLAUDE.md`, `DECISIONS.md`, `FLOW.md`, `.gitignore` and `docs/superpowers/`. Nothing is deleted, history is not rewritten, and nothing is force-pushed. Removing v1's tracked data (about 1.7 GB) from the working tree is not part of this step. It happens only after the data is copied to an archive outside the repo with a SHA-256 manifest, and it needs its own approval. The 2026-10-08 decision "Public showcase repo with no raw data" stays in force, and this entry stages how it is carried out.
**Options considered:** Add `v2/` and leave v1 where it is. Tag the current state, let v2 take over the root, and archive v1 under `archive/v1/`.
**Why:** Moving v1 as one block keeps it complete and browsable and gives the root a short list. Git records the move as renames, and a script compares the hash of every file before and after to prove nothing changed. Untracking data deletes files from the working tree of anyone who pulls, so it waits for a safe copy. Design: `docs/superpowers/specs/2026-10-09-repo-v1-v2-layout-design.md`.
````

- [ ] **Step 6: Check the ignore rules and the files, then commit**

```bash
for p in v1/data/gait/other_raw.csv v1/src/unimodal/speech/new_weights.npz notes.local.env v2/data/x.csv \
         v1/src/multimodal_fusion/embeddings/new_embeddings.npz v1/data/handwriting/new_file.csv; do
  git check-ignore -q --no-index "$p"; echo "exit $? : $p"
done
git status --short
```

Expected: the first four lines say `exit 0` (ignored: a non-handwriting data file, a weights file, a local-account file and a path under `v2/data/`). The last two say `exit 1` (not ignored, on purpose: the embeddings exception and the handwriting allowlist keep those paths tracked). Check one path at a time: `git check-ignore -q` accepts only one path, and with `-v` git also prints re-include rules and exits 0, which looks like a failure but is not. `git status` shows `M AGENTS.md`, `M DECISIONS.md`, `M v1/.gitignore`, and untracked `.gitignore`, `FLOW.md` and `README.md`.

```bash
git add README.md .gitignore v1/.gitignore AGENTS.md FLOW.md DECISIONS.md
git diff --cached --stat
git commit -q -F - <<'EOF'
Add the root README, split .gitignore, and update the shared files

The root README now routes to v1 and v2. The ignore rules are split: the
root keeps the generic ones (and ignores v2/data/ as a safety net), and
v1/.gitignore keeps the rules that name v1 paths. AGENTS.md gets the new
test command and says v1 is frozen. FLOW.md is new and maps v1's entry
points from the code, marking what the repo does not record. DECISIONS.md
gets two new entries; no existing line is edited.
EOF
git log -1 --format='%h %s'
```

Add the attribution lines at the end of the message. Expected: `git diff --cached --stat` lists six files, and one new commit is created.

---

### Task 4: Commit 3, start v2

**Files:**
- Create: `v2/README.md`, `v2/docs/dataset-survey.md`

**Interfaces:**
- Consumes: the root `README.md` from Task 3, which links to `v2/README.md`.
- Produces: `v2/README.md`, which links to `docs/dataset-survey.md` and `../AGENTS.md`. Task 5 checks both links.

- [ ] **Step 1: Create the v2 README**

Create `v2/README.md` with exactly this content:

````markdown
# v2: Parkinson's classification on paired-modality data

Status: planning. There is no code yet.

## Goal

v1 trained models on separate cohorts, so its fusion scores come from randomly pairing people across them. v2 uses datasets where the same people have more than one modality, and tests fusion on those people.

## Plan

- **mPower** (Sage Bionetworks, on Synapse): voice, finger tapping and walking from the same phone users. The published analysis found 2,729 people with all three, of whom 645 reported a PD diagnosis. The labels are self-reported, so the target is called "self-reported PD vs not" everywhere. Access needs a Synapse account, a certification quiz, an identity document and an intended-use statement. The data cannot be redistributed.
- **PADS** (PhysioNet): smartwatch motion on 11 tasks plus a symptom questionnaire for 469 people, openly downloadable under CC BY-NC-SA 4.0. It has no speech, so it serves as a movement-only track.

Why these two, and what else was considered: [`docs/dataset-survey.md`](docs/dataset-survey.md).

## Ground rules

The rules in [`../AGENTS.md`](../AGENTS.md) apply. In particular: no raw or participant-level data in git, the data root comes from `PD_DATA_ROOT`, splits are by person, and fusion is tested only on people who have every fused modality.

## Next

Separate specs and plans, in this order: data access and manifests; shared infrastructure; per-modality models; fusion and explainability.
````

- [ ] **Step 2: Create the dataset survey**

Create `v2/docs/dataset-survey.md` with exactly this content:

````markdown
# Dataset survey for v2

Date: 2026-10-08. Purpose: find datasets where the same participants have more than one modality and a Parkinson's disease (PD) label, so fusion can be tested on real people.

## How this was done

Three research agents searched public web pages and papers in parallel. They did not download data, create accounts or accept any data terms. The main session then read the key pages itself. In this document, **verified** means the main session read the claim at the source, with the URL given. **Agent-reported** means a research agent reported it from the source it cites, and it has not been checked again. Nothing here comes from memory.

## Core findings

- No open dataset was found where the same people have gait or IMU data plus speech or handwriting, with healthy controls. (agent-reported)
- None was found with speech plus handwriting from the same patients. A 2025 Scientific Reports paper (PMC12371065) says so directly. (agent-reported; the paper was not opened)
- Handwriting is therefore unlikely to be a v2 modality. The open options are voice, finger tapping and walking from one phone app (mPower), and wrist motion on 11 tasks plus a questionnaire (PADS).

## The two chosen datasets

### mPower: the main paired fusion test

- **Source:** Synapse project `syn4993293` (https://www.synapse.org/Synapse:syn4993293). Paper: Bot et al., Scientific Data 3:160011 (2016), doi 10.1038/sdata.2016.11.
- **Activities in the study** (verified on the Synapse wiki): voice, tapping, walking and a memory game, plus demographic, MDS-UPDRS and PDQ-8 surveys. Durations and sensor details are agent-reported.
- **People with all three of voice, tapping and walking** (verified, Deng et al., Communications Biology 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC8763910): "We identified a total of 2729 individuals (645 PwP) in the mPower dataset who had all three types of data." The non-PD count of 2,084 is the difference of those two numbers. The paper does not state it.
- **Label:** a self-reported professional diagnosis, never verified (verified, same paper). The target is called "self-reported PD vs not" in all v2 work.
- **Published result to compare with** (verified, same paper): a combined AUC of 0.944, with splits by person and no outside cohort. The Results section describes an average over four models, and the abstract says three.
- **Known risk** (agent-reported, from a second source): the share of PD rises steeply with age in this sample (reported as 56% at ages 50-65 and 0.95% at ages 35 and under), so age alone can leak the label. Matching or stratifying by age and sex is required.
- **Coverage** (verified, Synapse wiki): data from the first six months of the study, and only people who consented to broad sharing.
- **Access** (verified on the Synapse wiki pages "1 - Accessing the mPower data" and "5 - FAQs"):
  1. A Synapse account, with the Synapse governance policies and the Awareness and Ethics Pledge.
  2. A certification quiz of 15 questions.
  3. A validated profile: a complete profile, a public ORCID, and the signed Synapse Pledge.
  4. Identity attestation, by any one of these: a letter on letterhead from a signing official (not the applicant, and dated within the past calendar month), a notarized letter, or a professional license. Work or student ID badges and diplomas are not accepted.
  5. An intended-data-use statement, which is posted publicly, and agreement to the Conditions for Use.
  No turnaround time is stated.
- **Conditions for Use** (verified): no re-identification; keep the data confidential and secure; use it only as described in the intended-use statement; report any misuse within 5 business days; publish findings in open-access venues; acknowledge the mPower participants and study in every publication. **The data may not be redistributed.** The pages do not mention AI tools. v2 treats the data as code-only anyway (see "Controlled data and AI tools").
- The MDS-UPDRS and PDQ-8 surveys need separate approval from their copyright holders (verified). v2 does not need them.

### PADS: a parallel movement-only track

- **Source:** PhysioNet (https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/). Paper: Varghese et al., npj Parkinson's Disease 10:9 (2024), doi 10.1038/s41531-023-00625-7.
- **Verified on the PhysioNet page:** license CC BY-NC-SA 4.0; "Anyone can access the files, as long as they conform to the terms of the specified license"; 469 individuals and 5,159 measurement steps; smartwatch acceleration and rotation at 100 Hz; 11 movement tasks of 10 to 20 seconds; a questionnaire (age, height, weight, gender, kinship with PD, effect of alcohol on tremor) plus 30 yes/no non-motor symptom questions; 1.4 GB uncompressed; no speech or audio.
- **Agent-reported (from the paper):** 276 PD, 79 healthy controls and 114 people with other diagnoses, diagnosed by neurologists; balanced accuracy of 91.2% for PD against controls with nested 5-fold cross-validation and one sample per person (78.99% from movement alone, 89.79% from the questionnaire alone); the control group is mostly female and the PD group mostly male.
- **What it means:** the second "modality" is a questionnaire, not a second sensor, and the questionnaire alone nearly matches the combination. PADS suits a movement-only track, not a speech fusion test.

## Other candidates (all agent-reported, not checked)

| Dataset | What the same people have | Note |
|---|---|---|
| WearGait-PD (used in v1) | Xsens IMUs, insoles and a pressure walkway for 185 people (100 PD, 85 control) | All gait, with no speech or handwriting. CC BY 4.0, Synapse registration. An open outside gait cohort exists (MyGait, Zenodo 15672744), with different sensors. |
| PPMI | Wearable data on 353 people (148 PD, 35 control, 158 prodromal), plus imaging | Application review. Only 35 controls have watch data. Terms below. |
| WATCH-PD | Gait, tremor, tapping and speech on 82 early PD and 50 controls | Access through a consortium or a steering-committee proposal. Not planned around. |
| Rochester webcam study | Finger tapping, a smile and a spoken sentence, 845 people | The authors promise extracted features only. No download link was found. |
| NTUH smartphone study | Voice, tapping and walking, 496 people (213 PD, 283 control) | The repository linked from the paper holds code and models, and no participant data. |
| i-PROGNOSIS phone data | Phone motion plus typing | A small labeled set of 22 people, open on Zenodo. How many people appear in both data types is unverified. |

## Outside validation

Open second cohorts were found for voice only (MDVR-KCL and NeuroVoz; agent-reported). None was found for tapping or walking. So v2 results for those two modalities come from one cohort with splits by person, and say so.

## Controlled data and AI tools

PPMI's Data Use Agreement (version 5.0, April 2026; verified at https://www.ppmi-info.org/sites/default/files/docs/ppmi-data-use-agreement.pdf) says PPMI prohibits sharing participant-level data even if it is processed or altered, and that such data may only be distributed through a PPMI-approved repository. It also bars AI tools that do not guarantee containment of the data. v2 therefore treats any controlled-access data as code-only for agents: they may see code, schemas, counts and aggregate outputs, and never participant rows. The same is the default for mPower until its owners say otherwise.

## What was not checked

The mPower access turnaround and total download size; mPower per-task counts beyond the published 2,729; the WATCH-PD, Rochester and NTUH access terms; and every other agent-reported number above.
````

- [ ] **Step 3: Commit**

```bash
git add v2/README.md v2/docs/dataset-survey.md
git diff --cached --stat
git commit -q -F - <<'EOF'
Start v2 with a README and the dataset survey

v2 is the redo on datasets where the same people have several modalities.
The README states the plan (mPower as the main paired fusion test, PADS as
a parallel movement-only track). The survey records what was verified at
the source and what was only reported by research agents. No code, data or
participant-level information is included.
EOF
git log -1 --format='%h %s'
```

Add the attribution lines at the end of the message. Expected: two files staged, one new commit.

---

### Task 5: Final verification and push

**Files:** none changed.

**Interfaces:**
- Consumes: the three local commits and `$WORK/baseline.json`.
- Produces: `main` on the remote advanced by exactly those three commits.

- [ ] **Step 1: Run the final verifier**

```bash
python3 -I "$WORK/verify_layout.py" check --repo "$REPO" --baseline "$WORK/baseline.json" --stage final; echo "exit: $?"
```

Expected: every line starts with `PASS`, the last line says `all 13 checks passed`, and `exit: 0`. On any `FAIL`, stop and report. Do not edit the verifier. Nothing has left the machine yet. To undo the three commits, see "Rollback" at the end.

- [ ] **Step 2: Run the tests and look at the whole change**

```bash
python3 -m unittest discover -s v1/tests -v 2>&1 | tail -4
git status -sb | head -3
git log --oneline -4
git diff --name-status -M origin/main HEAD | cut -f1 | sed 's/[0-9]*$//' | sort | uniq -c
```

Expected: `Ran 6 tests` and `OK`; `## main...origin/main [ahead 3]`; the three new commits on top of the plan commit; and the summary `744 R`, `5 A`, `4 M`, with no `D`. The 744 renames are the moved files other than `README.md` and `.gitignore`. Those two still exist at the root with new content, so they show as `M` (`README.md`, `.gitignore`, plus `AGENTS.md` and `DECISIONS.md` make 4), while `v1/README.md` and `v1/.gitignore` show as `A`, together with `FLOW.md`, `v2/README.md` and `v2/docs/dataset-survey.md` (5 `A`). A `D` means a file was lost: stop.

- [ ] **Step 3: Push**

```bash
git push origin main
```

This is a normal push. If it is rejected because the remote moved, stop and report. Never force.

- [ ] **Step 4: Confirm the remote matches**

```bash
git fetch origin
git status -sb | head -2
git rev-parse --short HEAD origin/main
```

Expected: `## main...origin/main` with nothing ahead or behind, and both hashes equal.

- [ ] **Step 5: Report**

Report to the repo owner: the three commit hashes, the verifier's last line from Task 2 and Task 5, and a reminder of what is still deferred (untracking the 1.7 GB of data after an archive copy, the PPMI files, history, and the v1 account strings). Teammates who pull will find any ignored local files they had inside the old folders left at the old paths. Nothing is deleted, and they can move those files into `v1/` by hand.

---

## Rollback

Before the push, revert the three commits, newest first. Use the hashes from `git log --oneline -3`:

```bash
git revert --no-edit HEAD HEAD~1 HEAD~2
```

On the rehearsal clone this restored a tree whose hash was identical to the baseline tree. It makes new commits and rewrites nothing. After the push the same command still works, followed by a normal push.

A revert moves only tracked files. Ignored local files (data someone downloaded, notebook checkpoints) stay under `v1/`. Move each back by hand, for example `mv v1/data/gait/some_local_file.csv data/gait/`, or leave them under `v1/`.

## Self-review notes

Spec coverage:

- Purpose and goals 1-5: Tasks 2 (v1 intact), 3 (short root, router README), 4 (v2 first document), 1 and 5 (checks as commands).
- Target layout: Tasks 2, 3 and 4 produce it, and Task 5 step 1 checks the root list.
- Procedure: pre-flight is Task 1, commits 1-3 are Tasks 2-4, and the single push is Task 5.
- Verification items 1-9: hashes, 100% renames, file count, ignored local files, ignore rules, `--follow`, `git status` and `check-ignore`, links, scans are all verifier checks (Tasks 2 and 5). The tests item is Task 2 step 4 and Task 5 step 2.
- Rollback: `git revert` instructions in Tasks 2 and 5.
- Effects on others: stated in Task 5 step 5.
- Deferred: nothing in this plan does any of it.
