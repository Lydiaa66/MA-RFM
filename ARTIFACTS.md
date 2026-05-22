# Intermediate Artifacts

Large experiment artifacts are intentionally kept out of Git. They are useful for
local reproduction and fast plotting, but they make the repository too large for
normal GitHub use.

The local files are still stored at their original paths. The repository commit
only stops tracking them and adds ignore rules so they are not uploaded again.

See `ARTIFACTS_MANIFEST.tsv` for the current list of artifacts that were moved
out of Git tracking. The manifest records each path, file type, tracked size,
and whether the local file still exists.

Current artifact summary:

```text
539 files, about 1.1 GiB tracked size

.npy   215 files, about 742.7 MiB
.pkl    50 files, about 338.0 MiB
.pth   170 files, about 21.1 MiB
.pyc   103 files, about 1.2 MiB
.npz     1 file, about 117 KiB
```

Suggested handling:

- Keep source code, notebooks, README files, and final selected figures in Git.
- Keep `.npy`, `.pkl`, `.pth`, `.npz`, and Python bytecode files outside Git.
- If a small generated figure or table is needed for the paper or README, add it
  explicitly with `git add -f <path>`.
- For long-term reproducibility, store the artifact bundle outside GitHub, for
  example on a lab server, institutional storage, or a data repository, and add
  a download link here.
