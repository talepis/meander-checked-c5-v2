Checked C5/V2 closed meander computation
========================================

Research release candidate 1.0.0-rc1, 27 October 2026.
Copyright (c) 2026 Efthimios Alepis. Source license: MIT.
ORCID: https://orcid.org/0000-0002-3862-7218 (confirmed by the author; see provenance/AUTHOR_IDENTITY.json).

Scope
-----

This archive contains an exact CPU implementation of the C5/V2 transfer for
closed meander numbers M_n, plus a modular C5/V2 reproduction mode and a
separate original-transfer modular worker for n <= 31. The audited domain is
n=0..34. M35 was still running at release preparation and is deliberately
absent. No GPU or disk-backed coefficient arrays are used. The fast V2 bound
is a *checked prediction*, not a proved universal maximizer: a rejected
prediction causes a clean retry under an unconditional proved cap. See
``docs/METHOD_EN.txt`` and ``docs/proofs/CHECKED_V2_CORRECTNESS_EN.txt``.

Historical CRT records for M29..M34 are in ``evidence/``. M29..M31 have an
independent original-transfer audit relative to C5/V2; M32..M34 use the
checked V2 mathematics and are not independent evidence for its identities.
The public checker verifies SHA-256 of every retained numerical record,
congruences, CRT reconstruction, a Catalan-squared uniqueness bound, and the
0..34 table. Hashes document file integrity; they are not a third-party
attestation that a past job ran. Proof prerequisites and limitations are
explicit in ``docs/``.

Requirements
------------

Python 3.12, x86-64 Linux or Windows, a C compiler on Linux for the RAM guard,
NumPy, Numba with OpenMP, and psutil. Use a fresh virtual environment. A
reasonable first run is n=8 and 2 GB free memory. Large n can require many
hours and hundreds of GiB; do not launch several large jobs concurrently.
``--ram-gib`` caps usable memory and ``--reserve-gib`` preserves a margin.
Both free host memory and Linux cgroup limits are checked at startup. The
program rejects too-small budgets; it does not silently spill coefficient
arrays to disk. The first invocation includes JIT compilation. This public distribution wrapper was smoke tested on Windows x64; Linux has not been retested in this release task because WSL access was denied (see evidence/SMOKE_TEST_REPORT.json).

On Linux::

    python3.12 -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python meander.py 8 --threads 2 --ram-gib 4

On Windows PowerShell::

    py -3.12 -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python meander.py 8 --threads 2 --ram-gib 4

The exact answer is printed and recorded as JSON in ``runs/``. For a chosen
modulus and a bounded small order::

    python meander.py 8 --mode modular --modulus 1927 --threads 2 --ram-gib 4
    python meander.py 8 --mode independent --modulus 1927 --threads 2 --ram-gib 4

For a full three-pass fresh-process CRT reproduction::

    python tools/run_crt.py 8 --mode independent --threads 2 --ram-gib 4 --output-dir runs/crt8

This command can be far more expensive than a single exact calculation. Check
historical evidence without recomputing the large orders::

    python tools/verify_certificates.py

Data, evidence and attribution
------------------------------

``data/b005315.txt`` is a candidate b-file through n=34, not an OEIS-approved
extension. The separately recorded OEIS reference covers n<=28; consult
``docs/REFERENCES.txt`` before asserting novelty or priority. ``data/`` also
contains a precomputed closing map; its SHA-256 is checked before loading.
``provenance/NUMERICAL_SOURCE_SHA256.json`` identifies preserved numerical
sources. Public CRT numerical records were sanitized from internal execution
logs and rebound by public SHA-256 in ``evidence/CRT_CERTIFICATES.json``.
The transformations are described in ``provenance/``; raw private service
credentials and machine orchestration are deliberately excluded.

The MIT license covers code and documentation contributed to this release;
third-party packages are installed separately, not relicensed here. See
``THIRD_PARTY_NOTICES.txt``. The code and prose were assembled with AI
assistance and require human scientific review before citation/submission.
See ``docs/PUBLISH_GR.txt`` for release steps and OEIS editorial guidance.

Results must be read with their scope: the scalar and memory guards and
included tests support the checked implementation but do not constitute a
formal proof of the entire Python/Numba/compiler stack. A CRT reconstruction
is conditional on correct residues; independence differs by order. Use the public hosting records for the current repository URL and DOI; an offline copy of this candidate package does not certify either assignment.
