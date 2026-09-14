"""Download the LINCS 2020 beta files from the CLUE S3 bucket.

    python -m genetomol.download --dest ../lincs --what metadata
    python -m genetomol.download --dest ../lincs --what matrix

Metadata first. It is ~470 MB and `census.py` can already tell you from it
whether the project is viable; the level-5 matrix is **35.5 GB** and there is no
point pulling it before the census looks sane.

Resumable: transfers go to a `.part` file and continue with an HTTP Range
request if interrupted. The bucket returns `Accept-Ranges: bytes`, so a dropped
35 GB download does not start over.

Verification is by **byte size only**. CLUE does not publish checksums for these
objects, so a size match means "complete", not "uncorrupted". `--verify` re-checks
files already on disk against the sizes recorded here, which were read from the
server's Content-Length headers.

Stdlib only -- no requests, no boto3, no AWS credentials. The bucket is public
for reads but does not permit listing, so the file table below is hardcoded.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020"


@dataclass(frozen=True)
class Remote:
    name: str          # local filename
    path: str          # path under BASE_URL
    size: int          # bytes, from the server's Content-Length
    group: str
    note: str = ""

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.path}"


FILES: tuple[Remote, ...] = (
    Remote("geneinfo_beta.txt", "geneinfo_beta.txt", 1_141_389, "metadata",
           "gene_id -> symbol + feature_space (landmark flag)"),
    Remote("cellinfo_beta.txt", "cellinfo_beta.txt", 37_979, "metadata",
           "cell line annotations; not used by the model, handy for diagnostics"),
    Remote("compoundinfo_beta.txt", "compoundinfo_beta.txt", 4_631_014, "metadata",
           "pert_id -> canonical_smiles, moa, target"),
    Remote("siginfo_beta.txt", "siginfo_beta.txt", 465_242_319, "metadata",
           "per-signature metadata incl. tas and is_exemplar_sig"),
    Remote("level5_beta_trt_cp_n720216x12328.gctx",
           "level5/level5_beta_trt_cp_n720216x12328.gctx",
           35_518_405_386, "matrix",
           "level-5 compound signatures, 720216 x 12328"),
)

_BY_GROUP = {"metadata": [f for f in FILES if f.group == "metadata"],
             "matrix": [f for f in FILES if f.group == "matrix"],
             "all": list(FILES)}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:.1f} TB"


def _open(url: str, offset: int = 0, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "genetomol/0.1"})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=timeout)


def download_one(
    remote: Remote,
    dest_dir: Path,
    chunk: int = 1 << 20,
    retries: int = 5,
    progress_every: float = 15.0,
) -> Path:
    """Fetch one file, resuming a partial `.part` if present."""
    final = dest_dir / remote.name
    part = dest_dir / (remote.name + ".part")

    if final.exists():
        actual = final.stat().st_size
        if actual == remote.size:
            logger.info("%s already complete (%s) -- skipping", remote.name, human(actual))
            return final
        logger.warning(
            "%s exists but is %s, expected %s -- re-downloading",
            remote.name, human(actual), human(remote.size),
        )
        final.unlink()

    attempt = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if offset > remote.size:
            logger.warning("%s partial is larger than expected -- restarting", remote.name)
            part.unlink()
            offset = 0
        if offset == remote.size:
            break

        try:
            resp = _open(remote.url, offset)
            # If the server ignored our Range header we must start over, or we
            # would append the whole file onto the partial and silently corrupt it.
            if offset and resp.status != 206:
                logger.warning("server ignored Range (status %s) -- restarting", resp.status)
                resp.close()
                part.unlink(missing_ok=True)
                continue

            mode = "ab" if offset else "wb"
            done, t0, last = offset, time.monotonic(), time.monotonic()
            with resp, open(part, mode) as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    now = time.monotonic()
                    if now - last >= progress_every:
                        rate = (done - offset) / max(now - t0, 1e-9)
                        eta = (remote.size - done) / rate if rate > 0 else float("inf")
                        logger.info(
                            "  %s  %s / %s (%.1f%%)  %s/s  eta %s",
                            remote.name, human(done), human(remote.size),
                            100 * done / remote.size, human(rate),
                            time.strftime("%H:%M:%S", time.gmtime(eta)) if eta < 1e6 else "?",
                        )
                        last = now
            attempt = 0  # a clean pass resets the retry budget
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"{remote.name}: giving up after {retries} retries ({exc}). "
                    f"The partial file is kept at {part}; re-run to resume."
                ) from exc
            wait = min(2 ** attempt, 60)
            logger.warning("%s: %s -- retry %d/%d in %ds", remote.name, exc,
                           attempt, retries, wait)
            time.sleep(wait)
            continue

        if part.stat().st_size >= remote.size:
            break

    actual = part.stat().st_size
    if actual != remote.size:
        raise RuntimeError(
            f"{remote.name}: got {actual:,} bytes, expected {remote.size:,}. "
            f"Left at {part} -- re-run to resume."
        )
    part.replace(final)
    logger.info("%s complete (%s)", remote.name, human(actual))
    return final


def verify(dest_dir: Path, wanted: list[Remote]) -> bool:
    ok = True
    print(f"{'file':<46}{'on disk':>14}{'expected':>14}  status")
    print("-" * 92)
    for r in wanted:
        p = dest_dir / r.name
        if not p.exists():
            part = dest_dir / (r.name + ".part")
            status = (f"PARTIAL {100*part.stat().st_size/r.size:.1f}%"
                      if part.exists() else "MISSING")
            got = human(part.stat().st_size) if part.exists() else "-"
            ok = False
        else:
            got = human(p.stat().st_size)
            if p.stat().st_size == r.size:
                status = "ok"
            else:
                status = "SIZE MISMATCH"
                ok = False
        print(f"{r.name:<46}{got:>14}{human(r.size):>14}  {status}")
    print("\nsize check only -- CLUE publishes no checksums for these objects.")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dest", required=True, help="directory to download into")
    ap.add_argument("--what", default="metadata", choices=("metadata", "matrix", "all"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="download just these filenames")
    ap.add_argument("--verify", action="store_true",
                    help="check sizes of what is already on disk, download nothing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dest = Path(args.dest)

    wanted = _BY_GROUP[args.what]
    if args.only:
        names = set(args.only)
        wanted = [f for f in FILES if f.name in names]
        missing = names - {f.name for f in wanted}
        if missing:
            ap.error(f"unknown file(s): {sorted(missing)}. "
                     f"Known: {sorted(f.name for f in FILES)}")

    if args.verify:
        dest.mkdir(parents=True, exist_ok=True)
        raise SystemExit(0 if verify(dest, wanted) else 1)

    total = sum(f.size for f in wanted)
    print(f"\ndestination: {dest.resolve()}")
    print(f"{'file':<46}{'size':>12}   note")
    print("-" * 100)
    for f in wanted:
        print(f"{f.name:<46}{human(f.size):>12}   {f.note}")
    print("-" * 100)
    print(f"{'TOTAL':<46}{human(total):>12}\n")

    dest.mkdir(parents=True, exist_ok=True)
    already = sum(
        (dest / f.name).stat().st_size for f in wanted if (dest / f.name).exists()
    )
    free = shutil.disk_usage(dest).free
    need = total - already
    print(f"free space on target drive: {human(free)}   still to fetch: {human(need)}")
    if free < need * 1.05:
        print(f"\nNOT ENOUGH SPACE: need ~{human(need * 1.05)} including headroom.")
        raise SystemExit(2)

    if args.dry_run:
        print("\n--dry-run: nothing downloaded")
        return

    t0 = time.monotonic()
    for f in wanted:
        download_one(f, dest, retries=args.retries)
    print(f"\ndone in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}")
    verify(dest, wanted)

    if args.what == "metadata":
        print("\nNext:")
        print(f"  python -m genetomol.census \\")
        print(f"      --siginfo {dest / 'siginfo_beta.txt'} \\")
        print(f"      --compoundinfo {dest / 'compoundinfo_beta.txt'} \\")
        print(f"      --out {dest / 'census.json'}")


if __name__ == "__main__":
    main()
