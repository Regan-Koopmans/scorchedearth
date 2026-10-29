"""Precomputed embeddings: vectors someone else computed, looked up by place and year.

Unlike live models, nothing is run here. Each task patch is located on the map, the
stored pixel embeddings under it are averaged, and the result is scored by the same
probes. Only tasks whose imagery carries coordinates can be used.

AlphaEarth Foundations (Google / Google DeepMind), annual, 64 dims per 10 m pixel.
Read anonymously from the TGE Labs copy on Source Cooperative, CC-BY 4.0. That copy is
not official. Its COGs are stored bottom-up (origin at the south edge); rows are mapped
by hand rather than through the VRTs, which proved ~3x slower.

Patches are averaged from the 80 m overview: each overview pixel is the normalised mean
of the 10 m pixels under it. Against full resolution, patch embeddings agree to cosine
>= 0.9997 on spot checks, and a whole file's 80 m layer is a single 1024x1024 block, so
one read serves every patch in it.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.warp import transform_bounds

INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
INDEX = Path("data/precomputed/aef/aef_index.parquet")
S3 = "s3://us-west-2.opendata.source.coop/"
HTTPS = "/vsicurl/https://data.source.coop/"
# Source Cooperative rejects some default client User-Agents.
USER_AGENT = "scorchedearth/0.1 (+https://github.com/Regan-Koopmans/scorchedearth)"
LEVEL = 2  # overview index: 2 -> 80 m
NODATA = -128
DIM = 64
FIRST_YEAR = 2017  # first annual AlphaEarth layer

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_USERAGENT": USER_AGENT,
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
}


def download_index() -> Path:
    if not INDEX.exists():
        import urllib.request

        INDEX.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
        INDEX.write_bytes(urllib.request.urlopen(req).read())
    return INDEX


def load_index(year: int):
    cols = ["crs", "year", "path", "utm_west", "utm_south", "utm_east", "utm_north",
            "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
    t = pq.read_table(download_index(), columns=cols).to_pandas()
    return t[t.year == year].reset_index(drop=True)


def dequantise(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.float32)
    return (v / 127.5) ** 2 * np.sign(v)


def locate(patches: list[tuple], idx) -> list[int | None]:
    """For each patch (crs, west, south, east, north), the index row of the file whose pixel
    grid covers the largest share of it, or None if none overlaps. The WGS84 outlines only
    shortlist candidates (they can overhang a file's real extent); overlap is measured in
    each file's UTM bounds. A patch on a file edge uses the file holding most of it."""
    out = []
    w, s, e, n = (idx[c].to_numpy() for c in ("wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"))
    for crs, *b in patches:
        lo = transform_bounds(crs, "EPSG:4326", *b)
        near = np.flatnonzero((w <= lo[2]) & (e >= lo[0]) & (s <= lo[3]) & (n >= lo[1]))
        best, best_share = None, 0.0
        for r in near:
            row = idx.iloc[r]
            ub = transform_bounds(crs, row.crs, *b)
            ox = max(0.0, min(ub[2], row.utm_east) - max(ub[0], row.utm_west))
            oy = max(0.0, min(ub[3], row.utm_north) - max(ub[1], row.utm_south))
            share = ox * oy / ((ub[2] - ub[0]) * (ub[3] - ub[1]))
            if share > best_share:
                best, best_share = int(r), share
        out.append(best)
    return out


def _read_layer(tiff: str, tries: int = 5):
    """A file's whole 80 m layer. The server sometimes truncates a response mid-tile, so
    retry with backoff; GDAL's own retry does not cover a short read."""
    for attempt in range(tries):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(tiff, overview_level=LEVEL) as src:
                return src.read(), src.transform, abs(src.transform.e)
        except rasterio.errors.RasterioIOError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2**attempt)


def _embed_file(row, items: list[tuple[int, tuple]]) -> list[tuple[int, np.ndarray]]:
    """Read one file's 80 m layer once and average each patch window from it."""
    tiff = row.path.replace(S3, HTTPS)
    a, tr, res = _read_layer(tiff)  # a: (64, H, W), int8
    if tr.e < 0:
        raise ValueError(f"Expected a bottom-up COG, got top-down: {tiff}")
    out = []
    for i, (crs, *b) in items:
        ub = transform_bounds(crs, row.crs, *b)
        # Bottom-up: row 0 is the south edge; rows grow northward.
        c0, c1 = int(np.floor((ub[0] - tr.c) / res)), int(np.ceil((ub[2] - tr.c) / res))
        r0, r1 = int(np.floor((ub[1] - tr.f) / res)), int(np.ceil((ub[3] - tr.f) / res))
        c0, r0 = max(c0, 0), max(r0, 0)
        win = a[:, r0:r1, c0:c1]
        valid = (win != NODATA).all(axis=0)
        out.append((i, dequantise(win)[:, valid].mean(axis=1) if valid.any() else np.full(DIM, np.nan, np.float32)))
    return out


def embed_patches(patches: list[tuple], year: int, workers: int = 3, log=print, partial: Path | None = None) -> np.ndarray:
    """AlphaEarth embedding per patch. patches: (crs, west, south, east, north) in the
    patch's own CRS. Returns (N, 64) float32; rows with no data are NaN.

    With `partial`, each finished file's results are saved there as they complete, and a
    rerun skips files already done. A file that still fails after retries is logged and
    its patches stay NaN, so one bad file cannot sink the run."""
    idx = load_index(year)
    rows = locate(patches, idx)
    groups: dict[int, list] = {}
    for i, (p, r) in enumerate(zip(patches, rows)):
        if r is not None:
            groups.setdefault(r, []).append((i, p))
    out = np.full((len(patches), DIM), np.nan, np.float32)
    done_files: set[str] = set()
    if partial is not None and partial.exists():
        z = np.load(partial, allow_pickle=False)
        out, done_files = z["emb"], set(z["done"].tolist())
    todo = {r: items for r, items in groups.items() if idx.iloc[r].path not in done_files}
    log(f"  {len(patches):,} patches in {len(groups)} AlphaEarth files; {sum(r is None for r in rows)} uncovered; "
        f"{len(groups) - len(todo)} files already done")

    def save():
        if partial is not None:
            np.savez(partial, emb=out, done=np.array(sorted(done_files)))

    t0, n, failed = time.time(), 0, []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_embed_file, idx.iloc[r], items): r for r, items in todo.items()}
        for f in as_completed(futs):
            path = idx.iloc[futs[f]].path
            try:
                for i, v in f.result():
                    out[i] = v
                done_files.add(path)
            except Exception as e:
                failed.append(path)
                log(f"  FAILED after retries, patches left empty: {path.rsplit('/', 3)[-3:]} ({type(e).__name__})")
            n += 1
            if n % 10 == 0 or n == len(todo):
                save()
                log(f"  {n}/{len(todo)} files, {time.time() - t0:.0f}s")
    save()
    if failed:
        log(f"  {len(failed)} files failed; rerun to retry them")
    return out


# --- OlmoEarth v1.3 (Ai2) ------------------------------------------------------------
# Annual, 128 dims per 10 m pixel (the first 64 are a complete Matryoshka embedding), on
# Source Cooperative as a Zarr v3 store with one group per UTM zone. No overviews, so each
# patch is averaged at 10 m. Reads are grouped by 256 px chunk row so that patches sharing
# chunks fetch them once. OlmoEarth Artifact License.

OLMO_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
OLMO_DIM = 64  # bands 0:64, stored in their own chunks, half the download of all 128
CHUNK = 256


def _olmo_root():
    import zarr
    from obstore.store import S3Store

    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store.from_url(OLMO_URL, endpoint="https://data.source.coop", skip_signature=True)
    return zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")


def _olmo_windows(patches: list[tuple], root) -> list[tuple]:
    """(zone, row0, row1, col0, col1) pixel window per patch in its UTM zone's grid."""
    from pyproj import Transformer

    out, tf_cache = [], {}
    for crs, w, s_, e, n in patches:
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform((w + e) / 2, (s_ + n) / 2)
        zone = f"utm{int((lon + 180) // 6) + 1:02d}"
        zg = root[zone]
        key = (str(crs), zone)
        if key not in tf_cache:
            tf_cache[key] = (Transformer.from_crs(crs, zg.attrs["proj:code"], always_xy=True), zg.attrs["spatial:transform"])
        tf, (a, _, c, _, ey, f) = tf_cache[key]
        xs, ys = tf.transform([w, e, w, e], [s_, s_, n, n])
        out.append((zone, int((max(ys) - f) / ey), int(np.ceil((min(ys) - f) / ey)),
                    int((min(xs) - c) / a), int(np.ceil((max(xs) - c) / a))))
    return out


def _olmo_dequantise(q: np.ndarray) -> np.ndarray:
    v = np.where(q == NODATA, np.nan, q).astype(np.float32)
    return (np.abs(v) / 127.5) ** 2.0 * np.sign(v)


def _olmo_block(root, zone, t, rows, cols, items, tries=5):
    """Read one block of whole chunks once and average every patch window inside it."""
    r0, r1, c0, c1 = rows[0] * CHUNK, (rows[1] + 1) * CHUNK, cols[0] * CHUNK, (cols[1] + 1) * CHUNK
    for attempt in range(tries):
        try:
            q = root[zone]["embeddings"][t, 0:OLMO_DIM, r0:r1, c0:c1]
            break
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2**attempt)
    v = _olmo_dequantise(q)
    out = []
    for i, (_, pr0, pr1, pc0, pc1) in items:
        win = v[:, pr0 - r0 : pr1 - r0, pc0 - c0 : pc1 - c0]
        ok = np.isfinite(win).all(axis=0)
        out.append((i, win[:, ok].mean(axis=1) if ok.any() else np.full(OLMO_DIM, np.nan, np.float32)))
    return out


def olmoearth_patches(patches: list[tuple], year: int, workers: int = 8, log=print, partial: Path | None = None) -> np.ndarray:
    """OlmoEarth v1.3 embedding (first 64 dims) per patch, averaged at 10 m. Same contract
    as embed_patches: (N, 64) float32, NaN rows where no data; resumable via `partial`."""
    root = _olmo_root()
    wins = _olmo_windows(patches, root)
    times = {z: int(np.where(root[z]["time"][:] == year)[0][0]) for z in {w[0] for w in wins}}
    # Group patches by (zone, chunk-row band): one read per band of chunks they share.
    groups: dict[tuple, list] = {}
    for i, w in enumerate(wins):
        zone, pr0, pr1, pc0, pc1 = w
        groups.setdefault((zone, pr0 // CHUNK, (pr1 - 1) // CHUNK, pc0 // (CHUNK * 8)), []).append((i, w))
    out = np.full((len(patches), OLMO_DIM), np.nan, np.float32)
    done: set[str] = set()
    if partial is not None and partial.exists():
        z = np.load(partial, allow_pickle=False)
        out, done = z["emb"], set(z["done"].tolist())
    keyname = lambda k: "|".join(map(str, k))
    todo = {k: v for k, v in groups.items() if keyname(k) not in done}
    log(f"  {len(patches):,} patches in {len(groups)} read blocks across {len(times)} UTM zones; {len(groups) - len(todo)} done")

    def save():
        if partial is not None:
            np.savez(partial, emb=out, done=np.array(sorted(done)))

    t0, n, failed = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {}
        for k, items in todo.items():
            zone = k[0]
            rows = (min(w[1] for _, w in items) // CHUNK, (max(w[2] for _, w in items) - 1) // CHUNK)
            cols = (min(w[3] for _, w in items) // CHUNK, (max(w[4] for _, w in items) - 1) // CHUNK)
            futs[pool.submit(_olmo_block, root, zone, times[zone], rows, cols, items)] = k
        for f in as_completed(futs):
            k = futs[f]
            try:
                for i, v in f.result():
                    out[i] = v
                done.add(keyname(k))
            except Exception as e:
                failed += 1
                log(f"  FAILED after retries, patches left empty: {k} ({type(e).__name__})")
            n += 1
            if n % 25 == 0 or n == len(todo):
                save()
                log(f"  {n}/{len(todo)} blocks, {time.time() - t0:.0f}s")
    save()
    if failed:
        log(f"  {failed} blocks failed; rerun to retry them")
    return out


CACHE = Path("cache/precomputed")
SOURCES = {"alphaearth": embed_patches, "olmoearth": olmoearth_patches}


def build(task_module, source: str = "alphaearth") -> dict:
    """Look up and cache precomputed embeddings for every kept patch of a task.
    The task module must provide patch_footprints(split) and YEAR."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("training", "validation"):
        path = CACHE / f"{task_module.NAME}-{source}-{split}.npz"
        if path.exists():
            out[split] = path
            print(f"[{task_module.NAME}/{split}] cached: {path}")
            continue
        got = task_module.patch_footprints(split)
        feet, labels = got[0], got[1]
        # Per-patch event years where a task has them, else the task's single year.
        years = got[2] if len(got) > 2 else np.full(len(feet), task_module.YEAR)
        years = np.maximum(years, FIRST_YEAR)  # earlier events use the first available year
        print(f"[{task_module.NAME}/{split}] {len(feet):,} patches, years {sorted(set(years.tolist()))}", flush=True)
        emb = np.full((len(feet), DIM), np.nan, np.float32)
        for y in sorted(set(years.tolist())):
            sel = np.flatnonzero(years == y)
            partial = CACHE / f"{task_module.NAME}-{source}-{split}-{y}.partial.npz"
            emb[sel] = SOURCES[source]([feet[i] for i in sel], int(y), log=lambda m: print(m, flush=True), partial=partial)
        if np.isnan(emb).any(axis=1).mean() > 0.01:
            print(f"[{task_module.NAME}/{split}] {int(np.isnan(emb).any(axis=1).sum())} patches still empty; "
                  "not finalising. Rerun to retry failed files.", flush=True)
            continue
        np.savez(path, emb=emb, labels=labels, years=years)
        for y in set(years.tolist()):
            (CACHE / f"{task_module.NAME}-{source}-{split}-{y}.partial.npz").unlink(missing_ok=True)
        print(f"[{task_module.NAME}/{split}] saved, {int(np.isnan(emb).any(axis=1).sum())} patches without data", flush=True)
        out[split] = path
    return out
