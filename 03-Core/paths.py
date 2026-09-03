"""
Path + integrity resolution for Xpect AI.

The single biggest deployment failure mode for this app is that `chroma_data`
is tracked with Git LFS, and Streamlit Cloud clones the repo WITHOUT running
the LFS smudge filter. The folder therefore exists on the cloud box, but every
file inside it is a ~130 byte text pointer that looks like:

    version https://git-lfs.github.com/spec/v1
    oid sha256:492c30ae...
    size 32083968

ChromaDB opens that "database", finds no collections, and the app prints
"Collection not found. Run pipeline first."

So checking `os.path.exists("chroma_data")` is NOT enough - it is always True.
We have to check that the store is *real* and *non-empty*.
"""

import os
import shutil
import tempfile

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"

# 03-Core/paths.py  ->  03-Core  ->  repo root
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)


def is_lfs_pointer(path):
    """True if `path` is a Git LFS pointer stub rather than the real file."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def store_status(path):
    """
    Classify a candidate chroma_data directory.

    Returns one of: "missing", "empty", "lfs-pointers", "ok".
    """
    if not os.path.isdir(path):
        return "missing"

    files = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            files.append(os.path.join(root, name))

    if not files:
        return "empty"
    if all(is_lfs_pointer(f) for f in files):
        return "lfs-pointers"
    sqlite = os.path.join(path, "chroma.sqlite3")
    if os.path.isfile(sqlite) and is_lfs_pointer(sqlite):
        return "lfs-pointers"
    return "ok"


def candidate_chroma_paths():
    """Every place a prebuilt chroma_data could reasonably live."""
    env = os.getenv("CHROMA_PATH")
    candidates = []
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(BASE_DIR, "chroma_data"),   # repo root (normal layout)
        os.path.join(CORE_DIR, "chroma_data"),   # 03-Core/chroma_data
        os.path.join(os.getcwd(), "chroma_data"),
    ]
    # de-dupe, preserve order
    seen, out = set(), []
    for c in candidates:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def writable_build_path():
    """
    Where we build the DB when no usable prebuilt store exists.

    Prefer the repo root so a local run produces the familiar ./chroma_data.
    Fall back to a temp dir when the deploy filesystem is read-only.
    """
    preferred = os.path.abspath(
        os.getenv("CHROMA_PATH") or os.path.join(BASE_DIR, "chroma_data")
    )
    parent = os.path.dirname(preferred)
    if os.access(parent, os.W_OK):
        return preferred
    return os.path.join(tempfile.gettempdir(), "xpect_chroma_data")


def resolve_chroma_path(verbose=True):
    """
    Find a usable chroma_data directory.

    Returns (path, status) where status is "ok" if the store is real and
    ready to open, otherwise the reason we have to rebuild.
    """
    report = []
    for path in candidate_chroma_paths():
        status = store_status(path)
        report.append((path, status))
        if status == "ok":
            if verbose:
                _log_report(report, chosen=path)
            return path, "ok"

    build_path = writable_build_path()
    # If the chosen build target currently holds LFS stubs, clear them out so
    # ChromaDB does not choke on a corrupt sqlite file.
    if store_status(build_path) == "lfs-pointers":
        if verbose:
            print(f"[xpect] Removing Git LFS pointer stubs at {build_path}")
        try:
            shutil.rmtree(build_path)
        except OSError as exc:
            print(f"[xpect] Could not remove stub store: {exc}")
            build_path = os.path.join(tempfile.gettempdir(), "xpect_chroma_data")

    reason = report[0][1] if report else "missing"
    if verbose:
        _log_report(report, chosen=None, build_path=build_path)
    return build_path, reason


def _log_report(report, chosen=None, build_path=None):
    print("[xpect] chroma_data lookup:")
    for path, status in report:
        print(f"[xpect]   {status:<13} {path}")
    if chosen:
        print(f"[xpect] Using prebuilt store: {chosen}")
    else:
        print("[xpect] No usable prebuilt store found.")
        print(f"[xpect] Will build the collection at: {build_path}")


def resolve_csv_path():
    """Locate the Netflix CSV used to (re)build the collection."""
    names = ["netflix_titles.csv", "cleaned_netflix_titles.csv"]
    roots = [
        os.path.join(BASE_DIR, "01-Data"),
        os.path.join(CORE_DIR, "01-Data"),
        os.path.join(CORE_DIR, "..", "01-Data"),
        os.path.join(os.getcwd(), "01-Data"),
        BASE_DIR,
        os.getcwd(),
    ]
    for root in roots:
        for name in names:
            candidate = os.path.abspath(os.path.join(root, name))
            if os.path.isfile(candidate) and not is_lfs_pointer(candidate):
                return candidate
    raise FileNotFoundError(
        "netflix_titles.csv not found. Looked in: "
        + ", ".join(os.path.abspath(r) for r in roots)
    )
