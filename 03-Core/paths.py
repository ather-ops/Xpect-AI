"""Locate and validate the ChromaDB vector store."""

import os
import shutil
import tempfile

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

COLLECTION = "netflix_titles"
EMBED_MODEL = "all-MiniLM-L6-v2"

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CORE_DIR)


def is_lfs_pointer(path):
    """True if path is a Git LFS pointer stub rather than the real file."""
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def store_status(path):
    """Classify a chroma_data directory: missing, empty, lfs-pointers or ok."""
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
    """Every location a prebuilt chroma_data could live in."""
    env = os.getenv("CHROMA_PATH")
    candidates = [env] if env else []
    candidates += [
        os.path.join(BASE_DIR, "chroma_data"),
        os.path.join(CORE_DIR, "chroma_data"),
        os.path.join(os.getcwd(), "chroma_data"),
    ]

    seen, out = set(), []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def writable_build_path():
    """Where to build the store, falling back to temp on read-only filesystems."""
    preferred = os.path.abspath(
        os.getenv("CHROMA_PATH") or os.path.join(BASE_DIR, "chroma_data")
    )
    if os.access(os.path.dirname(preferred), os.W_OK):
        return preferred
    return os.path.join(tempfile.gettempdir(), "xpect_chroma_data")


def resolve_chroma_path(verbose=True):
    """Return (path, status) for a usable store, or a build target to populate."""
    report = []
    for path in candidate_chroma_paths():
        status = store_status(path)
        report.append((path, status))
        if status == "ok":
            if verbose:
                _log_report(report, chosen=path)
            return path, "ok"

    build_path = writable_build_path()

    if store_status(build_path) == "lfs-pointers":
        if verbose:
            print(f"[xpect] Removing Git LFS pointer stubs at {build_path}")
        try:
            shutil.rmtree(build_path)
        except OSError as exc:
            print(f"[xpect] Could not remove stub store: {exc}")
            build_path = os.path.join(tempfile.gettempdir(), "xpect_chroma_data")

    if verbose:
        _log_report(report, chosen=None, build_path=build_path)
    return build_path, report[0][1] if report else "missing"


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
    """Locate the Netflix CSV used to build the collection."""
    names = ["netflix_titles.csv", "cleaned_netflix_titles.csv"]
    roots = [
        os.path.join(BASE_DIR, "01-Data"),
        os.path.join(CORE_DIR, "01-Data"),
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
