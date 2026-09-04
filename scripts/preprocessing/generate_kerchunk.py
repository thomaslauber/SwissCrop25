import os
import tarfile
import zipfile
import json
import io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from kerchunk.utils import consolidate
import math
from collections import defaultdict
import string
import itertools

# ---------------- Helper: Generate short template keys ---------------- #
def natural_sort_key(path_str):
    """
    Generate a key for natural sorting (e.g., file1, file2, file10 instead of file1, file10, file2).
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path_str)]


def generate_placeholder_names():
    """
    Generate short placeholder names: a, b, c, ..., z, aa, ab, ..., zz, aaa, ...
    """
    alphabet = string.ascii_lowercase
    length = 1
    while True:
        for name in map("".join, itertools.product(alphabet, repeat=length)):
            yield name
        length += 1


# ---------------- Core ZIP entry parsing ---------------- #

def parse_zip_entries(zf, source, container_abs_path, tar_offset=0,
                      parallel=False, num_workers=32, show_progress=False):
    """
    Parse entries from an open ZipFile `zf`.
    - source: Path (for on-disk zip) or bytes (for an in-memory zip).
    - container_abs_path: absolute path (string) of the container file (used in refs).
    - tar_offset: if zip lives inside a tar, offsets are relative to tar_offset.
    - parallel: use threads (only valid when `source` is a Path).
    - show_progress: show tqdm progress in the current process (don't use inside worker processes).
    Returns: dict {entry_name: [container_abs_path, offset, compress_size], ...}
    """
    refs = {}
    info_list = zf.infolist()
    container_name = Path(container_abs_path).name

    # Compute top-level folders (tiles)
    top_level_dirs = sorted({name.strip("/").split("/")[0] for name in zf.namelist() if name.strip()})
    num_tiles = max(len(top_level_dirs), 1)
    num_tiles = math.ceil(num_tiles)  # <-- ensures integer total for tqdm

    def process_entry(zi):
        if zi.is_dir() or zi.file_size == 0:
            return None
        header_offset = zi.header_offset
        if isinstance(source, Path):
            with open(source, "rb") as f:
                f.seek(header_offset + 26)
                name_len = int.from_bytes(f.read(2), "little")
                extra_len = int.from_bytes(f.read(2), "little")
                data_offset = header_offset + 30 + name_len + extra_len
        else:
            buf = memoryview(source)
            name_len = int.from_bytes(buf[header_offset + 26:header_offset + 28], "little")
            extra_len = int.from_bytes(buf[header_offset + 28:header_offset + 30], "little")
            data_offset = header_offset + 30 + name_len + extra_len

        return zi.filename, [container_abs_path, tar_offset + data_offset, zi.compress_size]

    # Initialize progress bar
    if show_progress:
        pbar = tqdm(total=num_tiles, desc=f"{container_name}", unit="dirs")
        scale = max(len(info_list) / num_tiles, 1)
        accumulated = 0.0
    else:
        pbar = None
        accumulated = None

    # Process entries
    if parallel and isinstance(source, Path):
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for result in ex.map(process_entry, info_list):
                if result:
                    fname, ref = result
                    refs[fname] = ref
                if pbar:
                    accumulated += 1 / scale
                    int_update = int(accumulated)
                    if int_update >= 1:
                        pbar.update(int_update)
                        accumulated -= int_update
    else:
        for zi in info_list:
            result = process_entry(zi)
            if result:
                fname, ref = result
                refs[fname] = ref
            if pbar:
                accumulated += 1 / scale
                int_update = int(accumulated)
                if int_update >= 1:
                    pbar.update(int_update)
                    accumulated -= int_update

    if pbar:
        # ensure progress bar reaches total
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
        pbar.close()

    return refs


def parse_zip_from_file(zip_path, num_workers=32, show_progress=True):
    """Parse a standalone zip on disk in parallel (shows progress)."""
    zip_path = Path(zip_path)
    abs_path = str(zip_path.resolve())
    with zipfile.ZipFile(zip_path) as zf:
        return parse_zip_entries(zf, zip_path, abs_path,
                                 tar_offset=0, parallel=True,
                                 num_workers=num_workers, show_progress=show_progress)


def parse_zip_from_bytes(zip_bytes, tar_abs_path, tar_offset, show_progress=False):
    """Parse a zip supplied as bytes (e.g. extracted from a tar). Runs single-threaded (for worker processes)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return parse_zip_entries(zf, zip_bytes, tar_abs_path,
                                 tar_offset=tar_offset, parallel=False,
                                 num_workers=1, show_progress=show_progress)


# ---------------- File-level processors ---------------- #

def process_zip(zip_path, num_workers=32):
    """Return refs mapping for a standalone zip file (container path is absolute)."""
    return parse_zip_from_file(zip_path, num_workers=num_workers, show_progress=True)


def process_tar(tar_path, max_workers=8, show_progress=True):
    """
    Process a .tar file that contains .zip files.
    Returns a refs mapping where keys are namespaced as <zip_stem>/<entry> and
    each ref points to container_abs_path (the tar) with the correct byte offsets.
    """
    tar_path = Path(tar_path)
    tar_abs = str(tar_path.resolve())
    all_refs = {}

    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".zip")]

        # submit jobs: extract zip bytes in main process (TarFile objects are not picklable)
        jobs = []
        for m in members:
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            data = fobj.read()
            # pass bytes + absolute tar path + member.offset_data
            jobs.append((m, data))

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [(ex.submit(parse_zip_from_bytes, data, tar_abs, m.offset_data), m) for (m, data) in jobs]

            for fut, m in tqdm(futures, total=len(futures),
                               desc=f"{tar_path.name}", unit="zip",
                               disable=not show_progress):
                refs = fut.result()
                # Keep .zarr extension in the prefix for proper zarr group access
                prefix = Path(m.name).name.replace('.zarr.zip', '.zarr').replace('.zip', '.zarr')
                # namespace entries so they don't collide
                namespaced = {f"{prefix}/{k}": v for k, v in refs.items()}
                all_refs.update(namespaced)

    return all_refs


def process_single_tar_for_webdataset(tar_path):
    """
    Process a single tar file for webdataset (no parallel processing within tar).
    Returns refs directly without additional namespacing (tar path is in template).
    """
    tar_refs = process_tar(tar_path, max_workers=1, show_progress=False)  # Sequential within tar, no progress bar
    return tar_refs  # Return directly - tar path is already in the template reference


def process_webdataset(folder_path, max_workers=8):
    """
    Process a folder containing .tar files (webdataset), where each tar contains .zarr.zip files.
    Parallelizes across tar files rather than within them.
    Returns (all_refs, templates) tuple.
    """
    folder_path = Path(folder_path)
    all_refs = {}
    templates = {}
    tar_files = sorted(folder_path.glob("*.tar"), key=lambda p: natural_sort_key(str(p)))

    if not tar_files:
        print(f"No .tar files found in {folder_path}")
        return all_refs, templates

    placeholder_gen = generate_placeholder_names()

    # Process tar files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [(executor.submit(process_single_tar_for_webdataset, tar_path), tar_path, next(placeholder_gen))
                   for tar_path in tar_files]

        for future, tar_path, placeholder in tqdm(futures, desc=f"Processing webdataset in {folder_path.name}", unit="tar"):
            tar_refs = future.result()
            templates[placeholder] = str(tar_path)

            # Replace tar path with placeholder in all refs
            for k, v in tar_refs.items():
                if isinstance(v, list) and len(v) >= 1:
                    v[0] = f"{{{{{placeholder}}}}}"

            all_refs.update(tar_refs)

    return all_refs, templates


# ---------------- Driver / JSON output ---------------- #
def kerchunk_general(input_files, out_dir=None, max_workers=8, num_workers_zip=32):
    """Process tar, zip files, or webdataset folders and write JSON outputs."""
    for file_path in input_files:
        file_path = Path(file_path).resolve()
        out_dir_resolved = Path(out_dir) if out_dir else file_path.parent
        out_dir_resolved.mkdir(parents=True, exist_ok=True)
        all_refs = {}
        templates = {}
        is_webdataset = False

        if file_path.is_dir():
            # Process as webdataset (folder containing .tar files)
            all_refs, templates = process_webdataset(file_path, max_workers=max_workers)
            is_webdataset = True
            output_name = file_path.name

        elif file_path.suffix == ".tar":
            all_refs = process_tar(file_path, max_workers=max_workers)
            output_name = file_path.name
            # Consolidate and create single template
            all_refs = consolidate(all_refs)
            refs = all_refs["refs"]
            placeholder_gen = generate_placeholder_names()
            placeholder = next(placeholder_gen)
            templates = {placeholder: str(file_path)}
            # Replace paths with placeholder
            for k, v in refs.items():
                if isinstance(v, list) and v:
                    v[0] = f"{{{{{placeholder}}}}}"
            all_refs = refs

        elif file_path.suffix == ".zip":
            all_refs = process_zip(file_path, num_workers=num_workers_zip)
            output_name = file_path.name
            # Consolidate and create single template
            all_refs = consolidate(all_refs)
            refs = all_refs["refs"]
            placeholder_gen = generate_placeholder_names()
            placeholder = next(placeholder_gen)
            templates = {placeholder: str(file_path)}
            # Replace paths with placeholder
            for k, v in refs.items():
                if isinstance(v, list) and v:
                    v[0] = f"{{{{{placeholder}}}}}"
            all_refs = refs

        else:
            print(f"Skipping unknown file type: {file_path}")
            continue

        # Add .zgroup for each .zarr folder
        zarr_folders = set()
        for k in list(all_refs.keys()):
            parts = k.split("/")
            for i, p in enumerate(parts):
                if p.endswith(".zarr"):
                    # Build path up to and including .zarr
                    zarr_path = "/".join(parts[:i+1])
                    zarr_folders.add(zarr_path)
                    break
        for folder in zarr_folders:
            zgroup_key = folder + "/.zgroup"
            if zgroup_key not in all_refs:
                all_refs[zgroup_key] = {"zarr_format": 2}

        # Add zarr group marker at the root level
        all_refs[".zgroup"] = {"zarr_format": 2}

        out_json = out_dir_resolved / f"{output_name}.json"
        # Store template paths relative to the JSON's own directory so the
        # dataset is portable and not tied to an absolute path on one machine.
        # See https://github.com/fsspec/kerchunk/issues/348
        relative_templates = {
            k: os.path.relpath(v, out_json.parent) if Path(v).is_absolute() else v
            for k, v in templates.items()
        }
        out_json.write_text(json.dumps({
            "version": 1,
            "templates": relative_templates,
            "refs": all_refs
        }, indent=2))
        print(f"Kerchunk JSON written to {out_json}")
        if is_webdataset:
            print(f"  Processed {len(templates)} tar files")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Kerchunk JSON reference files for .tar, .zarr.zip, or webdataset folders.",
        epilog="""
Examples:
  # Single .tar (e.g. GT tar):
  python generate_kerchunk.py -o $SCRATCH/data/GTs_Sentinel $SCRATCH/data/GTs_Sentinel/2025.tar

  # Webdataset folder (folder of .tar files, e.g. Sentinel-2 year):
  python generate_kerchunk.py -o $STORE/data/satellite/sentinel2/raw/CH $STORE/data/satellite/sentinel2/raw/CH/2025

  # Single .zarr.zip (e.g. Landsat):
  python generate_kerchunk.py -o $STORE/data/satellite/landsat/raw/CH/89 $STORE/data/satellite/landsat/raw/CH/89/2025.zarr.zip

  # Multiple inputs at once:
  python generate_kerchunk.py -o $STORE/data/satellite/landsat/raw/CH/89 \\
      $STORE/data/satellite/landsat/raw/CH/89/2023.zarr.zip \\
      $STORE/data/satellite/landsat/raw/CH/89/2024.zarr.zip
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Input files or folders (.tar, .zarr.zip, or webdataset folder)")
    parser.add_argument("-o", "--out_dir", default=None, help="Output directory for Kerchunk JSON files (default: same directory as input)")
    parser.add_argument("--max_workers", type=int, default=8, help="Parallel workers across tar files (default: 8)")
    parser.add_argument("--num_workers_zip", type=int, default=32, help="Parallel workers within a zip file (default: 32)")
    args = parser.parse_args()

    kerchunk_general(args.inputs, out_dir=args.out_dir,
                     max_workers=args.max_workers, num_workers_zip=args.num_workers_zip)
