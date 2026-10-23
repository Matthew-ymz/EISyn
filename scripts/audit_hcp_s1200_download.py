#!/usr/bin/env python3
"""Read-only, full-file HCP download and numerical audit; no model fitting."""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/hcp_s1200_rest1_task_lr_schaefer500_1000_timeseries"
OUT = ROOT / "docs/log/hcp_s1200_download_audit_20261022.json"
LENGTHS = dict(REST1_LR=1200, EMOTION=176, GAMBLING=253, LANGUAGE=316,
               MOTOR=284, RELATIONAL=232, SOCIAL=274, WM=405)
FOLDERS = dict(REST1_LR="rest_minimal_status", LANGUAGE="task_status",
               WM="wm_status", MOTOR="motor_status", EMOTION="emotion_status",
               GAMBLING="gambling_status", RELATIONAL="relational_status",
               SOCIAL="social_status")
SD_TOL = 1e-8  # Native saved-signal units; diagnostic near-constant threshold.
Z_THRESHOLD = 8.0  # Descriptive flag, never an automatic imaging exclusion.


def array_digest(array):
    """Exactly the producer's dtype + shape + C-order byte digest."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def fields(state):
    prefix = "" if state == "REST1_LR" else state + "_LR_"
    return [f"{prefix}Schaefer{n}_{v}" for n in (500, 1000)
            for v in ("NoGSR", "GSR")]


def quantiles(values):
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def provenance():
    records, summaries = {}, {}
    with zipfile.ZipFile(DATA / "metadata/nas_publication_records.zip") as z:
        for state, folder in FOLDERS.items():
            status, frames, tr, settings, configs = (Counter() for _ in range(5))
            failures, identity_issues = [], []
            for name in z.namelist():
                if not (name.startswith(folder + "/") and name.endswith(".json")):
                    continue
                d = json.loads(z.read(name))
                sid = str(d["subject"])
                status[d.get("status")] += 1
                if d.get("status") != "complete":
                    failures.append(dict(subject=sid, status=d.get("status"),
                                         error=str(d.get("error", "")).splitlines()[-1:]))
                    continue
                diag = d["diagnostics"]
                frames[str(diag["frames"])] += 1
                tr[str(diag.get("tr_s"))] += 1
                cfg = d.get("configuration", {})
                configs[json.dumps(cfg, sort_keys=True)] += 1
                fsf = d.get("fsf", {})
                settings[json.dumps({k: fsf.get(k) for k in
                         ("settings", "stage_flags", "analysis")}, sort_keys=True)] += 1
                run = d.get("run") if state == "REST1_LR" else d.get("input", {}).get("run")
                expected_run = "rfMRI_REST1_LR" if state == "REST1_LR" else f"tfMRI_{state}_LR"
                if Path(name).stem != sid or run != expected_run:
                    identity_issues.append(dict(record=name, subject=sid, run=run))
                if state == "REST1_LR" and d.get("active_rest_version") != "S1200_minimal_no_FIX":
                    identity_issues.append(dict(record=name, active_rest_version=d.get("active_rest_version")))
                for key, digest in diag["array_sha256"].items():
                    pair = sid, key
                    if pair in records:
                        raise ValueError(f"Duplicate provenance: {pair}")
                    records[pair] = dict(sha256=digest, frames=diag["frames"], record=name)
            summaries[state] = dict(status_counts=dict(status), frames=dict(frames),
                tr_s=dict(tr), failures=failures, identity_issues=identity_issues,
                configurations=[dict(value=json.loads(k), count=v) for k, v in configs.items()],
                preprocessing=[dict(value=json.loads(k), count=v) for k, v in settings.items()])
        rest_state = json.loads(z.read("rest_minimal_import_state.json"))
    return records, summaries, rest_state


def matrix_audit(a, expected_shape):
    row = dict(shape=list(a.shape), dtype=str(a.dtype),
               expected_shape=list(expected_shape), shape_ok=a.shape == expected_shape,
               nonfinite_count=int(np.count_nonzero(~np.isfinite(a))))
    if a.ndim != 2 or row["nonfinite_count"] or min(a.shape) < 2:
        row["numerically_valid"] = False
        return row
    x = np.array(a, dtype=np.float64)
    mean = x.mean(axis=0)
    row.update(value_min=float(x.min()), value_max=float(x.max()),
               roi_mean_median=float(np.median(mean)),
               zero_frames=int(np.count_nonzero(np.all(a == 0, axis=1))))
    x -= mean
    energy = np.sum(x*x, axis=0)
    sd = np.sqrt(energy / len(x))
    row.update(constant_rois=int(np.count_nonzero(sd == 0)),
               near_constant_rois=int(np.count_nonzero((sd > 0) & (sd <= SD_TOL))),
               roi_sd=quantiles(sd), parcel_mean_signal_sd=float(x.mean(axis=1).std()))
    delta = np.diff(x, axis=0)
    row["identical_consecutive_frames"] = int(np.count_nonzero(np.all(delta == 0, axis=1)))
    if np.all(sd > SD_TOL):
        den = np.sqrt((energy-x[-1]**2)*(energy-x[0]**2))
        lag = np.sum(x[:-1]*x[1:], axis=0) / den
        row["roi_lag1_median"] = float(np.median(lag))
        z = x / sd
        row["max_abs_within_roi_z"] = float(np.max(np.abs(z)))
        row["abs_z_above_8_count"] = int(np.count_nonzero(np.abs(z) > Z_THRESHOLD))
        dvars = np.sqrt(np.mean((delta/sd)**2, axis=1))
        row["standardized_dvars"] = dict(median=float(np.median(dvars)),
            p95=float(np.quantile(dvars, .95)), maximum=float(dvars.max()))
    row["numerically_valid"] = bool(row["dtype"] == "float32" and
        not row["constant_rois"] and not row["near_constant_rois"] and
        not row["zero_frames"] and not row["identical_consecutive_frames"])
    return row


def main():
    start = time.monotonic()
    manifest_path = DATA / "metadata/nas_publish_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    source = {str(r["subject"]): r for r in manifest["subjects"]}
    if len(source) != len(manifest["subjects"]):
        raise ValueError("Duplicate subjects in source manifest")
    p_records, p_summary, rest_state = provenance()
    files = sorted(DATA.glob("*.mat"))
    rows, array_seen, duplicate_arrays = [], {}, []
    for i, path in enumerate(files):
        before = path.stat()
        raw = path.read_bytes()
        sid = path.stem
        ref = source.get(sid, {})
        digest = hashlib.sha256(raw).hexdigest()
        row = dict(subject=sid, filename=path.name, bytes=len(raw), sha256=digest,
            filename_ok=bool(re.fullmatch(r"\d{6}", sid)),
            bytes_match=len(raw) == ref.get("bytes"), sha256_match=digest == ref.get("sha256"),
            matrices={}, missing_states=[], partial_states=[])
        try:
            arrays = {k: v for k, v in loadmat(io.BytesIO(raw),
                verify_compressed_data_integrity=True).items() if not k.startswith("__")}
            row["fields_match"] = sorted(arrays) == sorted(ref.get("fields", []))
            expected_fields = set()
            for state, t in LENGTHS.items():
                keys = fields(state)
                expected_fields.update(keys)
                existing = [k for k in keys if k in arrays]
                if not existing:
                    row["missing_states"].append(state)
                elif len(existing) != 4:
                    row["partial_states"].append(state)
                for key in existing:
                    a = arrays[key]
                    n = 1000 if "Schaefer1000" in key else 500
                    q = matrix_audit(a, (t, n))
                    q.update(state=state, resolution=n,
                        variant="NoGSR" if key.endswith("NoGSR") else "GSR")
                    q["array_sha256"] = array_digest(a)
                    pr = p_records.get((sid, key), {})
                    q["provenance_hash_match"] = q["array_sha256"] == pr.get("sha256")
                    q["provenance_frame_match"] = a.shape[0] == pr.get("frames")
                    prior = array_seen.get(q["array_sha256"])
                    if prior is not None:
                        duplicate_arrays.append(dict(first=prior, second=[sid, key]))
                    else:
                        array_seen[q["array_sha256"]] = [sid, key]
                    row["matrices"][key] = q
            row["unexpected_fields"] = sorted(set(arrays)-expected_fields)
        except Exception as exc:
            row["read_error"] = f"{type(exc).__name__}: {exc}"
        after = path.stat()
        row["unchanged_during_read"] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        rows.append(row)
        if (i+1) % 100 == 0 or i+1 == len(files):
            print(json.dumps(dict(checked=i+1, total=len(files), elapsed_s=round(time.monotonic()-start, 1))), flush=True)
    actual = {r["subject"] for r in rows}
    all_mats = [m for r in rows for m in r["matrices"].values()]
    cohorts, coverage, anomalies = {}, {}, []
    for state, t in LENGTHS.items():
        present, usable = [], []
        frame_counts = Counter()
        for r in rows:
            ms = [m for m in r["matrices"].values() if m["state"] == state]
            if ms:
                present.append(r["subject"])
                frame_counts[str(ms[0]["shape"][0])] += 1
            good = len(ms) == 4 and all(m["shape_ok"] and m["numerically_valid"] and
                m["provenance_hash_match"] and m["provenance_frame_match"] for m in ms)
            if good and all(r.get(k, False) for k in
                    ("bytes_match", "sha256_match", "fields_match", "unchanged_during_read")):
                usable.append(r["subject"])
            if ms and any(not m["shape_ok"] or not m["numerically_valid"] for m in ms):
                anomalies.append(dict(subject=r["subject"], state=state,
                    expected_frames=t, actual_shapes=sorted({tuple(m["shape"]) for m in ms}),
                    numeric_issues=[k for k, m in r["matrices"].items()
                                    if m["state"] == state and not m["numerically_valid"]]))
        coverage[state] = dict(expected_frames=t, present=len(present), standard_valid=len(usable),
            missing=len(actual-set(present)), frame_counts=dict(frame_counts),
            manifest_count=manifest["condition_counts"][state])
        cohorts[state] = dict(present=present, standard_valid=usable, missing=sorted(actual-set(present)))
    for state in LENGTHS:
        pair = sorted(set(cohorts["REST1_LR"]["standard_valid"]) & set(cohorts[state]["standard_valid"]))
        coverage[state]["paired_standard_rest"] = len(pair)
        cohorts[state]["paired_standard_rest"] = pair
    all_standard = sorted(set.intersection(*(set(c["standard_valid"]) for c in cohorts.values())))
    all_present = sorted(set.intersection(*(set(c["present"]) for c in cohorts.values())))
    file_digests = defaultdict(list)
    for r in rows:
        file_digests[r["sha256"]].append(r["subject"])
    summary = dict(files=len(files), unique_subjects=len(actual), total_bytes=sum(r["bytes"] for r in rows),
        matrices=len(all_mats), read_errors=[r["subject"] for r in rows if "read_error" in r],
        file_integrity_failures=[r["subject"] for r in rows if not all(r.get(k, False) for k in
            ("filename_ok", "bytes_match", "sha256_match", "fields_match", "unchanged_during_read"))],
        missing_manifest_subjects=sorted(set(source)-actual), extra_subjects=sorted(actual-set(source)),
        partial_state_files=[r["subject"] for r in rows if r["partial_states"]],
        array_hash_mismatches=sum(not m["provenance_hash_match"] for m in all_mats),
        provenance_frame_mismatches=sum(not m["provenance_frame_match"] for m in all_mats),
        nonfinite_values=sum(m["nonfinite_count"] for m in all_mats),
        constant_rois=sum(m.get("constant_rois", 0) for m in all_mats),
        near_constant_rois=sum(m.get("near_constant_rois", 0) for m in all_mats),
        zero_frames=sum(m.get("zero_frames", 0) for m in all_mats),
        identical_consecutive_frames=sum(m.get("identical_consecutive_frames", 0) for m in all_mats),
        dtype_counts=dict(Counter(m["dtype"] for m in all_mats)),
        duplicate_file_groups=[v for v in file_digests.values() if len(v)>1],
        duplicate_arrays=duplicate_arrays, coverage=coverage, anomalous_runs=anomalies,
        all_eight_present=len(all_present), all_eight_standard_valid=len(all_standard),
        abs_z_above_8_count=sum(m.get("abs_z_above_8_count", 0) for m in all_mats))
    result = dict(checked_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="Full local files, producer checksums, numeric diagnostics; no imaging motion QC or method changes",
        data_root=str(DATA), elapsed_s=time.monotonic()-start,
        thresholds=dict(near_constant_sd_native=SD_TOL, descriptive_abs_z=Z_THRESHOLD,
            standardized_dvars="sqrt(mean_ROI((diff_time(x)/population_sd_time(x))**2)); not head-motion FD"),
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        source_publication_state=json.loads((DATA / "metadata/nas_publish_state.json").read_text()),
        provenance_summary=p_summary, rest_import_state=rest_state,
        summary=summary, cohorts=cohorts, all_eight_standard_subjects=all_standard,
        all_eight_present_subjects=all_present, files=rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
