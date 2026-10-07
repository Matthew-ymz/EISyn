#!/usr/bin/env python3
"""Recompute every legacy A-C record with the new runner before accepting expansion."""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hcp_timeseries_abc_979 import (
    LABELS, NETWORKS, LN2, TOL, OLD_TASK, atomic_json, fit_series, load_yeo7_groups, sha,
)
from scripts.tune_hcp_task_evoked_xi_hierarchy import bh_adjust

OLD_REST = ROOT / "data/hcp_s1200_schaefer500_1000_yeo7_minimalpreproc_rest1_timeseries_57_brain"
OLD_RESULTS = ROOT / "results/hcp_schaefer1000_task_evoked_xi_57/full"
NEW_RESULTS = ROOT / "results/hcp_timeseries_abc_replication_979"
OUT = NEW_RESULTS / "rest_diagnostic/legacy_abc_gate.json"
EQUIVALENCE_BITS = 1e-8
EQUIVALENCE_FRACTION = 1e-9


def old_atoms(row):
    return {tuple(a["sources"]):float(a["value"]) for a in row["atoms"]}


def new_atoms(row):
    return {tuple(n["sources"]):float(n["syn_bits"]) for n in row["nodes"] if len(n["sources"])>=2}


def order_mass(atoms):
    mass = np.zeros(6)
    for key, value in atoms.items():
        mass[len(key)-2]+=value
    return mass


def errors(row, old):
    oa, na = old_atoms(old), new_atoms(row)
    keys=set(oa)|set(na)
    return dict(
        system_xi_bits=abs(row["system_xi_bits"]-old["system_xi"]),
        joint_ei_bits=abs(row["joint_ei_bits"]-old["joint_ei"]),
        scalar_ei_sum_bits=abs(sum(row["scalar_ei_bits"])-old["scalar_ei_sum"]),
        cross_network_xi_bits=abs(row["cross_network_xi_bits"]-old["cross_network_xi"]),
        module_ei_bits=max(abs(row["module_ei_bits"][name]-old["module_ei"][name]) for name in NETWORKS),
        within_network_xi_bits=max(abs(row["within_network_xi_bits"][name]-old["within_network_xi"][name]) for name in NETWORKS),
        network_attribution_bits=max(abs(row["network_contribution_bits"][i]-old["network_attribution"][name]) for i,name in enumerate(NETWORKS)),
        network_share_fraction=max(abs(row["network_percent"][i]/100-old["network_attribution"][name]/old["system_xi"]) for i,name in enumerate(NETWORKS)),
        selected_spt_atom_bits=max(abs(na.get(key,0)-oa.get(key,0)) for key in keys),
        order_mass_bits=float(np.max(abs(np.asarray(row["order_mass_bits"])-order_mass(oa)))),
        pca_explained_fraction=max(abs(row["pca_explained"][i]-old["pca_cumulative_explained_variance"][name]) for i,name in enumerate(NETWORKS)),
        heldout_skill_ratio=abs(row["heldout"]["skill_ratio"]-old["heldout_skill_ratio"]),
        changed_spt_source_sets=set(na)!=set(oa))


def main():
    references=[json.loads(line) for line in (OLD_RESULTS/"records.jsonl").read_text().splitlines() if line.strip()]
    reference={(r["subject"].removeprefix("sub-"),r["state"]):r for r in references if r["config_id"]=="k1_p3_a1"}
    subjects=sorted({key[0] for key in reference})
    states=("REST", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
    expected={(s,state) for s in subjects for state in states}
    if len(subjects)!=57 or set(reference)!=expected:
        raise ValueError("Legacy reference must be complete 57 x 8 grid")
    groups=load_yeo7_groups(LABELS,expected_parcels=1000)
    validation=[]
    # Refit original MAT inputs; using the new runner for all eight states is the
    # independent check against the old frozen numerical results.
    for subject in subjects:
        for state in states:
            if state=="REST":
                path=OLD_REST/f"sub-{subject}/sub-{subject}_hcp_s1200_rfMRI_REST1_LR_schaefer500-1000_yeo7.mat"
                raw=loadmat(path,variable_names=["Schaefer1000"])["Schaefer1000"].astype(float)
                fitting=raw
            else:
                path=OLD_TASK/f"sub-{subject}/{state}_LR.mat"
                data=loadmat(path,variable_names=["Schaefer1000_taskRetained","Schaefer1000_taskRegressed"])
                raw=data["Schaefer1000_taskRetained"].astype(float)
                fitting=raw-data["Schaefer1000_taskRegressed"].astype(float)
            row,_=fit_series(raw,fitting,groups,subject,state,"legacy-verification")
            e=errors(row,reference[subject,state])
            failed=[key for key,value in e.items() if key.endswith("_bits") and value>EQUIVALENCE_BITS]
            failed += [key for key,value in e.items() if key.endswith("_fraction") and value>EQUIVALENCE_FRACTION]
            if e["heldout_skill_ratio"]>EQUIVALENCE_FRACTION or e["changed_spt_source_sets"]:
                failed.append("heldout/topology")
            validation.append(dict(subject=subject,state=state,n_timepoints=len(raw),
                input_path=str(path.resolve()),input_sha256=sha(path),errors=e,passed=not failed,
                system_xi_bits=row["system_xi_bits"],network_percent=row["network_percent"],
                order_mass_bits=row["order_mass_bits"],selected_spt_atoms_bits={"+".join(k):v for k,v in new_atoms(row).items()},
                numerical_audit=row["numerical_audit"],failure_fields=failed))
        if len(validation)%80==0 or len(validation)==456:
            print(f"Legacy A-C verification {len(validation)}/456",flush=True)
    metric_names=validation[0]["errors"]
    maxima={key:max(r["errors"][key] for r in validation) for key in metric_names}
    maxima["changed_spt_source_sets"]=sum(r["errors"]["changed_spt_source_sets"] for r in validation)
    bystate={state:{key:max(r["errors"][key] for r in validation if r["state"]==state) for key in metric_names} for state in states}
    newrows={(r["subject"],r["variant"]):r for r in (json.loads(line) for line in (NEW_RESULTS/"records.jsonl").read_text().splitlines())}
    input_comparison=[]
    for state in ("REST","EMOTION","LANGUAGE","MOTOR","WM"):
        old=np.array([reference[s,state]["system_xi"] for s in subjects])
        new=np.array([newrows[s,state]["system_xi_bits"] for s in subjects])
        full_errors=[errors(newrows[s,state],reference[s,state]) for s in subjects]
        old_orders=np.array([order_mass(old_atoms(reference[s,state])) for s in subjects])
        new_orders=np.array([newrows[s,state]["order_mass_bits"] for s in subjects])
        old_network=np.array([[100*reference[s,state]["network_attribution"][n]/reference[s,state]["system_xi"] for n in NETWORKS] for s in subjects])
        new_network=np.array([newrows[s,state]["network_percent"] for s in subjects])
        input_comparison.append(dict(state=state,old_mean_xi_nats=float(old.mean()*LN2),new_mean_xi_nats=float(new.mean()*LN2),
            mean_shift_nats=float((new-old).mean()*LN2),mean_relative_shift=float((new.mean()-old.mean())/old.mean()),
            subject_xi_pearson=float(np.corrcoef(old,new)[0,1]),maximum_subject_shift_nats=float(np.max(abs(new-old))*LN2),
            maximum_subject_relative_shift=float(np.max(abs(new-old)/old)),
            mean_network_shift_percentage_points=(new_network-old_network).mean(0).tolist(),
            maximum_subject_network_shift_percentage_points=float(np.max(abs(new_network-old_network))),
            mean_order_shift_nats=((new_orders-old_orders)*LN2).mean(0).tolist(),
            old_mean_peak_order=int(np.argmax(old_orders.mean(0))+2),new_mean_peak_order=int(np.argmax(new_orders.mean(0))+2),
            changed_spt_source_sets_count=sum(e["changed_spt_source_sets"] for e in full_errors),
            exact_equivalence=all(e["system_xi_bits"]<=EQUIVALENCE_BITS and e["network_share_fraction"]<=EQUIVALENCE_FRACTION and e["order_mass_bits"]<=EQUIVALENCE_BITS for e in full_errors)))
    selected={(r["subject"],r["state"]):r for r in validation}
    rest=np.array([selected[s,"REST"]["system_xi_bits"] for s in subjects])
    statistics=[]
    for state in states[1:]:
        task=np.array([selected[s,state]["system_xi_bits"] for s in subjects])
        statistics.append(dict(state=state,p=float(wilcoxon(rest-task).pvalue),mean_rest_minus_task_nats=float((rest-task).mean()*LN2)))
    for row,q in zip(statistics,bh_adjust([r["p"] for r in statistics])):row["q"]=float(q)
    oldstats=json.loads((OLD_RESULTS/"k1_p3_a1/summary.json").read_text())["rest_system_tests"]
    stats_max_error=max(abs(r[field]-old[field]) for r,old in zip(statistics,oldstats) for field in ("p","q"))
    passed=all(r["passed"] for r in validation) and stats_max_error<EQUIVALENCE_FRACTION
    atomic_json(OUT,dict(status="passed" if passed else "failed", n_subjects=57,n_models=456,
        scope="All eight original states and every subject-level numerical output behind A-C, including SPT atoms/order masses and network attribution; no behavioral rerun", tolerance_bits=TOL,
        numerical_equivalence=dict(bits=EQUIVALENCE_BITS,fractions=EQUIVALENCE_FRACTION),
        maxima=maxima,maximum_panel_a_statistical_error=stats_max_error,by_state= bystate,
        legacy_panel_a_statistics=statistics,new_input_comparison=input_comparison,
        new_input_gate="not passed: processed REST differs; tasks close but not bitwise/metric-identical; three tasks unavailable",
        main_979_interpretation="Exploratory output retained; not yet an equivalent-input replication of legacy REST-task contrast", rows=validation))
    print(json.dumps(dict(status=passed,maxima=maxima,statistics_error=stats_max_error,new_input_comparison=input_comparison),indent=2))
    if not passed:raise RuntimeError("Legacy A-C equivalence failed; see audit before expansion")


if __name__=="__main__":main()
