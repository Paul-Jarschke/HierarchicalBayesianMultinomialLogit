import pathlib, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parent
EXP  = ROOT / "hbmnl_mixture_experiments"
DRY  = "--dry-run" in sys.argv
SAMPLERS = {"NUTS", "HMC"}
EMPTY_NB = '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}\n'

def act(msg): print(("DRY  " if DRY else "DO   ") + msg)

# 1. Swap <sampler>/results/<run>  ->  <sampler>/<run>/results
for results_dir in list(EXP.rglob("results")):
    sampler_dir = results_dir.parent
    if sampler_dir.name not in SAMPLERS:        # skip the NEW results dirs
        continue
    for run_dir in [p for p in results_dir.iterdir() if p.is_dir()]:
        new_run     = sampler_dir / run_dir.name
        new_results = new_run / "results"
        act(f"mkdir   {new_results.relative_to(ROOT)}")
        if not DRY: new_results.mkdir(parents=True, exist_ok=True)
        for item in run_dir.iterdir():
            if item.name == "analysis.ipynb":
                act(f"delete  {item.relative_to(ROOT)}")
                if not DRY: item.unlink()
                continue
            act(f"move    {item.name} -> {new_results.relative_to(ROOT)}")
            if not DRY: shutil.move(str(item), str(new_results / item.name))
        nb = new_run / "liesel.ipynb"
        act(f"create  {nb.relative_to(ROOT)}")
        if not DRY: nb.write_text(EMPTY_NB, encoding="utf-8")
        act(f"rmdir   {run_dir.relative_to(ROOT)}")
        if not DRY: run_dir.rmdir()
    if not DRY and not any(results_dir.iterdir()):
        act(f"rmdir   {results_dir.relative_to(ROOT)}")
        results_dir.rmdir()

# 2. Clean up stray results/ at sampler level that contain only loose files
#    (left over when the original results/<run>/ had files directly in results/)
for results_dir in list(EXP.rglob("results")):
    sampler_dir = results_dir.parent
    if sampler_dir.name not in SAMPLERS:
        continue
    loose_files = [p for p in results_dir.iterdir() if p.is_file()]
    if not loose_files:
        continue
    # Find the sibling run folder (the one child of sampler_dir that is a dir and not named "results")
    siblings = [p for p in sampler_dir.iterdir() if p.is_dir() and p.name != "results"]
    if len(siblings) != 1:
        act(f"SKIP (ambiguous siblings)  {results_dir.relative_to(ROOT)}")
        continue
    target_results = siblings[0] / "results"
    act(f"mkdir   {target_results.relative_to(ROOT)}  (if missing)")
    if not DRY: target_results.mkdir(exist_ok=True)
    for f in loose_files:
        act(f"move    {f.name} -> {target_results.relative_to(ROOT)}")
        if not DRY: shutil.move(str(f), str(target_results / f.name))
    act(f"rmdir   {results_dir.relative_to(ROOT)}")
    if not DRY: results_dir.rmdir()

# 3. Create an empty bayesm/ beside NUTS and HMC in every <k>_comp
for comp_dir in EXP.glob("*/*_comp"):
    bdir = comp_dir / "bayesm"
    act(f"mkdir   {bdir.relative_to(ROOT)}  (+.gitkeep)")
    if not DRY:
        bdir.mkdir(exist_ok=True)
        (bdir / ".gitkeep").touch()   # git can't track an empty dir
