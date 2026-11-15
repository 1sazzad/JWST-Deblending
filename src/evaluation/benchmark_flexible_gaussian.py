"""Phase 8.3: matched flexible Gaussian development benchmark.

Run only the new baseline on the five prespecified base scenes and four
morphology families. Compare against archived Phase 7.5 measurements.
Use --smoke for one scene; it does not write a CSV.
"""
import argparse
import csv
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from src.baselines.baseline_flexible_gaussian import fit_flexible_gaussian
from src.evaluation.benchmark_baseline import (
    create_configuration, SEPARATIONS, FLUX_RATIOS, REPETITIONS, BASE_SEED,
)
from src.simulation.simulate_blends import simulate_scene

ROOT = Path(__file__).resolve().parents[2]
SCENE_IDS = (1,16,46,81,96)
FAMILIES = {'gaussian':None,'sersic1':1.,'sersic2':2.,'sersic4':4.}
METHODS = ('fixed_gaussian','scarlet2','scarlet2_neural')
BANDS = ('f150w','f277w','f444w')
FIELDS = ['morphology','scene_id','seed','separation_px','flux_ratio_f277w','source','band',
          'true_flux','estimated_flux','signed_error','absolute_error','combined_signed_error','chi2',
          'sigma_x','sigma_y','rho','sigma_major','sigma_minor','angle_deg','background',
          'optimizer_success','optimizer_message','iterations','function_evaluations','valid_starts','converged_starts',
          'start_losses','runtime_seconds','status','failure_reason'] + [
              f'{m}_{k}' for m in METHODS for k in ('flux','absolute_error','chi2')]


def load_reference(family, cases):
    """Validate archived identifiers and metrics without rerunning any fit."""
    name = 'benchmark_scarlet2'+('' if family=='gaussian' else '_'+family)+'.csv'
    path = ROOT/'results'/name
    configs = {i:(sep,ratio,seed) for i,sep,ratio,seed in cases}
    expected = {(i,m,s,b) for i in configs for m in METHODS for s in (1,2) for b in BANDS}
    refs = {}
    with path.open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            key = int(row['scene_id']),row['method'],int(row['source']),row['band'].lower()
            if key not in expected:
                continue
            sep,ratio,seed = configs[key[0]]
            if (key in refs or row['status']!='success' or int(row['seed'])!=seed
                or float(row['separation_px'])!=sep or float(row['flux_ratio_f277w'])!=ratio
                or (family!='gaussian' and row.get('morphology')!=family)):
                raise ValueError(f'Invalid archived row: {path}: {key}')
            values = {k:float(row[k]) for k in ('true_flux','estimated_flux','absolute_error','chi2')}
            if (not all(np.isfinite(v) for v in values.values()) or values['true_flux']<=0
                or values['chi2']<0 or not np.isclose(values['absolute_error'],abs(values['estimated_flux']/values['true_flux']-1),rtol=1e-5,atol=1e-7)):
                raise ValueError(f'Invalid archived metrics: {key}')
            refs[key]=values
    if set(refs)!=expected:
        raise ValueError(f'Missing archived references: {path}: {expected-set(refs)}')
    return refs


def run_scene(family,n,case,refs):
    """Generate matched inputs, fit without truth, then evaluate all six fluxes."""
    i,sep,ratio,seed=case
    config=create_configuration(sep,ratio,seed)
    scene=simulate_scene(**config) if n is None else simulate_scene(**config,sersic_n=n)
    truth=scene['input_flux']
    bands=tuple(str(b).lower() for b in scene['bands'])
    if bands!=BANDS or truth.shape!=(2,3):
        raise ValueError('Unexpected simulator layout')
    for s in range(2):
        for b,band in enumerate(bands):
            for m in METHODS:
                if not np.isclose(refs[i,m,s+1,band]['true_flux'],truth[s,b],rtol=1e-8,atol=1e-8):
                    raise ValueError(f'Archive truth mismatch: {family}, {i}')
    start=perf_counter()
    status,reason='success',''
    try:
        flux,model,d=fit_flexible_gaussian(scene['observed'],scene['psfs'],scene['positions'],scene['noise_sigma'])
        if flux.shape!=truth.shape or model.shape!=scene['observed'].shape:
            raise ValueError('Unexpected output shape')
        signed=(flux-truth)/truth
        combined=(flux.sum(axis=0)-truth.sum(axis=0))/truth.sum(axis=0)
        chi=float(np.mean(((scene['observed']-model)/scene['noise_sigma'][:,None,None])**2))
        if not np.isfinite(signed).all() or not np.isfinite(chi):
            raise ValueError('Invalid evaluation metrics')
    except Exception as exc:
        status,reason='failed',f'{type(exc).__name__}: {exc}'
    elapsed=perf_counter()-start
    rows=[]
    for s in range(2):
        for b,band in enumerate(bands):
            row=dict(morphology=family,scene_id=i,seed=seed,separation_px=sep,flux_ratio_f277w=ratio,
                     source=s+1,band=band,true_flux=truth[s,b],runtime_seconds=elapsed,status=status,failure_reason=reason)
            for m in METHODS:
                ref=refs[i,m,s+1,band]
                row.update({f'{m}_flux':ref['estimated_flux'],f'{m}_absolute_error':ref['absolute_error'],f'{m}_chi2':ref['chi2']})
            if status=='success':
                row.update(estimated_flux=flux[s,b],signed_error=signed[s,b],absolute_error=abs(signed[s,b]),
                           combined_signed_error=combined[b],chi2=chi,background=d['background'][b],
                           **d['shapes'][s],**{k:d[k] for k in ('optimizer_success','optimizer_message','iterations',
                               'function_evaluations','valid_starts','converged_starts','start_losses')})
            rows.append(row)
    if status=='success':
        print(f'{family} | scene {i} | fixed={100*np.mean([r["fixed_gaussian_absolute_error"] for r in rows]):.2f}% '
              f'| flexible={100*np.abs(signed).mean():.2f}% | chi2={chi:.5f} '
              f'| converged={d["optimizer_success"]} | {elapsed:.2f}s',flush=True)
    else:
        print(f'{family} | scene {i} | FAILED: {reason}',flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    grid=list(product(SEPARATIONS,FLUX_RATIOS,range(REPETITIONS)))
    ids=SCENE_IDS[:1] if args.smoke else SCENE_IDS
    cases=[(i,*grid[i-1][:2],BASE_SEED+i) for i in ids]
    families={'gaussian':None} if args.smoke else FAMILIES
    refs={family:load_reference(family,cases) for family in families}
    planned=len(cases)*len(families)
    print(f'Planned new fits: {planned} | Expected measurements: {planned*6}',flush=True)
    rows=[]
    def run(writer=None,stream=None):
        for family,n in families.items():
            for case in cases:
                batch=run_scene(family,n,case,refs[family])
                rows.extend(batch)
                if writer is not None:
                    writer.writerows(batch)
                    stream.flush()
    if args.smoke:
        run()
    else:
        path=ROOT/'results'/'benchmark_flexible_gaussian_phase8_3.csv'
        with path.open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            writer.writeheader()
            run(writer,stream)
        print(f'Results saved: {path}')
    print('\n--- Matched successful-fit comparison ---')
    for family in families:
        group=[r for r in rows if r['morphology']==family and r['status']=='success']
        if not group:
            continue
        metrics={m:100*np.mean([r[f'{m}_absolute_error'] for r in group]) for m in METHODS}
        metrics['flexible_gaussian']=100*np.mean([r['absolute_error'] for r in group])
        print(f'{family}: '+', '.join(f'{m}={v:.2f}%' for m,v in metrics.items()))
    failed=sum(r['status']=='failed' for r in rows)//6
    converged=sum(r['status']=='success' and r['optimizer_success'] for r in rows)//6
    print(f'Recorded fits: {len(rows)//6}/{planned} | Failed fits: {failed} | Selected starts converged: {converged}')


if __name__=='__main__':
    main()
