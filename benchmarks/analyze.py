"""Aggregate independent repetitions, preserve run-level variation, and export figures."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(os.environ.get('INFERENCEOPS_RESULTS_DIR', ROOT/'results/generated'))
MODES=['immediate','fixed','dynamic']
PROFILES=['low','moderate','saturation','burst']
COLORS=['#3569a5','#bd762d','#36836c']


def analyze():
    runs=[]
    for path in sorted(DEST.glob('*-r[0-9].json')):
        report=json.loads(path.read_text())
        s=report['summary']
        row={'run':path.stem,'policy':report['model']['mode'],'profile':report['profile'],
             'variant':'sensitivity' if 'sensitivity' in path.stem else 'primary','repeat':report['repeat'],
             **{k:v for k,v in s.items() if not isinstance(v,dict)},
             'client_p50_ms':s['successful_client_ms']['p50'],'client_p95_ms':s['successful_client_ms']['p95'],
             'all_client_p95_ms':s['all_sent_client_ms']['p95'],
             'queue_p95_ms':s['queue_wait_ms']['p95'],'accumulation_p95_ms':s['accumulation_ms']['p95'],
             'rejection_fraction':s['rejected']/s['offered'],'timeout_fraction':s['timed_out']/s['offered'],
             'cpu_cores':report['container_resources']['cpu_cores_used'],
             'throttled_seconds':report['container_resources']['throttled_seconds']}
        runs.append(row)
    if not runs:
        with (DEST/'runs.csv').open() as stream:
            for raw in csv.DictReader(stream):
                runs.append({k: (v if k in {'run','policy','profile','variant'} else int(float(v)) if k in {'repeat','offered','sent','missed','admitted','completed','rejected','timed_out','other_errors','deadline_compliant','prediction_mismatches','duplicate_request_ids','active_after_drain','queue_after_drain'} else float(v)) for k,v in raw.items()})
    with (DEST/'runs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(runs[0]),lineterminator="\n");writer.writeheader();writer.writerows(runs)
    keys=['success_rps','deadline_compliant_rps','client_p50_ms','client_p95_ms','all_client_p95_ms',
          'completed_fraction_of_offered','deadline_fraction_of_offered','rejection_fraction','timeout_fraction',
          'mean_batch_size','queue_p95_ms','accumulation_p95_ms','cpu_cores','throttled_seconds']
    aggregate=[]
    for variant in ['primary','sensitivity']:
        for profile in PROFILES:
            for mode in MODES:
                selected=[r for r in runs if r['variant']==variant and r['profile']==profile and r['policy']==mode]
                if not selected:
                    continue
                result={'variant':variant,'profile':profile,'policy':mode,'repeats':len(selected)}
                for key in keys:
                    values=[r[key] for r in selected]
                    result[key+'_mean']=float(np.mean(values))
                    result[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0
                aggregate.append(result)
    with (DEST/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]),lineterminator="\n");writer.writeheader();writer.writerows(aggregate)
    (DEST/'summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def figure(filename, metrics):
        fig,axes=plt.subplots(1,len(metrics),figsize=(6*len(metrics),4.5),squeeze=False)
        for axis,(key,title,ylabel) in zip(axes[0],metrics):
            x=np.arange(len(PROFILES));width=.24
            for i,mode in enumerate(MODES):
                selected=[next(r for r in aggregate if r['variant']=='primary' and r['profile']==p and r['policy']==mode) for p in PROFILES]
                axis.bar(x+(i-1)*width,[r[key+'_mean'] for r in selected],width,
                         yerr=[r[key+'_std'] for r in selected],capsize=3,color=COLORS[i],label=mode)
            axis.set_xticks(x,PROFILES);axis.set_title(title);axis.set_ylabel(ylabel);axis.set_ylim(bottom=0)
            axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
        axes[0][0].legend(frameon=False)
        fig.suptitle('ResNet-18 · Linux ARM64 · 2 CPU limit · 3 repeats (mean ± sample SD)',fontsize=12)
        fig.tight_layout();fig.savefig(DEST/(filename+'.png'),dpi=170);fig.savefig(DEST/(filename+'.pdf'));plt.close(fig)
    figure('performance',[('success_rps','Successful throughput','requests/s'),('deadline_compliant_rps','Completed within 100 ms','requests/s'),('client_p95_ms','Successful-request p95','ms')])
    figure('fairness',[('completed_fraction_of_offered','Completed / offered','fraction'),('deadline_fraction_of_offered','Within 100 ms / offered','fraction'),('rejection_fraction','Rejected / offered','fraction')])
    figure('timing',[('queue_p95_ms','Total prepared-queue wait p95','ms'),('accumulation_p95_ms','Intentional accumulation p95','ms'),('mean_batch_size','Mean executed batch size','images / native call')])
    totals={k:sum(r[k] for r in runs) for k in ['offered','sent','missed','completed','rejected','timed_out','other_errors','prediction_mismatches','duplicate_request_ids']}
    (DEST/'accounting.json').write_text(json.dumps(totals,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    analyze()
