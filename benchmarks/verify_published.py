"""Verify all published numeric claims against the sanitized run table."""
import csv
import json
from pathlib import Path
import statistics


def main():
    root=Path(__file__).resolve().parents[1]/'results/published'
    rows=list(csv.DictReader((root/'runs.csv').open()))
    assert len(rows)==42
    totals={key:sum(int(row[key]) for row in rows) for key in ['offered','sent','missed','completed','rejected','timed_out','other_errors','prediction_mismatches','duplicate_request_ids']}
    assert totals==json.loads((root/'accounting.json').read_text())
    for row in rows:
        n=lambda key:int(row[key])
        assert n('offered')==n('sent')+n('missed')
        assert n('sent')==n('completed')+n('rejected')+n('timed_out')+n('other_errors')
        assert n('admitted')==n('completed')  # No post-admission rejection in these recorded runs.
        assert n('active_after_drain')==n('queue_after_drain')==0
    for summary in json.loads((root/'summary.json').read_text()):
        selected=[r for r in rows if all(r[key]==summary[key] for key in ['variant','profile','policy'])]
        assert len(selected)==summary['repeats']==3
        for metric in ['success_rps','deadline_compliant_rps','mean_batch_size','client_p95_ms','rejection_fraction']:
            values=[float(r[metric]) for r in selected]
            assert abs(statistics.mean(values)-summary[metric+'_mean'])<1e-9
            assert abs(statistics.stdev(values)-summary[metric+'_std'])<1e-9
    print('PASS: 42 runs, complete request accounting and published mean/SD claims')


if __name__=='__main__':
    main()
