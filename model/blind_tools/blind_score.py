"""Blind test step 3: scoring. CPU only, reads files only.

    python3 -I model/blind_tools/blind_score.py NAME [--dry-run] [--no-push] [--no-commit]

Pairs the newest predictions_*.json with the newest measured_*.json of NAME. Per row (K): ratio = measured / predicted; inside the factor when 0.5 <= ratio <= 2.
Summary: rows inside the factor, rms of ln(ratio) (and exp of it), median |ln ratio| (the F236 figure), worst ratio. A missing or failed measurement is a missing row
(counted, never a pass). Writes scores_<ts>.json and .md next to the inputs (never overwritten), commits and pushes. No pass mark is set (RULE.md)."""
import os, sys, json, math, time, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import blind_common as bc
def score(P, M):
    rows = []
    for K in P['ks']:
        pred = P['pred'][str(K)]['ms']; m = M['rows'].get(str(K), {}); meas = m.get('search_ms_median')
        if meas is None or pred <= 0: rows.append(dict(K=K, predicted_ms=pred, measured_ms=None, missing=m.get('missing', 'no successful run'))); continue
        ratio = meas / pred; rows.append(dict(K=K, predicted_ms=pred, measured_ms=meas, ratio=ratio, ln_ratio=math.log(ratio), inside_factor_2=0.5 <= ratio <= 2.0, full_clock=m.get('all_runs_full_clock')))
    have = [r for r in rows if r.get('measured_ms') is not None]; ln = [r['ln_ratio'] for r in have]
    summ = dict(rows=len(rows), rows_measured=len(have), rows_missing=len(rows) - len(have), inside_factor_2=sum(r['inside_factor_2'] for r in have),
                rms_ln_ratio=math.sqrt(sum(x * x for x in ln) / len(ln)) if ln else None, rms_factor=math.exp(math.sqrt(sum(x * x for x in ln) / len(ln))) if ln else None,
                median_abs_ln_ratio=statistics.median(abs(x) for x in ln) if ln else None, worst_ratio=max((max(r['ratio'], 1 / r['ratio']) for r in have), default=None))
    return rows, summ
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--no-push', action='store_true'); ap.add_argument('--no-commit', action='store_true'); a = ap.parse_args()
    out = os.path.join(bc.REPO, 'model', 'blind', ('dry_run_' if a.dry_run else '') + a.name); fs = sorted(os.listdir(out))
    pf = [f for f in fs if f.startswith('predictions_') and f.endswith('.json')]; mf = [f for f in fs if f.startswith('measured_') and f.endswith('.json')]
    if not pf or not mf: raise SystemExit('need predictions and measured files in ' + out)
    P = json.load(open(os.path.join(out, pf[-1]))); M = json.load(open(os.path.join(out, mf[-1])))
    if M['predictions_file'] != os.path.relpath(os.path.join(out, pf[-1]), bc.REPO): raise SystemExit('measured file was taken against a different predictions file')
    if P['label'] != M['label']: raise SystemExit('label mismatch')
    rows, s = score(P, M); ts = time.strftime('%Y%m%dT%H%M%S'); label = P['label']
    sj = os.path.join(out, 'scores_%s.json' % ts); sm_ = os.path.join(out, 'scores_%s.md' % ts)
    with open(sj, 'x') as f: json.dump(dict(label=label, name=a.name, predictions=pf[-1], measured=mf[-1], rows=rows, summary={k: (float('%.6g' % v) if isinstance(v, float) else v) for k, v in s.items()}), f, indent=1)
    with open(sm_, 'x') as f:
        f.write('# %s: scores for %s (%s against %s)\n\n' % (label, a.name, mf[-1], pf[-1]))
        if label == 'DRY RUN': f.write('**DRY RUN: tests the tools only; the dataset was measured before; not a blind result.**\n\n')
        f.write('Ratio = measured RTNN search compute / predicted launch; inside the factor when 0.5 to 2.\n\n| K | predicted ms | measured ms | ratio | inside factor 2 | all runs at full clock |\n|---|---|---|---|---|---|\n')
        for r in rows:
            f.write('| %d | %.4f | %s | %s | %s | %s |\n' % (r['K'], r['predicted_ms'], '-' if r['measured_ms'] is None else '%.4f' % r['measured_ms'], '-' if r['measured_ms'] is None else '%.3g' % r['ratio'], 'missing' if r['measured_ms'] is None else ('yes' if r['inside_factor_2'] else 'no'), '-' if r['measured_ms'] is None else r.get('full_clock')))
        f.write('\nRows %d, measured %d, missing %d. Inside the factor: %d. rms ln ratio %s (factor %s). Median |ln ratio| %s. Worst ratio %s.\n' % (s['rows'], s['rows_measured'], s['rows_missing'], s['inside_factor_2'], *(('%.3f' % s[k]) if s[k] is not None else '-' for k in ('rms_ln_ratio', 'rms_factor', 'median_abs_ln_ratio', 'worst_ratio'))))
    print(open(sm_).read())
    if not a.no_commit: print('COMMIT (scores):', bc.commit([sj, sm_], '%s: scores for %s' % (label, a.name), not a.no_push))
main()
