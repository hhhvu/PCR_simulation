"""Build a clean MODEL_METRICS.md: 8 models x 6 datasets.
Metrics vs groundtruth_target @0.5, val+test combined, NaN-label rows dropped."""
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# dataset -> (display label, target csv)
DATASETS = [
    ('groundtruth',  'new_groundtruth_no_invalid (in-distribution test)', 'data/new_groundtruth_df_target_data_no_invalid.csv'),
    ('new_retest',   'new_retest',                                         'data/new_retest_df_target_data.csv'),
    ('new_retest_1', 'new_retest_1',                                       'data/new_retest_df_target_data_1.csv'),
    ('chip60',       'chip60 (external)',                                  'data/chip60_target_data.csv'),
    ('karlen',       'karlen (external, all-positive)',                    'data/karlen_target_data.csv'),
    ('known',        'known (external, dilution)',                         'data/known_target_data.csv'),
]

# fixed model display order
MODELS = ['fusiongene_large', 'SeqGeneModel_large', 'SeqModel_large', 'ImageModel_large',
          'kibnmcmb', '10_27_fusion_vit_delta64', 'fusion_vit_delta64', '10_31_ensemble_vit_delta64']

# fusiongene_large is now run live (evalcmp_*); the other three large models only have
# original-pipeline prediction CSVs (their checkpoints were never downloaded).
PIPELINE_PREFIX = {'SeqGeneModel_large': '3_20', 'SeqModel_large': '3_20', 'ImageModel_large': '3_20'}


def pred_path(model, ds):
    """Resolve the prediction CSV for (model, dataset), or None if absent."""
    if model in PIPELINE_PREFIX:
        ds_tok = 'retest' if ds == 'new_retest' else ds  # original pipeline names the old set 'retest'
        p = f"data/model_outputs/{PIPELINE_PREFIX[model]}_{model}_no_invalid_{ds_tok}_val_test_pred_df.csv"
    else:
        p = f"data/model_outputs/evalcmp_{model}_{ds}_pred_df.csv"
    return p if os.path.exists(p) else None


def metrics(y, s, thr=0.5):
    pred = (s > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, s) if y.nunique() > 1 else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return auc, accuracy_score(y, pred), fnr, fpr


def f(x):
    return f'{x:.4f}' if isinstance(x, float) else '—'


tcache = {}
out = ['# Model metrics', '',
       'AUROC / Accuracy / FNR / FPR vs `groundtruth_target` (threshold 0.5), val+test combined.', '']

for ds, label, tgt_file in DATASETS:
    if ds not in tcache:
        tcache[ds] = pd.read_csv(tgt_file)[['curve_idx', 'groundtruth_target']]
    tgt = tcache[ds]
    out += [f'## {label}', '', '| Model | N | AUROC | Accuracy | FNR | FPR |',
            '|-------|--:|------:|---------:|----:|----:|']
    for model in MODELS:
        p = pred_path(model, ds)
        if p is None:
            out.append(f'| {model} | — | — | — | — | — |')
            continue
        d = tgt.merge(pd.read_csv(p)[['curve_idx', 'outputs']], on='curve_idx').dropna(subset=['groundtruth_target'])
        if len(d) < 5:
            out.append(f'| {model} | — | — | — | — | — |')
            continue
        y = d['groundtruth_target'].astype(int)
        auc, acc, fnr, fpr = metrics(y, d['outputs'].astype(float))
        out.append(f'| {model} | {len(d)} | {f(auc)} | {f(acc)} | {f(fnr)} | {f(fpr)} |')
    out.append('')

with open('MODEL_METRICS.md', 'w') as fh:
    fh.write('\n'.join(out) + '\n')
print('\n'.join(out))
