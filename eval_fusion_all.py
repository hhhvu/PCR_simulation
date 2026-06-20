"""Evaluate the locally-available fusion checkpoints on all datasets through one pipeline.
Writes evalcmp_<model>_<dataset>_pred_df.csv (val+test combined). Self-contained."""
import os, sys
from os.path import dirname, realpath
import numpy as np, pandas as pd, torch
from torch import nn
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, dirname(realpath(__file__)))
from src.pcr_dataset import ImageSequenceGeneDataModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    ('kibnmcmb',                   'model_weights/kibnmcmb_best.ckpt'),
    ('10_27_fusion_vit_delta64',   'model_weights/10_27_fusion_model_vit_delta64'),
    ('fusion_vit_delta64',         'model_weights/fusion_vit_delta64.pth'),
    ('10_31_ensemble_vit_delta64', 'model_weights/10_31_ensemble_model_vit_delta64new.pth'),
]

# (name, curve_dict, target_df, img_dir, external)
DATASETS = [
    ('groundtruth', 'data/new_groundtruth_df_curve_dict_fn_no_invalid.pkl',
     'data/new_groundtruth_df_target_data_no_invalid.csv',
     '/data/sselvan/PCR/curve_imgs_new_cleaner_no_invalid', False),
    ('new_retest_1', 'data/new_retest_curve_dict_1.pkl', 'data/new_retest_df_target_data_1.csv',
     '/data/sselvan/PCR/curve_imgs_retest', True),
    ('new_retest', 'data/new_retest_curve_dict.pkl', 'data/new_retest_df_target_data.csv',
     '/data/sselvan/PCR/curve_imgs_retest', True),
    ('chip60', 'data/chip60_curve_dict.pkl', 'data/chip60_target_data.csv',
     '/data/sselvan/PCR/curve_imgs_chip60', True),
    ('karlen', 'data/karlen_curve_dict.pkl', 'data/karlen_target_data.csv',
     '/data/sselvan/PCR/curve_imgs_karlen', True),
    ('known', 'data/known_curve_dict.pkl', 'data/known_target_data.csv',
     '/data/sselvan/PCR/curve_imgs_known', True),
]


class FusionEval(nn.Module):
    """Generic ViT+LSTM fusion model with 3 heads, reconstructed to match a checkpoint.
    delta feature = LSTM-of-first-differences when fc.0 in == latent*3+genes, else broadcast max-min."""

    def __init__(self, sd, hp):
        super().__init__()
        latent, hidden, nl = hp['latent_dim'], hp['hidden_size'], hp['num_layers']
        genes, self.delta, input_size = hp['genes'], hp['delta'], hp['input_size']
        self.vit = models.vit_b_32(weights=None)
        self.vit_classifier = nn.Linear(self.vit.num_classes, latent)
        self.lstm = nn.LSTM(input_size, hidden, nl, batch_first=True)
        self.lstm_fc = nn.Linear(hidden, latent)
        fc0_in = sd['fc.0.weight'].shape[1]
        self.mode = 'lstm_delta' if fc0_in == latent * 3 + genes else 'maxmin'
        if self.mode == 'lstm_delta':
            self.lstm_delta = nn.LSTM(input_size, hidden, nl, batch_first=True)
            self.lstm_fc_delta = nn.Linear(hidden, latent)
        self.fc = nn.Sequential(
            nn.Linear(fc0_in, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        n_heads = sum(1 for k in sd if k.startswith('heads.') and k.endswith('.weight'))
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_heads)])

    def forward(self, image, sequence, genes):
        img = self.vit_classifier(self.vit(image))
        seq = self.lstm_fc(self.lstm(sequence)[0][:, -1, :])
        if genes.dim() == 3:
            genes = genes.squeeze(1)
        if self.mode == 'lstm_delta':
            dseq = sequence[:, 1:] - sequence[:, :-1]
            delta_feat = self.lstm_fc_delta(self.lstm_delta(dseq)[0][:, -1, :])
        else:
            dl = torch.max(sequence, dim=1)[0] - torch.min(sequence, dim=1)[0]
            delta_feat = dl.expand((-1, self.delta))
        out = self.fc(torch.cat((img, seq, genes, delta_feat), dim=1))
        return torch.stack([torch.sigmoid(h(out)) for h in self.heads], dim=-1).squeeze()


def load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict'], obj.get('hyper_parameters')
    return obj, None


def derive_hp(sd):
    latent = sd['lstm_fc.weight'].shape[0]
    fc0 = sd['fc.0.weight'].shape[1]
    genes = 6
    return dict(input_size=sd['lstm.weight_ih_l0'].shape[1],
                hidden_size=sd['lstm.weight_ih_l0'].shape[0] // 4,
                num_layers=len([k for k in sd if k.startswith('lstm.weight_ih_l')]),
                latent_dim=latent, genes=genes,
                delta=64 if fc0 == latent * 3 + genes else fc0 - latent * 2 - genes)


def build_models():
    out = {}
    for name, path in MODELS:
        sd, hp = load_state(path)
        if hp is None:
            hp = derive_hp(sd)
        m = FusionEval(sd, hp)
        m.load_state_dict(sd, strict=False)
        out[name] = m.to(DEVICE).eval()
        print(f'built {name} (mode={m.mode})')
    return out


def run(model, loader):
    pr, fp, fn, ids = [], [], [], []
    with torch.no_grad():
        for inp, _, cid in tqdm(loader, leave=False):
            inp = [x.to(DEVICE) for x in inp]
            o = model(*inp)
            if o.dim() == 1:
                o = o.unsqueeze(0)
            pr.append(o[:, 0].cpu().numpy()); fp.append(o[:, 1].cpu().numpy())
            fn.append(o[:, 2].cpu().numpy()); ids.append(np.asarray(cid))
    if not ids:
        return None
    return np.concatenate(ids), np.concatenate(pr), np.concatenate(fp), np.concatenate(fn)


def main():
    models_ = build_models()
    for ds, cd, tgt, img, ext in DATASETS:
        print(f'=== dataset {ds} (external={ext}) ===')
        dm = ImageSequenceGeneDataModule(curve_dict_path=cd, target_df_path=tgt, img_directory=img,
                                         batch_size=64, num_workers=8, igi_call=False,
                                         gen_preds=True, external=ext)
        dm.setup()
        loaders = [('val', dm.val_dataloader()), ('test', dm.test_dataloader())]
        for name, model in models_.items():
            frames = []
            for split, loader in loaders:
                r = run(model, loader)
                if r is None:
                    continue
                cidx, pr, fp, fn = r
                d = pd.DataFrame({'curve_idx': cidx, 'outputs': pr, 'igi_fp': fp, 'igi_fn': fn})
                d['split'] = split
                frames.append(d)
            out = pd.concat(frames, ignore_index=True)
            p = f'data/model_outputs/evalcmp_{name}_{ds}_pred_df.csv'
            out.to_csv(p, index=False)
            print(f'    {name}: saved {len(out)} -> {p}')


if __name__ == '__main__':
    main()
