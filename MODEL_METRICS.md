# Model metrics

AUROC / Accuracy / FNR / FPR vs `groundtruth_target` (threshold 0.5), val+test combined.

## new_groundtruth_no_invalid (in-distribution test)

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 53466 | 0.9997 | 0.9967 | 0.0024 | 0.0111 |
| SeqGeneModel_large | 53466 | 0.9998 | 0.9970 | 0.0024 | 0.0092 |
| SeqModel_large | 53466 | 0.9991 | 0.9942 | 0.0052 | 0.0111 |
| ImageModel_large | 53466 | 0.9992 | 0.9949 | 0.0034 | 0.0204 |
| kibnmcmb | 53466 | 0.9986 | 0.9905 | 0.0029 | 0.0703 |
| 10_27_fusion_vit_delta64 | 53466 | 0.9987 | 0.9883 | 0.0024 | 0.0972 |
| fusion_vit_delta64 | 53466 | 0.9987 | 0.9883 | 0.0024 | 0.0972 |
| 10_31_ensemble_vit_delta64 | 53466 | 0.9987 | 0.9896 | 0.0040 | 0.0686 |

## new_retest

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 16966 | 0.6877 | 0.7322 | 0.4136 | 0.1661 |
| SeqGeneModel_large | — | — | — | — | — |
| SeqModel_large | — | — | — | — | — |
| ImageModel_large | — | — | — | — | — |
| kibnmcmb | 16966 | 0.8712 | 0.8075 | 0.1306 | 0.2357 |
| 10_27_fusion_vit_delta64 | 16966 | 0.8862 | 0.7635 | 0.0599 | 0.3598 |
| fusion_vit_delta64 | 16966 | 0.8862 | 0.7635 | 0.0599 | 0.3598 |
| 10_31_ensemble_vit_delta64 | 16966 | 0.8731 | 0.7296 | 0.5308 | 0.0886 |

## new_retest_1

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 24923 | 0.7696 | 0.7711 | 0.3544 | 0.1549 |
| SeqGeneModel_large | 24923 | 0.9249 | 0.9219 | 0.1179 | 0.0546 |
| SeqModel_large | 24923 | 0.8163 | 0.7570 | 0.4461 | 0.1232 |
| ImageModel_large | 24923 | 0.8103 | 0.8027 | 0.3946 | 0.0808 |
| kibnmcmb | 24923 | 0.8751 | 0.7720 | 0.1166 | 0.2938 |
| 10_27_fusion_vit_delta64 | 24923 | 0.8749 | 0.6607 | 0.0427 | 0.5143 |
| fusion_vit_delta64 | 24923 | 0.8749 | 0.6607 | 0.0427 | 0.5143 |
| 10_31_ensemble_vit_delta64 | 24923 | 0.8694 | 0.7660 | 0.4350 | 0.1155 |

## chip60 (external)

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 32 | 0.9196 | 0.9688 | 0.0000 | 0.2500 |
| SeqGeneModel_large | 32 | 0.6696 | 0.6250 | 0.3929 | 0.2500 |
| SeqModel_large | 32 | 0.9196 | 0.6875 | 0.3214 | 0.2500 |
| ImageModel_large | 32 | 0.9018 | 0.9062 | 0.0000 | 0.7500 |
| kibnmcmb | 32 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 10_27_fusion_vit_delta64 | 32 | 1.0000 | 0.9375 | 0.0000 | 0.5000 |
| fusion_vit_delta64 | 32 | 1.0000 | 0.9375 | 0.0000 | 0.5000 |
| 10_31_ensemble_vit_delta64 | 32 | 0.9821 | 0.9062 | 0.0714 | 0.2500 |

## karlen (external, all-positive)

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 272 | — | 0.8162 | 0.1838 | — |
| SeqGeneModel_large | 272 | — | 0.8860 | 0.1140 | — |
| SeqModel_large | 272 | — | 0.8125 | 0.1875 | — |
| ImageModel_large | 272 | — | 1.0000 | 0.0000 | — |
| kibnmcmb | 272 | — | 0.8713 | 0.1287 | — |
| 10_27_fusion_vit_delta64 | 272 | — | 0.8235 | 0.1765 | — |
| fusion_vit_delta64 | 272 | — | 0.8235 | 0.1765 | — |
| 10_31_ensemble_vit_delta64 | 272 | — | 0.8015 | 0.1985 | — |

## known (external, dilution)

| Model | N | AUROC | Accuracy | FNR | FPR |
|-------|--:|------:|---------:|----:|----:|
| fusiongene_large | 288 | 0.7427 | 0.5590 | 0.1667 | 0.9896 |
| SeqGeneModel_large | 288 | 0.7549 | 0.5174 | 0.2240 | 1.0000 |
| SeqModel_large | 288 | 0.9791 | 0.9757 | 0.0365 | 0.0000 |
| ImageModel_large | 288 | 0.9487 | 0.9028 | 0.0833 | 0.1250 |
| kibnmcmb | 288 | 0.7624 | 0.5451 | 0.1823 | 1.0000 |
| 10_27_fusion_vit_delta64 | 288 | 0.7664 | 0.5625 | 0.1562 | 1.0000 |
| fusion_vit_delta64 | 288 | 0.7664 | 0.5625 | 0.1562 | 1.0000 |
| 10_31_ensemble_vit_delta64 | 288 | 0.8699 | 0.8889 | 0.1667 | 0.0000 |

