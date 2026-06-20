"""Run fusiongene_large through the same pipeline as eval_fusion_all (live, all datasets)."""
import eval_fusion_all as E

E.MODELS = [('fusiongene_large', 'model_weights/fusiongene_large.ckpt')]
E.main()
