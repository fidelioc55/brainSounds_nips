# MAAS: Direct Conditioning of Audio Diffusion Transformers with fMRI Signals

## Code Structure

**`dataset.py`** — PyTorch `Dataset` and data loading utilities. Handles z-score normalisation of brain data (fitted on train, applied to test), audio loading and resampling, and cross-validation splits.

**`model.py`** — MAAS model definition: ROI Projector, Brain Token Generator, and fMRI Adapter. The adapter injects brain-derived tokens into the cross-attention layers of the Stable Audio DiT, preserving RoPE asymmetry and GQA compatibility.

**`transformer.py`** — Transformer encoder/decoder used by the Brain Token Generator to aggregate ROI representations into a sequence of conditioning tokens via learned queries.

**`train.py`** — Training loop. Fine-tunes the cross-attention layers of the DiT jointly with the adapter, using v-prediction MSE loss with Min-SNR-γ weighting and classifier-free guidance dropout.

**`decode_audio.py`** — Inference with all ROIs active. Produces one reconstructed .wav per test stimulus.

**`decode_audio_roi_subset.py`** — Same as `decode_audio.py` with ROI ablation support (`--keep_rois` / `--drop_rois`), used for the leave-one-out and isolation analyses in the paper.

**`ablate_brain_to_text_conditioning.py`** — Brain-to-text baseline: decodes flan-T5 embeddings from brain activity via ridge regression and conditions the frozen DiT on the predicted sequence.

**`evaluate_ablations_models.ipynb`** — Computes all quantitative metrics (CLAP similarity, identification accuracy, Mel-SSIM, log-mel distance) and the perceptual fidelity metric Δᵢ. The NSL auditory-cortical representation requires `aud24.mat`.

**`evaluate_reconstructions.ipynb`** — Qualitative inspection of individual reconstructions: raw audio visualisation.

Generative backbone: https://huggingface.co/stabilityai/stable-audio-open-1.0
