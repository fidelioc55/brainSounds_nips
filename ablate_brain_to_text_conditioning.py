import argparse
import csv
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from diffusers import StableAudioPipeline
from diffusers.pipelines.stable_audio.pipeline_stable_audio import get_1d_rotary_pos_embed
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (
    BrainNormalizer,
    ROI_LIST,
    average_roi_data_by_sound,
    load_pooled_data,
    soundname_to_wav,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to the pooled_data file (.pkl/.npy serialized via pickle).")
    p.add_argument("--wav_dir", type=str, required=True,
                   help="Folder containing the original .wav files.")
    p.add_argument("--sound_names", type=str, required=True,
                   help="Path to SoundNames.npy.")
    p.add_argument("--labels_csv", type=str, default="Labels_288Sounds_ObjectSoundDescription.csv",
                   help="CSV file with sound captions/descriptions.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Folder where reconstructed wav files and artifacts are saved.")
    p.add_argument("--cv", type=str, default="CV2")
    p.add_argument("--stable_audio_id", type=str, default="stabilityai/stable-audio-open-1.0")
    p.add_argument("--target_duration_s", type=float, default=1.0)
    p.add_argument("--target_sr", type=int, default=44100)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 disables CFG.")
    p.add_argument("--average_test_repeats", action="store_true",
                   help="Averages repeated test trials for each sound.")
    p.add_argument("--lambda_ridge", type=float, default=1.0,
                   help="Compatibility with the old manual version; ignored when using Himalaya CV.")
    p.add_argument("--embedding_chunk_size", type=int, default=2048,
                   help="Target batch size used by Himalaya to control memory usage.")
    p.add_argument("--ridge_alphas", type=float, nargs="+",
                   default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                   help="Alpha grid for Himalaya RidgeCV.")
    p.add_argument("--text_batch_size", type=int, default=16,
                   help="Batch size for extracting textual embeddings from the text encoder.")
    p.add_argument("--decode_batch_size", type=int, default=1,
                   help="Batch size for audio reconstruction.")
    p.add_argument("--device", type=str, default="auto",
                   help="Examples: auto | cuda | cuda:0 | cpu")
    p.add_argument("--gpu_id", type=int, default=None,
                   help="GPU index to use when the device is CUDA.")
    p.add_argument("--save_groundtruth", action="store_true",
                   help="Also saves the target wav files for comparison.")
    p.add_argument("--save_predicted_embeddings", action="store_true",
                   help="Saves the predicted train/test textual embeddings as .pt files.")
    return p.parse_args()


def resolve_device(device_arg: str, gpu_id: int | None) -> torch.device:
    if device_arg != "auto":
        if device_arg == "cuda" and gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_wav_paths(wav_dir: str, sound_names: np.ndarray) -> list[str]:
    return [os.path.join(wav_dir, soundname_to_wav(name)) for name in sound_names]


def load_caption_map(labels_csv: str) -> dict[str, str]:
    caption_map: dict[str, str] = {}
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"Unexpected CSV row in {labels_csv}: {row}")
            object_label = row[0].strip()
            description = row[1].strip()
            wav_name = row[2].strip()
            caption = " ".join(part for part in [object_label, description] if part)
            caption_map[wav_name] = caption
    return caption_map


def build_sound_captions(sound_names: np.ndarray, caption_map: dict[str, str]) -> list[str]:
    captions = []
    missing = []
    for sound_name in sound_names:
        wav_name = soundname_to_wav(sound_name)
        if wav_name not in caption_map:
            missing.append(wav_name)
            captions.append("")
        else:
            captions.append(caption_map[wav_name])
    if missing:
        raise KeyError(f"Missing captions for {len(missing)} wav files. Examples: {missing[:5]}")
    return captions


@dataclass
class BrainSplit:
    train_brain: torch.Tensor
    test_brain: torch.Tensor
    train_sound_idx: np.ndarray
    test_sound_idx: np.ndarray
    brain_normalizer: BrainNormalizer


def prepare_brain_splits(
    pooled_data: dict,
    cv: str,
    average_test_repeats: bool,
) -> BrainSplit:
    train_roi, train_idx = load_pooled_data(pooled_data, cv, ROI_LIST, split="train")
    test_roi, test_idx = load_pooled_data(pooled_data, cv, ROI_LIST, split="test")

    if average_test_repeats:
        test_roi, test_idx = average_roi_data_by_sound(test_roi, test_idx)

    brain_normalizer = BrainNormalizer.fit(train_roi)

    train_brain = brain_normalizer.normalize(torch.from_numpy(train_roi.astype(np.float32)))
    test_brain = brain_normalizer.normalize(torch.from_numpy(test_roi.astype(np.float32)))

    return BrainSplit(
        train_brain=train_brain,
        test_brain=test_brain,
        train_sound_idx=train_idx,
        test_sound_idx=test_idx,
        brain_normalizer=brain_normalizer,
    )


@torch.no_grad()
def encode_captions_with_stable_audio(
    pipe: StableAudioPipeline,
    captions: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    unique_captions = sorted(set(captions))
    cache: dict[str, torch.Tensor] = {}

    for start in tqdm(range(0, len(unique_captions), batch_size), desc="Encoding unique captions"):
        batch_captions = unique_captions[start:start + batch_size]
        prompt_embeds = pipe.encode_prompt(
            prompt=batch_captions,
            device=device,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.detach().cpu().to(torch.float32)
        for caption, embed in zip(batch_captions, prompt_embeds):
            cache[caption] = embed

    stacked = torch.stack([cache[caption] for caption in captions], dim=0)
    return stacked


def fit_chunked_ridge_and_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    ridge_alphas: list[float],
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Fits a linear map x -> y using Himalaya RidgeCV and returns predictions.

    Shapes:
      x_train: [N_train, D_in]
      y_train: [N_train, D_out]
      x_eval:  [N_eval, D_in]
    """
    if device.type == "cuda":
        backend_name = "torch_cuda"
    elif device.type == "mps":
        backend_name = "torch_mps"
    else:
        backend_name = "torch"

    set_backend(backend_name)

    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device, dtype=torch.float32)
    x_eval = x_eval.to(device=device, dtype=torch.float32)

    model = RidgeCV(
        alphas=np.asarray(ridge_alphas, dtype=np.float32),
        fit_intercept=True,
        solver_params={"n_targets_batch": chunk_size},
    )
    model.fit(x_train, y_train)
    y_eval_pred = model.predict(x_eval)

    stats = {
        "backend": backend_name,
        "best_alphas": torch.as_tensor(model.best_alphas_).cpu(),
        "coef_shape": tuple(model.coef_.shape),
        "chunk_size": chunk_size,
        "input_dim": x_train.shape[1],
        "output_dim": y_train.shape[1],
    }
    return torch.as_tensor(y_eval_pred).detach().cpu(), stats


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return a @ b.T


def retrieve_nearest_captions(
    predicted_embeds: torch.Tensor,
    reference_embeds: torch.Tensor,
    reference_captions: list[str],
) -> tuple[list[str], list[float], list[int]]:
    pred_flat = predicted_embeds.reshape(predicted_embeds.shape[0], -1)
    ref_flat = reference_embeds.reshape(reference_embeds.shape[0], -1)
    sims = cosine_similarity_matrix(pred_flat, ref_flat)
    best_idx = sims.argmax(dim=1)
    best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    best_captions = [reference_captions[i] for i in best_idx.tolist()]
    return best_captions, best_scores.tolist(), best_idx.tolist()


class PredictedEmbeddingDataset(Dataset):
    def __init__(
        self,
        brain_data: torch.Tensor,
        sound_idx: np.ndarray,
        predicted_prompt_embeds: torch.Tensor,
        wav_paths: list[str],
        target_sr: int,
        target_duration_s: float,
        save_groundtruth: bool,
    ):
        self.brain_data = brain_data
        self.sound_idx = sound_idx
        self.predicted_prompt_embeds = predicted_prompt_embeds
        self.wav_paths = wav_paths
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration_s)
        self.save_groundtruth = save_groundtruth

    def __len__(self):
        return self.brain_data.shape[0]

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.squeeze(0)
        if wav.shape[0] < self.target_samples:
            wav = torch.nn.functional.pad(wav, (0, self.target_samples - wav.shape[0]))
        else:
            wav = wav[:self.target_samples]
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
        return wav

    def __getitem__(self, idx: int) -> dict:
        item = {
            "brain_data": self.brain_data[idx],
            "sound_idx": int(self.sound_idx[idx]),
            "predicted_prompt_embeds": self.predicted_prompt_embeds[idx],
        }
        if self.save_groundtruth:
            item["audio_target"] = self._load_audio(self.wav_paths[int(self.sound_idx[idx])])
        return item


@torch.no_grad()
def build_unconditional_conditioning(
    pipe: StableAudioPipeline,
    batch_size: int,
    target_duration_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty_prompt = pipe.encode_prompt(
        prompt=[""] * batch_size,
        device=device,
        do_classifier_free_guidance=False,
    ).to(dtype=dtype)

    seconds_start_hs, seconds_end_hs = pipe.encode_duration(
        audio_start_in_s=0.0,
        audio_end_in_s=target_duration_s,
        device=device,
        do_classifier_free_guidance=False,
        batch_size=batch_size,
    )

    encoder_hidden_states = torch.cat([empty_prompt, seconds_start_hs, seconds_end_hs], dim=1)
    global_hidden_states = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)
    return encoder_hidden_states, global_hidden_states


@torch.no_grad()
def decode_from_predicted_embeddings(
    pipe: StableAudioPipeline,
    predicted_prompt_embeds: torch.Tensor,
    sound_idx: np.ndarray,
    wav_paths: list[str],
    output_dir: str,
    target_duration_s: float,
    target_sr: int,
    num_inference_steps: int,
    guidance_scale: float,
    decode_batch_size: int,
    save_groundtruth: bool,
    brain_data: torch.Tensor,
):
    dataset = PredictedEmbeddingDataset(
        brain_data=brain_data,
        sound_idx=sound_idx,
        predicted_prompt_embeds=predicted_prompt_embeds,
        wav_paths=wav_paths,
        target_sr=target_sr,
        target_duration_s=target_duration_s,
        save_groundtruth=save_groundtruth,
    )
    loader = DataLoader(dataset, batch_size=decode_batch_size, shuffle=False, num_workers=0)
    device = next(pipe.transformer.parameters()).device
    scheduler = pipe.scheduler
    vae = pipe.vae.to(device).eval()

    sample_idx = 0
    for batch in tqdm(loader, desc="Decoding predicted text conditioning"):
        prompt_embeds = batch["predicted_prompt_embeds"].to(device=device, dtype=torch.float32)
        batch_sound_idx = batch["sound_idx"]
        batch_size = prompt_embeds.shape[0]

        seconds_start_hs, seconds_end_hs = pipe.encode_duration(
            audio_start_in_s=0.0,
            audio_end_in_s=target_duration_s,
            device=device,
            do_classifier_free_guidance=False,
            batch_size=batch_size,
        )

        encoder_hidden_states = torch.cat([prompt_embeds, seconds_start_hs, seconds_end_hs], dim=1)
        global_hidden_states = torch.cat([seconds_start_hs, seconds_end_hs], dim=2)

        uncond_embeds = None
        uncond_global = None
        if guidance_scale != 1.0:
            uncond_embeds, uncond_global = build_unconditional_conditioning(
                pipe=pipe,
                batch_size=batch_size,
                target_duration_s=target_duration_s,
                device=device,
                dtype=prompt_embeds.dtype,
            )

        scheduler.set_timesteps(num_inference_steps, device=device)
        waveform_length = int(pipe.transformer.config.sample_size)
        latent_shape = (batch_size, pipe.transformer.config.in_channels, waveform_length)
        latents = torch.randn(latent_shape, device=device, dtype=torch.float32)
        latents = latents * scheduler.init_noise_sigma

        rotary_embedding = get_1d_rotary_pos_embed(
            pipe.rotary_embed_dim,
            latents.shape[2] + global_hidden_states.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        rotary_embedding = tuple(r.to(device) for r in rotary_embedding)

        for t in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, t)

            noise_pred_cond = pipe.transformer(
                hidden_states=latent_input,
                timestep=t.unsqueeze(0).expand(batch_size),
                encoder_hidden_states=encoder_hidden_states,
                global_hidden_states=global_hidden_states,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

            if guidance_scale == 1.0:
                noise_pred = noise_pred_cond
            else:
                noise_pred_uncond = pipe.transformer(
                    hidden_states=latent_input,
                    timestep=t.unsqueeze(0).expand(batch_size),
                    encoder_hidden_states=uncond_embeds,
                    global_hidden_states=uncond_global,
                    rotary_embedding=rotary_embedding,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        waveform_end = int(target_duration_s * pipe.vae.config.sampling_rate)
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        audio_out = vae.decode(latents / scaling_factor).sample
        audio_out = audio_out[:, :, :waveform_end]
        audio_out = audio_out.mean(dim=1, keepdim=True).cpu()

        for i in range(batch_size):
            s_idx = int(batch_sound_idx[i])
            out_path = os.path.join(output_dir, f"pred_sound{s_idx:04d}_sample{sample_idx:04d}.wav")
            torchaudio.save(out_path, audio_out[i], target_sr)

            if save_groundtruth and "audio_target" in batch:
                gt_path = os.path.join(output_dir, f"gt_sound{s_idx:04d}_sample{sample_idx:04d}.wav")
                torchaudio.save(gt_path, batch["audio_target"][i].unsqueeze(0), target_sr)

            sample_idx += 1


def save_metadata_csv(
    path: str,
    sound_idx: np.ndarray,
    groundtruth_captions: list[str],
    retrieved_captions: list[str],
    retrieved_indices: list[int],
    retrieved_scores: list[float],
):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx",
            "sound_idx",
            "groundtruth_caption",
            "nearest_caption_from_embedding",
            "nearest_caption_sound_idx",
            "cosine_similarity",
        ])
        for sample_idx, s_idx in enumerate(sound_idx.tolist()):
            writer.writerow([
                sample_idx,
                int(s_idx),
                groundtruth_captions[s_idx],
                retrieved_captions[sample_idx],
                int(retrieved_indices[sample_idx]),
                float(retrieved_scores[sample_idx]),
            ])


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = resolve_device(args.device, args.gpu_id)
    print(f"Using device: {device}")

    with open(args.data_path, "rb") as f:
        pooled_data = pickle.load(f)
    sound_names = np.load(args.sound_names, allow_pickle=True)
    wav_paths = build_wav_paths(args.wav_dir, sound_names)

    caption_map = load_caption_map(args.labels_csv)
    sound_captions = build_sound_captions(sound_names, caption_map)

    print("Preparing brain train/test splits...")
    brain_split = prepare_brain_splits(
        pooled_data=pooled_data,
        cv=args.cv,
        average_test_repeats=args.average_test_repeats,
    )

    print("Loading Stable Audio pipeline...")
    pipe = StableAudioPipeline.from_pretrained(
        args.stable_audio_id,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.vae.eval()
    pipe.transformer.eval()
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.eval()

    print("Extracting text conditioning embeddings from captions...")
    all_sound_prompt_embeds = encode_captions_with_stable_audio(
        pipe=pipe,
        captions=sound_captions,
        device=device,
        batch_size=args.text_batch_size,
    )

    train_indices = torch.from_numpy(brain_split.train_sound_idx).long()
    test_indices = torch.from_numpy(brain_split.test_sound_idx).long()
    train_target_embeds = all_sound_prompt_embeds[train_indices]
    test_target_embeds = all_sound_prompt_embeds[test_indices]

    x_train = brain_split.train_brain.reshape(brain_split.train_brain.shape[0], -1)
    x_test = brain_split.test_brain.reshape(brain_split.test_brain.shape[0], -1)
    y_train = train_target_embeds.reshape(train_target_embeds.shape[0], -1)

    print(
        "Training linear ridge model with Himalaya "
        f"(input_dim={x_train.shape[1]}, output_dim={y_train.shape[1]}, "
        f"alphas={args.ridge_alphas})..."
    )
    predicted_test_flat, ridge_stats = fit_chunked_ridge_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_test,
        ridge_alphas=args.ridge_alphas,
        chunk_size=args.embedding_chunk_size,
        device=device,
    )
    predicted_test_embeds = predicted_test_flat.reshape_as(test_target_embeds)

    mse = torch.mean((predicted_test_embeds - test_target_embeds) ** 2).item()
    cosine = torch.nn.functional.cosine_similarity(
        predicted_test_embeds.reshape(predicted_test_embeds.shape[0], -1),
        test_target_embeds.reshape(test_target_embeds.shape[0], -1),
        dim=1,
    ).mean().item()
    print(f"Test embedding MSE: {mse:.6f}")
    print(f"Test embedding cosine similarity: {cosine:.6f}")

    retrieved_captions, retrieved_scores, retrieved_indices = retrieve_nearest_captions(
        predicted_embeds=predicted_test_embeds,
        reference_embeds=all_sound_prompt_embeds,
        reference_captions=sound_captions,
    )

    metadata_path = os.path.join(args.output_dir, "predicted_caption_matches.csv")
    save_metadata_csv(
        path=metadata_path,
        sound_idx=brain_split.test_sound_idx,
        groundtruth_captions=sound_captions,
        retrieved_captions=retrieved_captions,
        retrieved_indices=retrieved_indices,
        retrieved_scores=retrieved_scores,
    )

    summary_path = os.path.join(args.output_dir, "ridge_summary.pt")
    torch.save(
        {
            "brain_normalizer_mean": brain_split.brain_normalizer.mean,
            "brain_normalizer_std": brain_split.brain_normalizer.std,
            "ridge_stats": ridge_stats,
            "test_embedding_mse": mse,
            "test_embedding_cosine": cosine,
            "cv": args.cv,
            "average_test_repeats": args.average_test_repeats,
            "target_duration_s": args.target_duration_s,
            "target_sr": args.target_sr,
            "stable_audio_id": args.stable_audio_id,
        },
        summary_path,
    )

    if args.save_predicted_embeddings:
        embeds_path = os.path.join(args.output_dir, "predicted_test_prompt_embeds.pt")
        torch.save(
            {
                "predicted_test_prompt_embeds": predicted_test_embeds.cpu(),
                "groundtruth_test_prompt_embeds": test_target_embeds.cpu(),
                "test_sound_idx": brain_split.test_sound_idx,
            },
            embeds_path,
        )

    print("Running Stable Audio inference from predicted text embeddings...")
    decode_from_predicted_embeddings(
        pipe=pipe,
        predicted_prompt_embeds=predicted_test_embeds.cpu(),
        sound_idx=brain_split.test_sound_idx,
        wav_paths=wav_paths,
        output_dir=args.output_dir,
        target_duration_s=args.target_duration_s,
        target_sr=args.target_sr,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        decode_batch_size=args.decode_batch_size,
        save_groundtruth=args.save_groundtruth,
        brain_data=brain_split.test_brain.cpu(),
    )

    print(f"Done. Outputs saved in: {args.output_dir}")
    print(f"Metadata saved in: {metadata_path}")


if __name__ == "__main__":
    main()