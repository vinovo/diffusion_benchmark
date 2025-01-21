import os
from datetime import datetime
import numpy as np
from cleanfid import fid
import torch


def calculate_fid_for_folders(
    reference_folder: str,
    generated_folder: str,
    ref_cache_path: str | None = None,
    gen_cache_path: str | None = None,
    mode: str = "clean",
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    force_overwrite: bool = False,
    verbose: bool = True,
) -> float:
    """
    Calculate FID score between two folders of images.

    Args:
        reference_folder (str): Path to the folder containing reference images.
        generated_folder (str): Path to the folder containing generated images.
        ref_cache_path (str | None): Path to cache reference features (optional).
        gen_cache_path (str | None): Path to cache generated features (optional).
        mode (str): Resizing mode for CleanFID ("clean", "legacy", etc.).
        batch_size (int): Batch size for feature extraction.
        device (str | torch.device): Device for feature extraction ("cuda" or "cpu").
        force_overwrite (bool): Whether to overwrite cached features if present.
        verbose (bool): Whether to show progress logs.

    Returns:
        float: FID score.
    """
    def get_fid_features(folder: str, cache_path: str | None) -> tuple[np.ndarray, np.ndarray]:
        if cache_path is not None and os.path.exists(cache_path) and not force_overwrite:
            npz = np.load(cache_path)
            return npz["mu"], npz["sigma"]

        feat_model = fid.build_feature_extractor(mode, device)
        np_feats = fid.get_folder_features(
            folder,
            feat_model,
            num_workers=8,
            num=None,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            mode=mode,
            description=f"Extracting features from {folder}",
        )
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, mu=mu, sigma=sigma)
        return mu, sigma

    # Get FID features for reference and generated folders
    mu1, sigma1 = get_fid_features(reference_folder, ref_cache_path)
    mu2, sigma2 = get_fid_features(generated_folder, gen_cache_path)

    # Calculate FID score
    return float(fid.frechet_distance(mu1, sigma1, mu2, sigma2))


if __name__ == "__main__":
    # Input folders
    reference_folder = "./output/reference"
    generated_folder = "./output/generate"

    # Optional cache paths
    ref_cache_path = "./cache/ref_stats.npz"
    gen_cache_path = "./cache/gen_stats.npz"

    # FID calculation
    fid_score = calculate_fid_for_folders(
        reference_folder,
        generated_folder,
        ref_cache_path=ref_cache_path,
        gen_cache_path=gen_cache_path,
        mode="clean",
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        force_overwrite=False,
        verbose=True,
    )

    print(f"FID Score: {fid_score:.2f}")
