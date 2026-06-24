from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
import yaml

from scripts.embed_clap import (
    _embed_with_model,
    _load_clap,
    read_mono,
)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_config_path(
    config: dict,
    keys: tuple[str, ...],
    default: str,
) -> Path:
    for key in keys:
        value = config.get(key)
        if value:
            return Path(value)

    return Path(default)


def load_mapping(path: Path) -> list[str]:
    """
    Supports either:
      - JSON list
      - one path per line

    JSON is attempted first regardless of the filename extension.
    """
    text = path.read_text(encoding="utf-8")

    try:
        value = json.loads(text)

        if isinstance(value, list):
            return [str(item) for item in value]
    except json.JSONDecodeError:
        pass

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return values / norms


def normalize_song_name(name: str) -> str:
    """
    Normalize a filename for exclusion comparisons.

    Examples:
      "Midnight City.mp3" -> "midnight city"
      "MIDNIGHT_CITY.mp3" -> "midnight city"
    """
    stem = Path(name).stem.lower()
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem)

    return stem.strip()


def find_seed_tracks(
    seed_directory: Path,
    recursive: bool = False,
) -> list[Path]:
    if not seed_directory.exists():
        raise FileNotFoundError(
            f"Seed directory does not exist: {seed_directory}"
        )

    if not seed_directory.is_dir():
        raise NotADirectoryError(
            f"Seed path is not a directory: {seed_directory}"
        )

    iterator = (
        seed_directory.rglob("*.mp3")
        if recursive
        else seed_directory.glob("*.mp3")
    )

    tracks = sorted(
        path.resolve()
        for path in iterator
        if path.is_file()
    )

    if not tracks:
        raise ValueError(
            f"No MP3 files found in {seed_directory}"
        )

    return tracks


def embed_seed_tracks(seed_tracks: list[Path]) -> np.ndarray:
    print("Loading CLAP model...")
    model = _load_clap()

    embeddings: list[np.ndarray] = []
    failed: list[tuple[Path, str]] = []

    for number, path in enumerate(seed_tracks, start=1):
        print(
            f"Embedding seed {number}/{len(seed_tracks)}: "
            f"{path.name}"
        )

        try:
            audio, sample_rate = read_mono(
                path,
                target_sr=48000,
            )

            embedding = _embed_with_model(
                model,
                audio,
                sample_rate,
            ).astype(np.float32)

            embeddings.append(embedding)

        except Exception as error:
            failed.append((path, str(error)))
            print(f"  Failed: {error}")

    if failed:
        print(f"\nFailed to embed {len(failed)} seed files.")

    if not embeddings:
        raise RuntimeError(
            "No seed files could be embedded."
        )

    return normalize_rows(np.vstack(embeddings))


def recommend_from_directory(
    config_path: str,
    seed_directory: str,
    limit: int = 50,
    candidate_pool: int = 500,
    recursive: bool = False,
) -> dict:
    config = load_config(config_path)

    embeddings_path = get_config_path(
        config,
        ("embedding_file", "embeddings_file"),
        "data/embeddings.npy",
    )

    mapping_path = get_config_path(
        config,
        ("mapping_file",),
        "data/embedding_mapping.json",
    )

    index_path = get_config_path(
        config,
        ("faiss_index_file",),
        "data/faiss.index",
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Catalog embeddings not found: {embeddings_path}"
        )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Catalog mapping not found: {mapping_path}"
        )

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}"
        )

    # Load the already-ingested search catalog.
    catalog_embeddings = np.load(embeddings_path)
    catalog_embeddings = normalize_rows(catalog_embeddings)

    mapping = load_mapping(mapping_path)

    if len(mapping) != len(catalog_embeddings):
        raise ValueError(
            "Catalog mapping and embedding counts differ: "
            f"{len(mapping)} paths versus "
            f"{len(catalog_embeddings)} embeddings."
        )

    seed_tracks = find_seed_tracks(
        Path(seed_directory),
        recursive=recursive,
    )

    seed_embeddings = embed_seed_tracks(seed_tracks)

    print(
        f"\nSuccessfully embedded "
        f"{len(seed_embeddings)} seed tracks."
    )

    # Average embedding representing the playlist's general sound.
    playlist_profile = seed_embeddings.mean(
        axis=0,
        keepdims=True,
    )
    playlist_profile = normalize_rows(playlist_profile)

    index = faiss.read_index(str(index_path))

    candidate_pool = min(
        max(candidate_pool, limit * 10),
        len(mapping),
    )

    # Retrieve candidates around the overall playlist profile.
    _, profile_ids = index.search(
        playlist_profile,
        candidate_pool,
    )

    # Also retrieve candidates near individual seed songs so that
    # smaller styles inside a diverse playlist remain represented.
    per_seed_results = min(25, len(mapping))

    _, individual_ids = index.search(
        seed_embeddings,
        per_seed_results,
    )

    candidate_ids = np.unique(
        np.concatenate(
            [
                profile_ids.reshape(-1),
                individual_ids.reshape(-1),
            ]
        )
    )

    seed_names = {
        normalize_song_name(path.name)
        for path in seed_tracks
    }

    eligible_ids: list[int] = []

    for candidate_id in candidate_ids:
        candidate_id = int(candidate_id)

        if candidate_id < 0:
            continue

        candidate_path = Path(mapping[candidate_id])
        candidate_name = normalize_song_name(
            candidate_path.name
        )

        # Do not recommend a song already in the seed directory.
        if candidate_name in seed_names:
            continue

        eligible_ids.append(candidate_id)

    if not eligible_ids:
        raise RuntimeError(
            "No eligible catalog candidates remained "
            "after excluding seed songs."
        )

    eligible_ids_array = np.asarray(
        eligible_ids,
        dtype=np.int64,
    )

    candidate_embeddings = catalog_embeddings[
        eligible_ids_array
    ]

    # Score 1: similarity to the average playlist profile.
    profile_scores = (
        candidate_embeddings @ playlist_profile[0]
    )

    # Score 2: average similarity to the three closest seed songs.
    seed_similarity_matrix = (
        candidate_embeddings @ seed_embeddings.T
    )

    nearest_count = min(
        3,
        len(seed_embeddings),
    )

    nearest_seed_scores = np.partition(
        seed_similarity_matrix,
        -nearest_count,
        axis=1,
    )[:, -nearest_count:].mean(axis=1)

    # Hybrid score:
    # general playlist taste + strong local song similarities.
    final_scores = ( # Final song recommendation weightings
        0.65 * profile_scores
        + 0.35 * nearest_seed_scores
    )

    ranked_positions = np.argsort(
        final_scores
    )[::-1][:limit]

    recommendations = []

    for rank, position in enumerate(
        ranked_positions,
        start=1,
    ):
        catalog_id = int(
            eligible_ids_array[position]
        )

        recommendation_path = mapping[catalog_id]

        recommendations.append(
            {
                "rank": rank,
                "title": Path(recommendation_path).stem,
                "path": recommendation_path,
                "score": float(final_scores[position]),
                "playlist_similarity": float(
                    profile_scores[position]
                ),
                "nearest_seed_similarity": float(
                    nearest_seed_scores[position]
                ),
            }
        )

    return {
        "seed_directory": str(
            Path(seed_directory).resolve()
        ),
        "seed_tracks": [
            {
                "title": path.stem,
                "path": str(path),
            }
            for path in seed_tracks
        ],
        "seed_track_count": len(seed_tracks),
        "embedded_seed_count": len(seed_embeddings),
        "recommendation_count": len(recommendations),
        "tracks": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend catalog songs based on a directory "
            "of liked MP3 files."
        )
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--seed-dir",
        required=True,
        help="Directory containing liked SONGNAME.mp3 files.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for MP3 files.",
    )

    parser.add_argument(
        "--output",
        default="data/directory_recommendations.json",
    )

    args = parser.parse_args()

    result = recommend_from_directory(
        config_path=args.config,
        seed_directory=args.seed_dir,
        limit=max(1, args.limit),
        candidate_pool=max(1, args.candidate_pool),
        recursive=args.recursive,
    )

    print("\nRecommendations:\n")

    for track in result["tracks"]:
        print(
            f"{track['rank']:3}. "
            f"{track['title']} "
            f"(score={track['score']:.3f})"
        )
        print(f"     {track['path']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nSaved recommendations to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
