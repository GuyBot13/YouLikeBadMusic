# Audio playlist recommendation tool
#
# This program:
#   - embeds liked MP3 files with CLAP
#   - searches an existing FAISS catalog
#   - ranks songs against the playlist and its closest seed tracks
#   - optionally asks Ollama for short recommendation explanations
#

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import faiss
import numpy as np
import yaml

from scripts.embed_clap import (
  _embed_with_model,
  _load_clap,
  read_mono,
)

# ---------------- Recommendation Labels ---------------- #

EXPLANATION_ANGLES = [
  (
    "Focus primarily on how the candidate's likely genre or style "
    "relates to the playlist."
  ),
  "Focus primarily on mood and emotional character.",
  "Focus primarily on tone, texture, and overall sonic character.",
  (
    "Focus on how the candidate complements the playlist while "
    "still introducing some variety."
  ),
  "Focus on the strongest connection to the nearest seed songs.",
  "Give a balanced explanation using style, mood, and tone.",
]

GENRE_LABELS = [
  "hip-hop",
  "rap",
  "rock",
  "indie rock",
  "alternative rock",
  "punk",
  "metal",
  "jazz",
  "classical",
  "orchestral",
  "electronic",
  "ambient",
  "vaporwave",
  "funk",
  "soul",
  "R&B",
  "pop",
  "folk",
  "experimental",
]

MOOD_LABELS = [
  "energetic",
  "calm",
  "melancholic",
  "uplifting",
  "dark",
  "dreamy",
  "tense",
  "playful",
  "reflective",
  "aggressive",
  "mysterious",
  "nostalgic",
  "triumphant",
  "somber",
]

TONE_LABELS = [
  "warm",
  "bright",
  "dark",
  "raw",
  "polished",
  "spacious",
  "intimate",
  "atmospheric",
  "dense",
  "minimal",
  "rhythmic",
  "smooth",
  "abrasive",
  "cinematic",
]


# ---------------- Config and File Utilities ---------------- #

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

  return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------- Audio Utilities ---------------- #

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


def embed_seed_tracks(
  seed_tracks: list[Path],
) -> tuple[np.ndarray, list[Path]]:
  print("Loading CLAP model...")
  model = _load_clap()

  embeddings: list[np.ndarray] = []
  successful_tracks: list[Path] = []
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
      successful_tracks.append(path)

    except Exception as error:
      failed.append((path, str(error)))
      print(f"  Failed: {error}")

  if failed:
    print(f"\nFailed to embed {len(failed)} seed files.")

  if not embeddings:
    raise RuntimeError("No seed files could be embedded.")

  return (
    normalize_rows(np.vstack(embeddings)),
    successful_tracks,
    model,
  )


# ---------------- Ollama Explanations ---------------- #

def call_ollama(
  prompt: str,
  config: dict,
) -> str:
  """
    Send a prompt to the local Ollama API.

    Raises RuntimeError when Ollama is unreachable, the model is missing,
    or Ollama returns an invalid response.
    """
  llm_config = config.get("llm", {})

  model = llm_config.get("model")

  if not model:
    raise RuntimeError(
      "No Ollama model configured under llm.model."
    )

  base_url = str(
    llm_config.get(
      "base_url",
      "http://127.0.0.1:11434",
    )
  ).rstrip("/")

  url = f"{base_url}/api/generate"

  max_tokens = int(
    llm_config.get("max_tokens", 120)
  )
  temperature = float(
    llm_config.get("temperature", 0.2)
  )
  timeout_seconds = float(
    llm_config.get("timeout_seconds", 120)
  )
  keep_alive = llm_config.get(
    "keep_alive",
    "10m",
  )

  payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "keep_alive": keep_alive,
    "options": {
      "num_predict": max_tokens,
      "temperature": temperature,
    },
  }

  request = Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "Content-Type": "application/json",
    },
    method="POST",
  )

  try:
    with urlopen(
      request,
      timeout=timeout_seconds,
    ) as response:
      response_data = json.loads(
        response.read().decode("utf-8")
      )

  except HTTPError as error:
    try:
      body = error.read().decode(
        "utf-8",
        errors="replace",
      )
    except Exception:
      body = ""

    raise RuntimeError(
      f"Ollama returned HTTP {error.code}: {body}"
    ) from error

  except URLError as error:
    raise RuntimeError(
      "Could not connect to Ollama at "
      f"{base_url}. Make sure Ollama is running."
    ) from error

  except TimeoutError as error:
    raise RuntimeError(
      "Ollama did not respond before the timeout."
    ) from error

  except json.JSONDecodeError as error:
    raise RuntimeError(
      "Ollama returned invalid JSON."
    ) from error

  generated_text = str(
    response_data.get("response", "")
  ).strip()

  if not generated_text:
    raise RuntimeError(
      "Ollama returned an empty explanation."
    )

  return generated_text


def fallback_explanation(track: dict) -> str:
  """
    Produce a grounded explanation without an LLM.
    """
  nearest_tracks = track.get(
    "nearest_seed_tracks",
    [],
  )

  if nearest_tracks:
    nearest_names = ", ".join(
      item["title"]
      for item in nearest_tracks[:2]
    )

    return (
      "This track ranked highly because it is close to "
      "the playlist's overall audio profile and is especially "
      f"similar to seed tracks such as {nearest_names}."
    )

  return (
    "This track ranked highly because its CLAP embedding is "
    "similar to the combined audio profile of the seed playlist."
  )


def add_ollama_explanations(
  result: dict,
  config: dict,
  max_tracks: int = 10,
) -> None:
  tracks = result.get("tracks", [])

  explanation_count = min(
    max(0, max_tracks),
    len(tracks),
  )

  if explanation_count == 0:
    return

  print(
    f"\nGenerating {explanation_count} "
    "Ollama explanations..."
  )

  previous_explanations: list[str] = []

  for number, track in enumerate(
    tracks[:explanation_count],
    start=1,
  ):
    nearest_tracks = track.get(
      "nearest_seed_tracks",
      [],
    )

    nearest_lines = "\n".join(
      f"- {item['title']}: "
      f"similarity={item['similarity']:.3f}"
      for item in nearest_tracks
    )

    if not nearest_lines:
      nearest_lines = "- No individual seed matches available"

    angle = EXPLANATION_ANGLES[
      (track["rank"] - 1) % len(EXPLANATION_ANGLES)
    ]

    candidate_descriptor_text = format_audio_descriptors(
      track.get("audio_descriptors", {})
    )

    playlist_descriptor_text = format_audio_descriptors(
      result.get("playlist_descriptors", {})
    )

    previous_text = "\n".join(
      f"- {explanation}"
      for explanation in previous_explanations[-3:]
    )

    if not previous_text:
      previous_text = "- None yet"

    prompt = f"""
You are explaining a recommendation made by an offline audio
recommendation system.

Write one or two natural sentences explaining why the candidate fits
the user's playlist.

Explanation angle:
{angle}

Candidate:
{track["title"]}

Candidate audio-analysis signals:
{candidate_descriptor_text}

Overall playlist audio-analysis signals:
{playlist_descriptor_text}

Overall playlist-profile similarity:
{track["playlist_similarity"]:.3f}

Average similarity to nearest seed tracks:
{track["nearest_seed_similarity"]:.3f}

Nearest seed tracks:
{nearest_lines}

Recent explanations already used:
{previous_text}

Rules:
- Discuss genre, mood, tone, texture, similarity, or contrast when
  supported by the supplied signals.
- Treat descriptor labels as estimates, not certain facts.
- Use wording such as "leans toward", "suggests", or "appears to share".
- Do not invent artists, instruments, lyrics, release dates, or
  historical facts.
- Do not mention CLAP, FAISS, embeddings, scores, or algorithms.
- Do not start with "This track was recommended because".
- Avoid repeatedly saying "similar to the playlist".
- Use noticeably different wording from the recent explanations.
- Return only the explanation.
""".strip()

    print(
      f"  Explaining {number}/{explanation_count}: "
      f"{track['title']}"
    )

    try:
      explanation = call_ollama(
        prompt,
        config,
      )

      track["explanation"] = explanation
      track["explanation_source"] = "ollama"

      previous_explanations.append(explanation)

    except Exception as error:
      print(f"  Ollama explanation failed: {error}")

      explanation = fallback_explanation(track)

      track["explanation"] = explanation
      track["explanation_source"] = "fallback"
      track["explanation_error"] = str(error)

      previous_explanations.append(explanation)


# ---------------- Audio Descriptors ---------------- #

def build_text_label_embeddings(
  model,
  labels: list[str],
  prompt_template: str,
) -> np.ndarray:
  prompts = [
    prompt_template.format(label=label)
    for label in labels
  ]

  embeddings = model.get_text_embedding(
    prompts,
    use_tensor=False,
  )

  return normalize_rows(
    np.asarray(embeddings, dtype=np.float32)
  )


def build_descriptor_banks(model) -> dict:
  """
    Create text embeddings once and reuse them for every recommendation.
    """
  return {
    "genre": {
      "labels": GENRE_LABELS,
      "embeddings": build_text_label_embeddings(
        model,
        GENRE_LABELS,
        "This audio is a {label} song.",
      ),
    },
    "mood": {
      "labels": MOOD_LABELS,
      "embeddings": build_text_label_embeddings(
        model,
        MOOD_LABELS,
        "This music has a {label} mood.",
      ),
    },
    "tone": {
      "labels": TONE_LABELS,
      "embeddings": build_text_label_embeddings(
        model,
        TONE_LABELS,
        "This music has a {label} tone and texture.",
      ),
    },
  }


def score_label_bank(
  audio_embedding: np.ndarray,
  labels: list[str],
  text_embeddings: np.ndarray,
  top_k: int = 3,
) -> list[dict]:
  audio_embedding = normalize_rows(
    np.asarray(
      audio_embedding,
      dtype=np.float32,
    ).reshape(1, -1)
  )[0]

  similarities = audio_embedding @ text_embeddings.T

  top_indices = np.argsort(
    similarities
  )[::-1][:top_k]

  return [
    {
      "label": labels[int(index)],
      "score": float(similarities[int(index)]),
    }
    for index in top_indices
  ]


def describe_audio_embedding(
  audio_embedding: np.ndarray,
  descriptor_banks: dict,
) -> dict:
  result = {}

  for category, bank in descriptor_banks.items():
    result[category] = score_label_bank(
      audio_embedding=audio_embedding,
      labels=bank["labels"],
      text_embeddings=bank["embeddings"],
      top_k=3,
    )

  return result


# ---------------- Recommendation Search ---------------- #

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

  # load the indexed song catalog
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

  (
    seed_embeddings,
    embedded_seed_tracks,
    clap_model,
  ) = embed_seed_tracks(seed_tracks)

  print("Building genre, mood, and tone label embeddings...")

  descriptor_banks = build_descriptor_banks(clap_model)

  print(
    f"\nSuccessfully embedded "
    f"{len(seed_embeddings)} seed tracks."
  )

  # average the seed tracks into one playlist profile
  playlist_profile = seed_embeddings.mean(
    axis=0,
    keepdims=True,
  )
  playlist_profile = normalize_rows(playlist_profile)

  playlist_descriptors = describe_audio_embedding(
    playlist_profile[0],
    descriptor_banks,
  )

  index = faiss.read_index(str(index_path))

  candidate_pool = min(
    max(candidate_pool, limit * 10),
    len(mapping),
  )

  # search around the overall playlist sound
  _, profile_ids = index.search(
    playlist_profile,
    candidate_pool,
  )

  # also search around each seed so smaller styles are not lost
  per_seed_results = min(25, len(mapping))

  _, individual_ids = index.search(
    seed_embeddings,
    per_seed_results,
  )

  candidate_ids = np.unique(
    np.concatenate([
      profile_ids.reshape(-1),
      individual_ids.reshape(-1),
    ])
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

    # skip songs already in the liked-song directory
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

  # general similarity to the playlist as a whole
  profile_scores = (
    candidate_embeddings @ playlist_profile[0]
  )

  # local similarity to the nearest seed songs
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

  nearest_seed_positions = np.argsort(
    seed_similarity_matrix,
    axis=1,
  )[:, -nearest_count:][:, ::-1]

  # blend the overall playlist match with the closest seed matches
  final_scores = (
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

    nearest_seed_tracks = []

    for seed_position in nearest_seed_positions[position]:
      seed_position = int(seed_position)

      nearest_seed_tracks.append({
        "title": embedded_seed_tracks[
          seed_position
        ].stem,
        "path": str(
          embedded_seed_tracks[seed_position]
        ),
        "similarity": float(
          seed_similarity_matrix[
            position,
            seed_position,
          ]
        ),
      })

    audio_descriptors = describe_audio_embedding(
      candidate_embeddings[position],
      descriptor_banks,
    )

    recommendations.append({
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
      "nearest_seed_tracks": nearest_seed_tracks,
      "audio_descriptors": audio_descriptors,
    })

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
    "embedded_seed_count": len(embedded_seed_tracks),
    "playlist_descriptors": playlist_descriptors,
    "recommendation_count": len(recommendations),
    "tracks": recommendations,
  }


# ---------------- Output Formatting ---------------- #

def format_descriptor_list(
  descriptors: list[dict],
) -> str:
  return ", ".join(
    f"{item['label']} ({item['score']:.3f})"
    for item in descriptors
  )


def format_audio_descriptors(
  descriptors: dict,
) -> str:
  return (
    "Possible genre/style signals: "
    f"{format_descriptor_list(descriptors.get('genre', []))}\n"
    "Possible mood signals: "
    f"{format_descriptor_list(descriptors.get('mood', []))}\n"
    "Possible tone/texture signals: "
    f"{format_descriptor_list(descriptors.get('tone', []))}"
  )


# ---------------- Main ---------------- #

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

  parser.add_argument(
    "--explain",
    action="store_true",
    help="Generate recommendation explanations with Ollama.",
  )

  parser.add_argument(
    "--explain-limit",
    type=int,
    default=10,
    help=(
      "Maximum number of recommendations to explain. "
      "Defaults to 10."
    ),
  )

  args = parser.parse_args()

  result = recommend_from_directory(
    config_path=args.config,
    seed_directory=args.seed_dir,
    limit=max(1, args.limit),
    candidate_pool=max(1, args.candidate_pool),
    recursive=args.recursive,
  )

  config = load_config(args.config)

  llm_enabled = bool(
    config.get("llm", {}).get("enabled", False)
  )

  if args.explain or llm_enabled:
    add_ollama_explanations(
      result=result,
      config=config,
      max_tracks=max(0, args.explain_limit),
    )

  print("\nRecommendations:\n")

  for track in result["tracks"]:
    print(
      f"{track['rank']:3}. "
      f"{track['title']} "
      f"(score={track['score']:.3f})"
    )
    print(f"     {track['path']}")

    if track.get("explanation"):
      print(
        f"     Why: {track['explanation']}"
      )

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
