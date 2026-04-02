import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import io
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import sys
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


def detect_silences(audio_path: Optional[str] = None,
                   audio: Optional[np.ndarray] = None,
                   sr: Optional[int] = None,
                   quiet_threshold_db: float = -28,
                   min_silence_duration: float = 0.03,
                   smooth_window_ms: float = 20,
                   merge_gap_ms: float = 30,
                   protect_speech_margin: float = 0.01) -> Tuple[float, List[List]]:
    """
    Detecte les zones de silence/respiration avec des seuils absolus (pas de percentiles).

    Approche : energie RMS lissee + seuil unique + fusion des zones proches.
    """
    if audio is None:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr

    frame_length = 512
    hop_length = 128

    # Energie RMS en dB
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(rms, ref=np.max)

    frame_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    frame_duration = hop_length / sr

    # Lissage par moyenne glissante pour absorber les pics brefs dans les respirations
    smooth_frames = max(1, int((smooth_window_ms / 1000) / frame_duration))
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        smoothed_db = np.convolve(energy_db, kernel, mode='same')
    else:
        smoothed_db = energy_db

    # Seuil unique sur l'energie lissee
    quiet_mask = smoothed_db < quiet_threshold_db

    # Marge de protection autour du speech
    if protect_speech_margin > 0:
        margin_frames = max(1, int(protect_speech_margin / frame_duration))
        transitions = np.diff(np.concatenate(([0], (~quiet_mask).astype(int), [0])))
        speech_starts = np.where(transitions == 1)[0]
        speech_ends = np.where(transitions == -1)[0]

        for start in speech_starts:
            quiet_mask[max(0, start - margin_frames):start] = False
        for end in speech_ends:
            quiet_mask[end:min(len(quiet_mask), end + margin_frames)] = False

    # Extraire zones continues
    changes = np.diff(np.concatenate(([0], quiet_mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    # Fusionner zones proches
    merge_frames = max(1, int((merge_gap_ms / 1000) / frame_duration))
    if len(starts) > 1:
        merged_starts = [starts[0]]
        merged_ends = []
        for i in range(1, len(starts)):
            if starts[i] - ends[i - 1] < merge_frames:
                continue  # fusionner
            merged_ends.append(ends[i - 1])
            merged_starts.append(starts[i])
        merged_ends.append(ends[-1])
        starts = np.array(merged_starts)
        ends = np.array(merged_ends)

    # Filtrer par duree minimale et construire la liste
    min_frames = max(1, int(min_silence_duration / frame_duration))
    silences = []

    for start_idx, end_idx in zip(starts, ends):
        if start_idx >= len(frame_times) or end_idx > len(frame_times):
            continue
        if end_idx - start_idx < min_frames:
            continue

        start_time = frame_times[start_idx]
        end_time = frame_times[min(end_idx - 1, len(frame_times) - 1)]

        is_start = (start_time <= 0.001)
        is_end = (end_time >= duration - 0.001)

        silences.append([is_start, float(start_time), float(end_time), is_end])

    return duration, silences


def _apply_trim(audio: np.ndarray, sr: int, silences: List[List], max_silence: float) -> np.ndarray:
    """Applique le trim sur un array audio a partir d'une liste de silences detectes."""
    replacement_samples = int(max_silence * sr)
    silence_replacement = np.zeros(replacement_samples)

    parts = []
    prev_end_sample = 0

    for is_start, start, end, is_end in silences:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        if start_sample > prev_end_sample:
            parts.append(audio[prev_end_sample:start_sample])

        silence_duration = end - start

        if silence_duration <= max_silence:
            parts.append(audio[start_sample:end_sample])
        else:
            if is_start or is_end:
                tiny = int(0.01 * sr)
                parts.append(np.zeros(tiny))
            else:
                parts.append(silence_replacement)

        prev_end_sample = end_sample

    if prev_end_sample < len(audio):
        parts.append(audio[prev_end_sample:])

    return np.concatenate(parts)


def trim_silences(audio_path: str, max_silence: float = 0.05) -> Tuple[np.ndarray, int, float, float]:
    """
    Supprime les silences d'un fichier audio et les remplace par max_silence secondes.
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    original_duration = len(audio) / sr

    duration, silences = detect_silences(audio=audio, sr=sr)

    long_silences = [s for s in silences if (s[2] - s[1]) > max_silence]
    if not long_silences:
        return audio, sr, original_duration, original_duration

    audio = _apply_trim(audio, sr, silences, max_silence)
    new_duration = len(audio) / sr

    return audio, sr, original_duration, new_duration


def _process_one(args):
    """Traite un seul fichier. Fonction top-level pour multiprocessing."""
    mp3_path_str, out_path_str, max_silence = args
    try:
        Path(out_path_str).parent.mkdir(parents=True, exist_ok=True)
        trimmed, sr, orig_dur, new_dur = trim_silences(mp3_path_str, max_silence)
        saved = orig_dur - new_dur

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, trimmed, sr, format='WAV')
        wav_buffer.seek(0)
        audio_seg = AudioSegment.from_wav(wav_buffer)
        audio_seg.export(out_path_str, format='mp3', bitrate='192k')

        return (True, orig_dur, new_dur, saved, None)
    except Exception as e:
        return (False, 0, 0, 0, str(e))


def process_all(root_dir: str, max_silence: float = 0.05):
    """Traite tous les fichiers MP3 en parallele, sauvegarde dans trimmed/."""
    root = Path(root_dir)
    output_dir = root / "trimmed"
    mp3_files = sorted([f for f in root.rglob("*.mp3") if "trimmed" not in f.parts])

    tasks = []
    skipped = 0
    for mp3_path in mp3_files:
        rel = mp3_path.relative_to(root)
        out_path = output_dir / rel
        if out_path.exists():
            skipped += 1
        else:
            tasks.append((str(mp3_path), str(out_path), max_silence))

    total = len(mp3_files)
    to_process = len(tasks)
    workers = min(cpu_count(), 10)

    print(f"Fichiers trouves: {total}")
    print(f"Deja faits (skip): {skipped}")
    print(f"A traiter: {to_process}")
    print(f"Workers: {workers}")
    print(f"Silence max: {max_silence}s")
    print(f"Sortie: {output_dir}")
    print("-" * 60, flush=True)

    total_saved = 0.0
    processed = 0
    errors = []

    with Pool(workers) as pool:
        for i, (args, result) in enumerate(zip(tasks, pool.imap(_process_one, tasks)), 1):
            mp3_path_str = args[0]
            rel = Path(mp3_path_str).relative_to(root)
            ok, orig_dur, new_dur, saved, err = result

            if ok:
                total_saved += saved
                processed += 1
                status = f"-{saved:.2f}s" if saved > 0.01 else "~"
                print(f"[{i}/{to_process}] {rel}  {orig_dur:.2f}s -> {new_dur:.2f}s  ({status})", flush=True)
            else:
                errors.append((str(rel), err))
                print(f"[{i}/{to_process}] {rel}  ERREUR: {err}", flush=True)

    print("-" * 60)
    print(f"Traites: {processed}/{to_process}")
    print(f"Temps total economise: {total_saved:.2f}s")
    if errors:
        print(f"Erreurs: {len(errors)}")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    root_dir = str(Path(__file__).parent)
    max_silence = 0.05

    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    if len(sys.argv) > 2:
        max_silence = float(sys.argv[2])

    process_all(root_dir, max_silence)
