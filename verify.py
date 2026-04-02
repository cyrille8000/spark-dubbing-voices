from trim_silences import detect_silences
from pathlib import Path
from multiprocessing import Pool, cpu_count


def check_one(mp3_str):
    try:
        dur, silences = detect_silences(audio_path=mp3_str)
        viols = [(s, e, e - s) for _, s, e, _ in silences if e - s > 0.06]
        return (mp3_str, viols, None)
    except Exception as e:
        return (mp3_str, [], str(e))


if __name__ == "__main__":
    root = Path(__file__).parent / "trimmed"
    mp3_files = sorted([str(f) for f in root.rglob("*.mp3")])
    total = len(mp3_files)
    workers = min(cpu_count(), 10)
    print(f"Verification de {total} fichiers avec {workers} workers...", flush=True)

    violations = []
    errors = []
    with Pool(workers) as pool:
        for i, (path, viols, err) in enumerate(pool.imap(check_one, mp3_files), 1):
            if err:
                errors.append((path, err))
            if viols:
                violations.extend([(path, s, e, d) for s, e, d in viols])
            if i % 200 == 0:
                print(f"[{i}/{total}]...", flush=True)

    print(f"Fichiers verifies: {total}")
    print(f"Erreurs: {len(errors)}")
    for p, e in errors:
        print(f"  ERR {p}: {e}")
    print(f"Violations (silence > 0.06s): {len(violations)}")
    if violations:
        for path, s, e, d in violations[:30]:
            rel = Path(path).relative_to(root)
            print(f"  {rel}  {s:.3f}-{e:.3f}  ({d:.3f}s)")
    else:
        print("OK - aucun silence > 0.06s detecte")
