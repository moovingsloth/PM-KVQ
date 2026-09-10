"""Local-only AIME selection, data preflight, provenance, and response coverage.

The checked-in mixed selection is generated once with Random(42), sampling 15
IDs from each year's I.1--I.15, II.1--II.15 pool, in 2024 then 2025 order.
Runtime only reads that manifest; it never resamples or downloads data.
"""

import csv
import hashlib
import json
from pathlib import Path
import re


MIXED_MANIFEST = Path(__file__).with_name("manifests") / "thinkv_mixed.json"
MIXED_COUNTS = {"smoke": 1, "day": 5, "full": 15}
ID_PATTERN = re.compile(r"aime_(2024|2025)_(I|II)\.([1-9]|1[0-5])")


def parse_problem_id(problem_id):
    match = ID_PATTERN.fullmatch(problem_id) if isinstance(problem_id, str) else None
    if match is None:
        raise ValueError(f"invalid AIME problem ID: {problem_id!r}")
    year, subset, number = match.groups()
    return int(year), subset, int(number)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


def validate_manifest(manifest, require_full=False):
    required = {"schema_version", "protocol", "seed", "problem_ids"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError(f"AIME manifest keys must be exactly {sorted(required)}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported AIME manifest schema")
    if manifest["protocol"] not in ("aime2025", "thinkv_mixed"):
        raise ValueError("unsupported AIME protocol")
    if type(manifest["seed"]) is not int or manifest["seed"] != 42:
        raise ValueError("AIME selection seed must be 42")
    ids = manifest["problem_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("AIME problem_ids must be a nonempty array")
    parsed = [parse_problem_id(problem_id) for problem_id in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate AIME problem IDs")
    counts = {year: sum(item[0] == year for item in parsed) for year in (2024, 2025)}
    if manifest["protocol"] == "thinkv_mixed":
        allowed = (15,) if require_full else (1, 5, 15)
        if counts[2024] != counts[2025] or counts[2024] not in allowed:
            raise ValueError("mixed AIME requires equal year counts: 15 each in the source "
                             "manifest; 1, 5, or 15 each in a run manifest")
    elif counts[2024] or (require_full and counts[2025] != 30):
        raise ValueError("aime2025 requires only 2025 IDs (30 in a full manifest)")
    return manifest


def load_manifest(path, require_full=False):
    return validate_manifest(load_json(path), require_full=require_full)


def select_mixed_manifest(preset):
    source = load_manifest(MIXED_MANIFEST, require_full=True)
    if source["protocol"] != "thinkv_mixed":
        raise ValueError("mixed source manifest must use the thinkv_mixed protocol")
    count = MIXED_COUNTS[preset]
    ids = [problem_id for year in (2024, 2025)
           for problem_id in [p for p in source["problem_ids"]
                              if parse_problem_id(p)[0] == year][:count]]
    return validate_manifest(dict(source, problem_ids=ids))


def manifest_bytes(manifest):
    validate_manifest(manifest)
    return (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def load_problems(dataset_path, manifest):
    """Load only selected problems. CSV id (or row ordinal) maps to problem number.

    Normalize three-digit answers as the legacy pandas loader does, but hash the
    original answer cell and problem bytes so content changes remain visible.
    """
    validate_manifest(manifest)
    root = Path(dataset_path)
    answers_by_subset = {}
    samples = []
    for problem_id in manifest["problem_ids"]:
        year, subset, number = parse_problem_id(problem_id)
        directory = root / f"aime_{year}_{subset}"
        if directory not in answers_by_subset:
            answer_path = directory / "answers.csv"
            answers = {}
            with answer_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "answer" not in reader.fieldnames:
                    raise ValueError(f"missing answer column: {answer_path}")
                for ordinal, row in enumerate(reader, 1):
                    raw_id = row.get("id", str(ordinal))
                    if not isinstance(raw_id, str) or not re.fullmatch(r"[1-9]|1[0-5]", raw_id):
                        raise ValueError(f"invalid answer ID in {answer_path}: {raw_id!r}")
                    index = int(raw_id)
                    if index in answers:
                        raise ValueError(f"duplicate answer ID {index} in {answer_path}")
                    raw_answer = row.get("answer")
                    if not isinstance(raw_answer, str) or not re.fullmatch(r"[0-9]{1,3}", raw_answer):
                        raise ValueError(f"invalid AIME answer for {directory.name}.{index}")
                    answers[index] = (str(int(raw_answer)), hashlib.sha256(raw_answer.encode("utf-8")).hexdigest())
            answers_by_subset[directory] = answers
        if number not in answers_by_subset[directory]:
            raise ValueError(f"missing answer for {problem_id}")
        problem_path = directory / "problems" / f"{number}.tex"
        content = problem_path.read_bytes()
        # Match text-mode universal-newline reading used by the legacy evaluator.
        problem = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if not problem.strip():
            raise ValueError(f"empty problem: {problem_path}")
        answer, answer_digest = answers_by_subset[directory][number]
        samples.append({"id": problem_id, "problem": problem, "answer": answer,
                        "problem_sha256": hashlib.sha256(content).hexdigest(),
                        "answer_sha256": answer_digest})
    return samples


def provenance(dataset_path, manifest, samples):
    if [sample["id"] for sample in samples] != manifest["problem_ids"]:
        raise ValueError("loaded AIME samples do not match manifest order")
    hashes = {sample["id"]: {key: sample[key] for key in ("problem_sha256", "answer_sha256")}
              for sample in samples}
    return {"protocol": manifest["protocol"], "dataset_path": str(Path(dataset_path).resolve()),
            "problem_ids": manifest["problem_ids"],
            "manifest_sha256": hashlib.sha256(manifest_bytes(manifest)).hexdigest(),
            "selected_content_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(),
            "content_hashes": hashes}


def expected_response_keys(manifest, n_responses):
    validate_manifest(manifest)
    if type(n_responses) is not int or n_responses <= 0:
        raise ValueError("manifest coverage requires explicit positive n_responses")
    return [f"{problem_id}.{response}" for problem_id in manifest["problem_ids"]
            for response in range(n_responses)]


def load_responses(responses_dir, manifest, n_responses):
    expected = expected_response_keys(manifest, n_responses)
    records = {}
    for path in sorted(Path(responses_dir).glob("*.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"response file must contain a JSON object: {path}")
        duplicates = records.keys() & data.keys()
        if duplicates:
            raise ValueError(f"duplicate response keys across files: {sorted(duplicates)}")
        records.update(data)
    missing, unexpected = set(expected) - records.keys(), records.keys() - set(expected)
    if missing or unexpected:
        raise ValueError(f"AIME response coverage mismatch: missing={sorted(missing)}, "
                         f"unexpected={sorted(unexpected)}")
    return {key: records[key] for key in expected}
