import os
import json

from pm_kvq.evaluation.aime_manifest import load_manifest, load_responses

DEFAULT_CODE_GENERATION_DATASET_PATH = "livecodebench/code_generation_lite"


def _judge_aime_manifest(responses_dir, manifest_path, n_responses):
    manifest = load_manifest(manifest_path)
    records = load_responses(responses_dir, manifest, n_responses)
    for key, record in records.items():
        if not isinstance(record, dict) or type(record.get("judgement")) is not bool:
            raise ValueError(f"invalid response judgement: {key}")
        if not isinstance(record.get("gold"), str) or not record["gold"]:
            raise ValueError(f"invalid response gold: {key}")
        if "response_answer" not in record or not (record["response_answer"] is None or isinstance(record["response_answer"], str)):
            raise ValueError(f"invalid response answer: {key}")
        for field in ("input_len", "output_len"):
            if type(record.get(field)) is not int or record[field] < 0:
                raise ValueError(f"invalid response {field}: {key}")
    pass_at_1, voting, lengths = [], [], []
    for problem_id in manifest["problem_ids"]:
        responses = [records[f"{problem_id}.{i}"] for i in range(n_responses)]
        gold = responses[0]["gold"]
        if any(record["gold"] != gold for record in responses):
            raise ValueError(f"inconsistent gold answers for {problem_id}")
        pass_at_1.append(sum(record["judgement"] for record in responses) / n_responses * 100)
        answers = [record["response_answer"] for record in responses]
        # Preserve voting semantics; ties use the earliest response ordinal.
        voting.append(max(answers, key=answers.count) == gold)
        lengths.append(sum(record["input_len"] + record["output_len"] for record in responses) / n_responses)
    print(f"complete test data: {len(records)}/{len(records)} ({manifest['protocol']})")
    return (sum(pass_at_1) / len(pass_at_1), sum(voting) / len(voting) * 100,
            pass_at_1, voting, lengths)


def judge_aime(responses_dir, version=2024, aime_manifest=None, n_responses=None):
    if aime_manifest is not None:
        return _judge_aime_manifest(responses_dir, aime_manifest, n_responses)
    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
            for k in data.keys():
                data[k].pop("response")
        test_data.update(data)

    if len(test_data) == 16 * 30:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/480")

    ids = [f"aime_{version}_I.{i}" for i in range(1, 16)] + [f"aime_{version}_II.{i}" for i in range(1, 16)]
    detailed_pass_at_1 = [0] * 30
    detailed_voting_judgements = [None] * 30
    avg_length = [0] * 30

    for i, idx in enumerate(ids):
        test_keys = [key for key in test_data.keys() if key.startswith(f"{idx}.")]
        if len(test_keys) == 0:
            continue
        gold = test_data[test_keys[0]]["gold"]
        judgement = [test_data[test_key]["judgement"] for test_key in test_keys]
        detailed_pass_at_1[i] = sum(judgement) / len(judgement) * 100
        length = [test_data[test_key]["input_len"] + test_data[test_key]["output_len"] for test_key in test_keys]
        avg_length[i] = sum(length) / len(length)

        response_answer = [test_data[test_key]["response_answer"] for test_key in test_keys]
        vote_answer = max(response_answer, key=response_answer.count)
        detailed_voting_judgements[i] = vote_answer == gold

    overall_judgements = [v["judgement"] for k, v in test_data.items()]
    overall_pass_at_1 = sum(overall_judgements) / len(overall_judgements) * 100
    overall_voting_acc = [x for x in detailed_voting_judgements if x is not None]
    overall_voting_acc = sum(overall_voting_acc) / len(overall_voting_acc) * 100

    return (
        overall_pass_at_1,
        overall_voting_acc,
        detailed_pass_at_1,
        detailed_voting_judgements,
        avg_length,
    )


def judge_cmimc(responses_dir, version):
    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
            for k in data.keys():
                data[k].pop("response")
        test_data.update(data)

    if len(test_data) == 16 * 30:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/480")

    ids = [f"cmimc_{version}.{i}" for i in range(1, 31)]
    detailed_pass_at_1 = [0] * 30
    detailed_voting_judgements = [None] * 30
    avg_length = [0] * 30

    for i, idx in enumerate(ids):
        test_keys = [key for key in test_data.keys() if key.startswith(f"{idx}.")]
        if len(test_keys) == 0:
            continue
        gold = test_data[test_keys[0]]["gold"]
        judgement = [test_data[test_key]["judgement"] for test_key in test_keys]
        detailed_pass_at_1[i] = sum(judgement) / len(judgement) * 100
        length = [test_data[test_key]["input_len"] + test_data[test_key]["output_len"] for test_key in test_keys]
        avg_length[i] = sum(length) / len(length)

        response_answer = [test_data[test_key]["response_answer"] for test_key in test_keys]
        vote_answer = max(response_answer, key=response_answer.count)
        detailed_voting_judgements[i] = vote_answer == gold

    overall_judgements = [v["judgement"] for k, v in test_data.items()]
    overall_pass_at_1 = sum(overall_judgements) / len(overall_judgements) * 100
    overall_voting_acc = [x for x in detailed_voting_judgements if x is not None]
    overall_voting_acc = sum(overall_voting_acc) / len(overall_voting_acc) * 100

    return (
        overall_pass_at_1,
        overall_voting_acc,
        detailed_pass_at_1,
        detailed_voting_judgements,
        avg_length,
    )


def judge_livecodebench(responses_dir, version):
    from pm_kvq.evaluation.eval_livecodebench.livecodebench import load_code_generation_dataset
    from pm_kvq.evaluation.eval_livecodebench.evaluator import score_code_generation

    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
        test_data.update(data)
    if len(test_data) == 175 * 4:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/700")
    references = [k.split(".")[0] for k in test_data.keys()]
    predictions = [v["response"] for v in test_data.values()]

    dataset = load_code_generation_dataset(DEFAULT_CODE_GENERATION_DATASET_PATH, version)["test"]
    results = score_code_generation(
        dataset,
        predictions,
        references,
    )
    return results
