"""Offline evaluation integrity and public output contract regressions."""
import copy
import json
import subprocess
import sys
import pytest
from docision.evaluate import build_manifest, score_benchmark, summarize

class Engine:
    def score_many(self, image, requests, **kwargs):
        return [{"candidates": [{"probability": .8}] + [{"probability": .2/(len(q["candidates"])-1)} for _ in q["candidates"][1:]],
                 "decision": {"index": 0}, "num_cached_tokens": 0} for q in requests]

def cohort(protocol="none"):
    questions=[{"id": name, "image": "fixture", "context": "context",
                "question": "Question?", "candidates": ["yes", "no"], "label": "A"} for name in ("a","b")]
    return (score_benchmark(Engine(),questions,protocol=protocol),
            build_manifest(questions,"0"*64,protocol=protocol))

def test_nonidentity_cannot_be_original():
    rows,_=cohort()
    rows[0].update(order=[1,0],target_index=1,semantic_prediction=1)
    with pytest.raises(ValueError,match="original candidate order"):summarize(rows)

def test_candidate_counts_must_agree():
    rows,_=cohort("circular")
    row=rows[1]
    row.update(order=[1,0,2],probabilities=[.8,.1,.1])
    with pytest.raises(ValueError,match="Candidate counts"):summarize(rows)

@pytest.mark.parametrize("protocol",["none","circular","random"])
def test_complete_cohort_coverage_verified(protocol):
    rows,manifest=cohort(protocol)
    coverage=summarize(rows,manifest)["coverage"]
    assert coverage["verified"] and coverage["expected_questions"]==2
    assert coverage["expected_variants"]==len(rows)

@pytest.mark.parametrize("change",["question","variant","extra","duplicate","fingerprint","target","order"])
def test_incomplete_or_altered_predictions_rejected(change):
    rows,manifest=cohort("circular")
    if change=="question":rows=[r for r in rows if r["id"]!="b"]
    elif change=="variant":rows.pop()
    elif change=="extra":
        extra=copy.deepcopy(rows[:2])
        for row in extra:row["id"]="extra"
        rows+=extra
    elif change=="duplicate":rows.append(copy.deepcopy(rows[0]))
    elif change=="fingerprint":rows[0]["question_sha256"]="1"*64
    elif change=="target":
        for row in rows[:2]:
            row["semantic_target"]=1;row["target_index"]=row["order"].index(1)
    elif change=="order":rows[1]["variant"]=3
    with pytest.raises(ValueError):summarize(rows,manifest)

def test_without_manifest_never_claims_verified_coverage():
    rows,_=cohort()
    assert summarize(rows)["coverage"]["verified"] is False

def test_offline_cli_requires_manifest_and_checks_missing_question(tmp_path):
    rows,manifest=cohort()
    prediction=tmp_path/"scores.predictions.jsonl"
    output=tmp_path/"report.json"
    def write(values):prediction.write_text("".join(json.dumps(row)+"\n" for row in values))
    def run(*extra):
        return subprocess.run([sys.executable,"-B","-m","docision.evaluate","--predictions",str(prediction),
                               "--output",str(output),*extra],capture_output=True,text=True)
    write(rows)
    assert run().returncode!=0
    assert run("--allow-unverified-coverage").returncode==0
    assert json.loads(output.read_text())["coverage"]["verified"] is False
    prediction.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    assert run().returncode==0
    assert json.loads(output.read_text())["coverage"]["verified"] is True
    # Delete the question's entire prediction set, not merely a variant.
    output.unlink();write(rows[:1])
    assert run().returncode!=0
    assert not output.exists()

def test_output_contract_checks_keys_not_document_text(assert_public_output):
    assert_public_output({"decision":{"tie_basis":"raw_logits"},"text":"raw_logit token_id"})
    for key in ("raw_logit","raw_logits","token_id"):
        with pytest.raises(AssertionError):
            assert_public_output({"nested":[{key:123}]})

def test_manifest_rejects_duplicate_ids_and_modified_protocol():
    rows,manifest=cohort("circular")
    altered=copy.deepcopy(manifest);altered["questions"].append(altered["questions"][0])
    with pytest.raises(ValueError):summarize(rows,altered)
    altered=copy.deepcopy(manifest);altered["protocol"]="none"
    with pytest.raises(ValueError):summarize(rows,altered)
