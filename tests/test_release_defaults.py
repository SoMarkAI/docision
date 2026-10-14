import base64
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pytest
from docision import Docision
from docision.local_app.runtime import score_file

def image_body():
    stream=io.BytesIO();Image.new("RGB",(20,20),"white").save(stream,format="PNG")
    return {"name":"page.png","data":base64.b64encode(stream.getvalue()).decode(),
            "questions":[{"question":"q","candidates":["yes","no"]} for _ in range(2)]}

def test_defaults_are_native_serial_without_cache(tmp_path):
    class Engine:
        _numerics_mode="native"
        def prepare_context(self,*a,**k):raise AssertionError("default must not build cache")
        def score_document(self,document):
            assert len(document["questions"])==1
            return [{"decision":{"text":"yes"},"num_cached_tokens":0}]
    result=score_file(image_body(),Engine(),tmp_path)
    assert len(result["rows"])==2
    assert all(r["cache_status"]=="disabled" and r["numerics"]=="native" and r["inference_batch_size"]==1 for r in result["rows"])
    assert inspect.signature(Docision.inspect).parameters["prefix_cache"].default is False
    assert inspect.signature(Docision).parameters["numerics"].default=="native"

def test_native_cache_request_rejected_before_inference(tmp_path):
    class Engine:
        _numerics_mode="native"
    body=image_body();body["prefix_cache"]=True
    with pytest.raises(ValueError,match="numerics=stable"):score_file(body,Engine(),tmp_path)

def test_python_profile_forwarded_to_loader(tmp_path):
    with patch("docision.api.load_model",return_value=(object(),("model","cpu"),{})) as loader:
        with Docision("model",device="cpu",state_dir=tmp_path,numerics="stable"):pass
    assert loader.call_args.args[0]["numerics"]=="stable"

def test_openapi_matches_false_cache_default():
    from importlib.resources import files
    schema=json.loads(files("docision.local_app").joinpath("assets/openapi.json").read_text())
    assert schema["components"]["schemas"]["Document"]["properties"]["prefix_cache"]["default"] is False
    assert schema["paths"]["/api/v1/model"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["numerics"]["default"]=="native"

@pytest.mark.parametrize("extra",[["--max-batch-size","2"],["--prefix-cache"]])
def test_image_service_requires_stable_for_cache_or_batch(extra):
    p=subprocess.run([sys.executable,"-B","-m","docision.service","--model","unused",*extra],capture_output=True,text=True)
    assert p.returncode!=0
    assert "--numerics stable" in p.stderr

def test_openapi_generator_is_reproducible(tmp_path):
    import shutil
    root=Path(__file__).resolve().parents[1]
    script=root/"deployment/integrations/make_openapi.py"
    if not script.exists():pytest.skip("source-only generator")
    target=tmp_path/"deployment/integrations/make_openapi.py";target.parent.mkdir(parents=True)
    shutil.copy2(script,target)
    output=tmp_path/"docision/local_app/assets";output.mkdir(parents=True)
    subprocess.run([sys.executable,"-B",str(target)],check=True)
    assert (output/"openapi.json").read_bytes()==(root/"docision/local_app/assets/openapi.json").read_bytes()
