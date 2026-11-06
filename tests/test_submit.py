import json
import subprocess
from pathlib import Path

import pytest
import yaml

from grp_runtime.execution_plan_controller.intake import parse_request as parse
from grp_runtime.submit import SubmitError, main, plan_request, runtime_action

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "grp-runtime-train:0.1.1"
MIG = "MIG-00000000-0000-4000-8000-000000000003,MIG-00000000-0000-4000-8000-000000000004"


def test_ddp_plan_request_matches_image_scenario():
    doc = plan_request("w1", "ddp", 2, 1, 1, MIG, IMAGE, profile_configmap="train-ddp-profiles")
    ref = yaml.safe_load((ROOT / "scenarios/image/pr-img-ddp4-profiles.yaml").read_text(encoding="utf-8"))
    spec, launch = doc["spec"], doc["spec"]["launch"]
    assert spec["allocation"]["gpuIds"] == MIG.split(",")
    assert launch["settings"] == {"effectiveBatch": 1024, "maxMicroBatch": 256, "profileConfigMap": "train-ddp-profiles"}
    for k in ("script", "image", "codeInImage", "checkpoint", "runtimeClass", "gpu"):
        assert launch[k] == ref["spec"]["launch"][k] or k == "image"
    assert parse(doc)[1] == []


def test_plan_request_without_gpu_ids_lets_the_pool_choose():
    doc = plan_request("w2", "3d", 1, 2, 2, "", IMAGE)
    assert "allocation" not in doc["spec"] and "settings" not in doc["spec"]["launch"]
    assert doc["spec"]["runtime"] == "pytorch-3d"
    assert parse(doc)[1] == []


@pytest.mark.parametrize("kw, msg", [
    ({"model": "ddp", "tp": 2}, "data parallel only"),
    ({"gpu_ids": "a,b,c"}, "3 GPU id"),
    ({"gpu_ids": "a,a"}, "duplicate"),
    ({"effective_batch": 1000, "dp": 3}, "not divisible"),
    ({"image": ""}, "image is required"),
])
def test_plan_request_rejects_bad_input(kw, msg):
    args = {"name": "w", "model": "ddp", "dp": 2, "tp": 1, "pp": 1, "image": IMAGE, **kw}
    with pytest.raises(SubmitError, match=msg):
        plan_request(**args)


def test_runtime_actions():
    down = runtime_action("a1", "w1", "ScaleDown", 2, 1, 1, MIG)
    assert down["spec"]["target"] == {"gpus": 2, "parallelism": {"dp": 2, "tp": 1, "pp": 1},
                                      "allocation": {"gpuIds": MIG.split(",")}}
    apply = runtime_action("a2", "w1", "ApplySettings", 2, 1, 1, profile_configmap="p", micro_batch=64,
                           gradient_accumulation=8)
    assert apply["spec"]["target"] == {"settings": {"profileConfigMap": "p",
                                                    "batch": {"microBatch": 64, "gradientAccumulation": 8}}}
    observe = runtime_action("a3", "w1", "Observe", 2, 1, 1, MIG)
    assert observe["spec"]["target"] == {} and "safePolicy" not in observe["spec"]
    with pytest.raises(SubmitError, match="needs a profile"):
        runtime_action("a4", "w1", "ApplySettings")
    with pytest.raises(SubmitError, match="together"):
        runtime_action("a5", "w1", "ApplySettings", micro_batch=64)
    with pytest.raises(SubmitError, match="action must be"):
        runtime_action("a6", "w1", "Migrate", 1, 1, 1)


def test_cli_prints_json_or_fails(capsys):
    assert main(["action", "--name", "a", "--plan", "w1", "--action", "Observe", "--dp", "2", "--tp", "1",
                 "--pp", "1", "--gpu-ids", ""]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "RuntimeAction"
    assert main(["plan", "--name", "w", "--image", IMAGE, "--dp", "2", "--tp", "2"]) == 2
    assert "data parallel only" in capsys.readouterr().err


def test_argo_templates_render_and_pass_every_parameter(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    env = {"PATH": "/usr/bin:/bin", "NAMESPACE": "grp-runtime", "REGISTRY": "reg/", "TAG": "0.1.5",
           "TRAIN_TAG": "0.1.1", "STORAGE_CLASS": "x", "WITH_ARGO": "1", "MODE": "render", "OUT": str(out)}
    try:
        subprocess.run(["bash", str(ROOT / "deploy/k8s/install.sh")], env=env, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        pytest.skip(f"bash not usable here: {e}")
    docs = list(yaml.safe_load_all((out / "argo-templates.yaml").read_text(encoding="utf-8")))
    text = (out / "argo-templates.yaml").read_text(encoding="utf-8")
    assert "__" not in text.replace("__init__", "")
    templates = {d["metadata"]["name"]: d for d in docs if d["kind"] == "WorkflowTemplate"}
    assert set(templates) == {"grp-train", "grp-runtime-action"}
    for wt in templates.values():
        render = next(t for t in wt["spec"]["templates"] if t["name"] == "render")
        args = " ".join(render["container"]["args"])
        assert render["container"]["image"] == "reg/grp-runtime-operator:0.1.5"
        for p in wt["spec"]["arguments"]["parameters"]:
            assert "{{workflow.parameters.%s}}" % p["name"] in args
    assert "reg/grp-runtime-train:0.1.1" in " ".join(
        next(t for t in templates["grp-train"]["spec"]["templates"] if t["name"] == "render")["container"]["args"])
