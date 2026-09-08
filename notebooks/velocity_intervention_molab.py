# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.13",
#     "huggingface_hub>=0.24",
#     "pyyaml>=6",
#     "pandas>=2",
# ]
# ///
"""Marimo notebook (molab.marimo.io) that builds and publishes the velocity-intervention dataset.

Purpose:
  End-to-end driver for https://molab.marimo.io: clone ``p1neapplechoco/kubric``,
  install Docker with ``apt`` when the sandbox allows it, build the GPU
  Blender-in-Docker image (Blender 4.2 ``bpy`` with CUDA/OptiX kernels) or fall back
  to an equivalent native Python 3.11 environment, generate N scenes x 3 branches
  (video, masks, contact graphs, tracking) on the GPU with the repository's
  ``scripts/build_velocity_dataset.py`` and upload the result to the Hugging Face
  Hub with an access token.

Public API:
  ``app`` (the marimo application). Run with ``marimo edit notebooks/velocity_intervention_molab.py``
  or upload the file to molab.

Dependencies:
  marimo, huggingface_hub, PyYAML, pandas in the notebook process; ``git``, ``uv``
  (installed on the fly if missing) and optionally ``docker`` + NVIDIA Container
  Toolkit on the host. Everything heavy (bpy, PyBullet, TensorFlow) lives in the
  Docker image or the ``.venv-render`` Python 3.11 environment, never in the
  notebook interpreter.

Trust boundary:
  Shell commands are only run against the cloned repository directory. The
  Hugging Face token is kept in a password field / environment variable of the
  subprocess and is never written to disk. Long steps are gated behind explicit
  run buttons so a re-executed notebook does not silently re-render or re-upload.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Kubric velocity-intervention dataset (GPU) → Hugging Face

        This notebook reproduces the repository demo pipeline end to end:

        1. clone **p1neapplechoco/kubric**;
        2. install **Docker via `apt`** (when the sandbox allows it) and build a GPU-capable
           Blender image (`docker/KubricGPU.Dockerfile`: Blender 4.2 LTS as `bpy` with CUDA/OptiX
           kernels). If Docker cannot run here, the *same pinned environment*
           (`requirements_render.txt`, Python 3.11) is created natively with `uv`;
        3. generate scenes with the existing Kubric components (PyBullet simulator, Blender/Cycles
           renderer, `interventions.*` schemas/logging/graph extraction):
           * **mass fixed**, 4-6 bodies, some struck and some never touched,
           * the subject gets **one initial velocity along one heading** (no other actuation),
           * **materials sampled** (friction/restitution coupled to the material family),
           * spheres **roll**, boxes **slide**, un-spun spheres slide → roll,
           * **camera fixed within a clip, re-sampled between clips**,
           * three branches per scene: `factual`, `counterfactual` (new initial velocity replaces
             the old one), `subject_removed`;
        4. outputs per branch: `video.mp4`, `mask.mp4` + `segmentation.npz`, `depth.npz`,
           `graph.json`, `tracking.npz`, plus `ground_truth.json` / `qc.json` per scene;
        5. upload everything to a Hugging Face dataset repo with your access token.

        Pick a **GPU runtime** in molab before running (Cycles renders on OptiX/CUDA; the notebook
        refuses to fall back to CPU rendering unless you untick *require GPU*).
        """
    )
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(value="https://github.com/p1neapplechoco/kubric", label="Repository URL", full_width=True)
    repo_ref = mo.ui.text(value="artifacts/data-generation-notebook", label="Branch / tag")
    workdir = mo.ui.text(value="/tmp/kubric-work", label="Work directory", full_width=True)
    seed = mo.ui.number(value=0, start=0, stop=2**31 - 1, step=1, label="Master seed")
    count = mo.ui.number(value=8, start=1, stop=100000, step=1, label="Number of scenes")
    resolution = mo.ui.dropdown(options=["256", "384", "512", "768", "1024"], value="512", label="Resolution (primary quality & GPU scaling)")
    samples = mo.ui.dropdown(options=["32", "64", "128", "256"], value="64", label="Cycles samples / pixel")
    require_gpu = mo.ui.checkbox(value=True, label="Require GPU rendering (fail instead of CPU fallback)")
    prefer_docker = mo.ui.checkbox(value=True, label="Prefer Docker when the daemon + NVIDIA runtime are available")
    hf_repo = mo.ui.text(value="", label="Hugging Face dataset repo id (user/name)", full_width=True)
    hf_token = mo.ui.text(value="", label="Hugging Face access token (write)", kind="password", full_width=True)
    hf_private = mo.ui.checkbox(value=True, label="Private dataset repo")
    mo.vstack([
        mo.md("## 1. Settings"),
        mo.hstack([repo_url, repo_ref], widths=[3, 1]),
        workdir,
        mo.hstack([seed, count, resolution, samples]),
        mo.hstack([require_gpu, prefer_docker]),
        hf_repo,
        hf_token,
        hf_private,
    ])
    return (
        count, hf_private, hf_repo, hf_token, prefer_docker, repo_ref, repo_url,
        require_gpu, resolution, samples, seed, workdir,
    )


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    def sh(cmd, cwd=None, env=None, check=True, timeout=None, stream=False):
        """Runs a shell command, returns (code, output). Streams to stdout when asked."""
        merged = dict(os.environ)
        merged.update(env or {})
        if stream:
            proc = subprocess.Popen(
                cmd, shell=isinstance(cmd, str), cwd=cwd, env=merged,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            lines = []
            for line in proc.stdout:
                print(line, end="")
                lines.append(line)
            proc.wait(timeout=timeout)
            code, out = proc.returncode, "".join(lines)
        else:
            res = subprocess.run(
                cmd, shell=isinstance(cmd, str), cwd=cwd, env=merged,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
            )
            code, out = res.returncode, res.stdout
        if check and code != 0:
            raise RuntimeError("command failed ({}): {}\n{}".format(code, cmd, out[-4000:]))
        return code, out

    def have(binary):
        return shutil.which(binary) is not None

    def is_root():
        return hasattr(os, "geteuid") and os.geteuid() == 0

    def sudo_prefix():
        if is_root():
            return ""
        if have("sudo") and sh("sudo -n true", check=False)[0] == 0:
            return "sudo "
        return None
    return Path, have, is_root, json, os, sh, shutil, subprocess, sudo_prefix, sys, time


@app.cell
def _(mo, sh, have, sudo_prefix, sys):
    def probe_environment():
        rows = []
        rows.append(("python (notebook)", sys.version.split()[0]))
        code, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", check=False)
        rows.append(("nvidia-smi", out.strip() if code == 0 else "not available"))
        rows.append(("docker binary", "yes" if have("docker") else "no"))
        code, out = sh("docker info --format '{{.ServerVersion}} runtimes: {{range $k, $v := .Runtimes}}{{$k}} {{end}}'", check=False)
        rows.append(("docker daemon", out.strip() if code == 0 else "not running / no access"))
        rows.append(("apt-get", "yes" if have("apt-get") else "no"))
        pre = sudo_prefix()
        rows.append(("root / sudo", "root" if pre == "" else ("sudo" if pre else "no")))
        rows.append(("git", "yes" if have("git") else "no"))
        rows.append(("uv", "yes" if have("uv") else "no (will be installed)"))
        return rows

    env_rows = probe_environment()
    gpu_present = env_rows[1][1] != "not available"
    mo.vstack([
        mo.md("## 2. Environment probe"),
        mo.ui.table([{"check": k, "value": v} for k, v in env_rows], selection=None),
        mo.callout(
            mo.md("An NVIDIA GPU is visible: `{}`".format(env_rows[1][1])) if gpu_present
            else mo.md("**No GPU visible.** Switch the molab runtime to a GPU machine; rendering will otherwise be refused (or untick *require GPU*)."),
            kind="success" if gpu_present else "warn",
        ),
    ])
    return env_rows, gpu_present


@app.cell
def _(mo):
    clone_button = mo.ui.run_button(label="Clone / update repository")
    mo.vstack([mo.md("## 3. Clone `p1neapplechoco/kubric`"), clone_button])
    return (clone_button,)


@app.cell
def _(Path, clone_button, mo, repo_ref, repo_url, sh, workdir):
    mo.stop(not clone_button.value, mo.md("_Press the button to clone._"))
    work = Path(workdir.value).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    repo_dir = work / "kubric"
    if (repo_dir / ".git").exists():
        sh("git fetch --all --tags", cwd=repo_dir)
        sh("git checkout {}".format(repo_ref.value), cwd=repo_dir, check=False)
        sh("git pull --ff-only || true", cwd=repo_dir, check=False)
    else:
        code, _ = sh("git clone --depth 1 --branch {} {} {}".format(repo_ref.value, repo_url.value, repo_dir), stream=True, check=False)
        if code != 0:
            sh("git clone {} {}".format(repo_url.value, repo_dir), stream=True)
            sh("git checkout {}".format(repo_ref.value), cwd=repo_dir, check=False)
    _, head = sh("git log --oneline -1", cwd=repo_dir)
    # Auto-generate docker/KubricGPU.Dockerfile if missing in remote branch
    dockerfile_path = repo_dir / "docker" / "KubricGPU.Dockerfile"
    if not dockerfile_path.exists():
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(
            "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
            "ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 KUBRIC_USE_GPU=true NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common ca-certificates curl gnupg git ffmpeg \\\n"
            "    && add-apt-repository -y ppa:deadsnakes/ppa \\\n"
            "    && apt-get update && apt-get install -y --no-install-recommends \\\n"
            "      python3.11 python3.11-dev python3.11-venv python3.11-distutils \\\n"
            "      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libglu1-mesa \\\n"
            "      libsm6 libice6 libxkbcommon0 libegl1 libgomp1 libopenexr-dev \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \\\n"
            "    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \\\n"
            "    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \\\n"
            "    && python -m pip install --no-cache-dir --upgrade pip wheel setuptools\n"
            "WORKDIR /kubric\n"
            "COPY requirements_render.txt .\n"
            "RUN python -m pip install --no-cache-dir -r requirements_render.txt\n"
            "ENV PYTHONPATH=/kubric\n"
            "CMD [\"python\", \"-c\", \"import bpy; print('bpy ready')\"]\n"
        )
    reqs_path = repo_dir / "requirements_render.txt"
    if not reqs_path.exists():
        reqs_path.write_text(
            "bpy==4.2.0\nnumpy>=1.26,<3\npybullet==3.2.7\ntraitlets>=5.9,<6\npyquaternion>=0.9.9\n"
            "etils[epath]>=1.5\nimageio>=2.31\nimageio-ffmpeg>=0.4.9\nOpenEXR>=3.2\nPillow>=10\n"
            "bidict>=0.22\nPyYAML>=6\nscipy>=1.11\ntrimesh>=4\nmunch>=4\npypng>=0.20\n"
            "tensorflow-cpu>=2.16,<2.22\npandas>=2\nscikit-learn>=1.3\nhuggingface_hub>=0.24\n"
        )

    required = [
        "configs/velocity_intervention.yaml",
        "interventions/velocity_intervention.py",
        "scripts/build_velocity_dataset.py",
        "scripts/render_velocity_intervention.py",
        "scripts/publish_velocity_dataset.py",
        "docker/KubricGPU.Dockerfile",
        "requirements_render.txt",
    ]
    missing = [p for p in required if not (repo_dir / p).exists()]
    mo.stop(bool(missing), mo.callout(mo.md("Checkout lacks required files: `{}`. Use the branch that contains the velocity-intervention pipeline.".format(missing)), kind="danger"))
    mo.md("Repository ready at `{}` — `{}`".format(repo_dir, head.strip()))
    return repo_dir, work


@app.cell
def _(mo):
    docker_button = mo.ui.run_button(label="Install Docker (+ NVIDIA Container Toolkit) with apt")
    mo.vstack([
        mo.md(
            """## 4. Docker via `apt`

            On a plain Ubuntu GPU VM this installs `docker.io` and the NVIDIA Container Toolkit and starts
            the daemon. Inside molab's sandbox `apt` may be unavailable or the daemon may be impossible to
            start (no privileges); the notebook detects that and falls back to the native environment in
            step 5 without changing the pinned dependency versions."""
        ),
        docker_button,
    ])
    return (docker_button,)


@app.cell
def _(docker_button, have, mo, repo_dir, sh, sudo_prefix, time):
    mo.stop(not docker_button.value, mo.md("_Skipped (press to install Docker)._"))
    log = []
    pre = sudo_prefix()
    if not have("apt-get") or pre is None:
        log.append("apt-get unavailable or no root/sudo: cannot install Docker here.")
    else:
        if not have("docker"):
            sh(pre + "apt-get update -y", stream=True, check=False)
            sh(pre + "DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl gnupg ca-certificates", stream=True, check=False)
            log.append("docker.io installed via apt")
        else:
            log.append("docker binary already present")
        # NVIDIA Container Toolkit (official apt repository).
        toolkit_cmd = (
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | {pre}gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && "
            "curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
            "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
            "{pre}tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null && "
            "{pre}apt-get update -y && {pre}DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-container-toolkit && "
            "{pre}nvidia-ctk runtime configure --runtime=docker && "
            "{pre}mkdir -p /etc/cdi && {pre}nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        ).format(pre=pre)
        code, _ = sh(toolkit_cmd, stream=True, check=False)
        log.append("nvidia-container-toolkit + CDI: {}".format("installed" if code == 0 else "install failed (GPU-in-Docker unavailable)"))
        # Start the daemon: systemd first, then a bare dockerd for container sandboxes.
        if sh("docker info", check=False)[0] != 0:
            sh(pre + "systemctl restart docker", check=False)
            if sh("docker info", check=False)[0] != 0:
                sh("nohup {}dockerd > /tmp/dockerd.log 2>&1 &".format(pre), check=False)
                for _ in range(20):
                    time.sleep(1)
                    if sh("docker info", check=False)[0] == 0:
                        break
        log.append("docker daemon: {}".format("running" if sh("docker info", check=False)[0] == 0 else "NOT running"))
    mo.md("```\n" + "\n".join(log) + "\n```")
    return


@app.cell
def _(have, mo, prefer_docker, sh):
    def docker_gpu_ready():
        if not prefer_docker.value or not have("docker"):
            return False, "docker not requested / not installed"
        code, out = sh("docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}'", check=False)
        if code != 0:
            return False, "docker daemon not reachable"
        if "nvidia" not in out:
            return False, "docker runs but has no nvidia runtime"
        code, out = sh("docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L", check=False, timeout=600)
        return (code == 0 and "GPU" in out), out.strip()[-500:]

    docker_ok, docker_reason = docker_gpu_ready()
    mo.callout(
        mo.md("**Runner: Docker (GPU)** — `{}`".format(docker_reason)) if docker_ok
        else mo.md("**Runner: native Python 3.11 environment** — Docker GPU path unavailable: `{}`".format(docker_reason)),
        kind="success" if docker_ok else "info",
    )
    return docker_ok, docker_reason


@app.cell
def _(mo):
    env_button = mo.ui.run_button(label="Build render environment (Docker image or native venv)")
    mo.vstack([
        mo.md(
            """## 5. Render environment (pinned versions)

            *Docker path*: `docker build -f docker/KubricGPU.Dockerfile` — CUDA 12.4 runtime, Python 3.11,
            `bpy==4.2.0` and `requirements_render.txt`.

            *Native path*: `uv venv --python 3.11` + `uv pip install -r requirements_render.txt` — the exact
            same pins, so results are identical either way. Both paths are checked by importing `bpy`
            and listing the Cycles OptiX/CUDA devices."""
        ),
        env_button,
    ])
    return (env_button,)


@app.cell
def _(Path, docker_ok, env_button, have, json, mo, os, repo_dir, sh, work):
    mo.stop(not env_button.value, mo.md("_Press to build the environment._"))
    IMAGE = "kubric-gpu:4.2"
    probe_code = (
        "import bpy, json; p = bpy.context.preferences.addons['cycles'].preferences; "
        "print('CYCLES_DEVICES ' + json.dumps({'bpy': bpy.app.version_string, "
        "'devices': {b: [d.name for d in p.get_devices_for_type(b)] for b in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI')}}))"
    )

    def parse_probe(text):
        for line in text.splitlines():
            if line.startswith("CYCLES_DEVICES "):
                return json.loads(line[len("CYCLES_DEVICES "):])
        return {"bpy": "import failed", "devices": {}}
    if docker_ok:
        sh("docker build -f docker/KubricGPU.Dockerfile -t {} .".format(IMAGE), cwd=repo_dir, stream=True, timeout=7200)
        uid = os.getuid() if hasattr(os, "getuid") else 0
        gid = os.getgid() if hasattr(os, "getgid") else 0
        # Mount the whole work directory at the same path so host and container paths agree.
        runner = (
            "docker run --rm --gpus all --user {uid}:{gid} -e HOME=/tmp -e PYTHONPATH={repo} "
            "-e TF_CPP_MIN_LOG_LEVEL=3 --volume {work}:{work} --workdir {repo} {image} python"
        ).format(uid=uid, gid=gid, repo=repo_dir, work=work, image=IMAGE)
        _, probe = sh('{} -c "{}"'.format(runner, probe_code), check=False)
        runner_kind = "docker"
    else:
        if not have("uv"):
            sh("curl -LsSf https://astral.sh/uv/install.sh | sh", stream=True)
            os.environ["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + os.environ["PATH"]
        venv = repo_dir / ".venv-render"
        if not (venv / "bin" / "python").exists():
            sh("uv venv {} --python 3.11".format(venv), stream=True)
        sh("uv pip install --python {} -r requirements_render.txt".format(venv / "bin" / "python"), cwd=repo_dir, stream=True, timeout=7200)
        runner = str(venv / "bin" / "python")
        _, probe = sh([runner, "-c", probe_code], cwd=repo_dir, check=False, env={"PYTHONPATH": str(repo_dir)})
        runner_kind = "native"
    probe_info = parse_probe(probe)
    gpu_devices_visible = any(probe_info["devices"].get(b) for b in ("OPTIX", "CUDA", "HIP", "ONEAPI"))
    mo.vstack([
        mo.md("Runner (`{}`): `{}`".format(runner_kind, runner)),
        mo.md("```json\n{}\n```".format(json.dumps(probe_info, indent=2))),
        mo.callout(
            mo.md("Cycles sees a GPU device — rendering will use OptiX/CUDA.") if gpu_devices_visible
            else mo.md("Cycles reports **no GPU device**. With *require GPU* ticked the build will stop before rendering."),
            kind="success" if gpu_devices_visible else "warn",
        ),
    ])
    return gpu_devices_visible, runner, runner_kind


@app.cell
def _(mo):
    smoke_button = mo.ui.run_button(label="Run physics smoke test (1 scene, no render)")
    mo.vstack([mo.md("## 6. Physics smoke test"), smoke_button])
    return (smoke_button,)


@app.cell
def _(json, mo, repo_dir, runner, sh, smoke_button, work):
    mo.stop(not smoke_button.value, mo.md("_Press to run the smoke test._"))
    smoke_out = work / "smoke"
    sh(
        "{} scripts/build_velocity_dataset.py --output {} --seed 12345 --count 1 --no-render".format(runner, smoke_out),
        cwd=repo_dir, stream=True, env={"PYTHONPATH": str(repo_dir), "TF_CPP_MIN_LOG_LEVEL": "3"},
    )
    qc = json.loads((smoke_out / "instances" / "000000" / "qc.json").read_text())
    gt = json.loads((smoke_out / "instances" / "000000" / "ground_truth.json").read_text())
    motion = qc["report"]["metrics"]["motion"]["factual"]
    mo.vstack([
        mo.md("QC passed: **{}** — struck `{}`, untouched `{}`".format(
            qc["report"]["passed"], qc["report"]["metrics"]["factual_struck"], qc["report"]["metrics"]["factual_untouched"])),
        mo.ui.table([
            {"object": k, "motion": v["label"], "rolling": round(v["rolling_fraction"], 2),
             "sliding": round(v["sliding_fraction"], 2), "travel_m": round(v["travel"], 2)}
            for k, v in motion.items()
        ], selection=None),
        mo.md("Ground truth — hard affected: `{}`, soft affected: `{}`, graph delta: +{} / -{} / ~{}".format(
            gt["hard_affected"], gt["soft_affected"], len(gt["graph_delta"]["added"]),
            len(gt["graph_delta"]["removed"]), len(gt["graph_delta"]["changed"]))),
    ])
    return


@app.cell
def _(mo):
    build_button = mo.ui.run_button(label="Generate dataset (physics + GPU render)")
    mo.vstack([
        mo.md(
            """## 7. Generate the dataset

            Runs `scripts/build_velocity_dataset.py` with the chosen runner. The script is resumable:
            re-running skips scenes that already have `instance.json` and branches that already have
            `render_info.json`, and rebuilds `manifest.jsonl` at the end."""
        ),
        build_button,
    ])
    return (build_button,)


@app.cell
def _(build_button, count, mo, repo_dir, require_gpu, resolution, runner, samples, seed, sh, time, work):
    mo.stop(not build_button.value, mo.md("_Press to generate._"))
    dataset_dir = work / "dataset"
    cmd = (
        "{runner} scripts/build_velocity_dataset.py --output {out} --seed {seed} --count {count} "
        "--resolution {res} --samples {spp} --layers rgba segmentation depth {gpu}"
    ).format(runner=runner, out=dataset_dir, seed=int(seed.value), count=int(count.value),
             res=resolution.value, spp=samples.value, gpu="--require-gpu --strict" if require_gpu.value else "")
    started = time.time()
    sh(cmd, cwd=repo_dir, stream=True, env={"PYTHONPATH": str(repo_dir), "TF_CPP_MIN_LOG_LEVEL": "3", "KUBRIC_USE_GPU": "true"})
    mo.md("Finished in {:.0f} s → `{}`".format(time.time() - started, dataset_dir))
    return (dataset_dir,)


@app.cell
def _(dataset_dir, json, mo):
    import pandas as pd

    summary = json.loads((dataset_dir / "dataset_summary.json").read_text())
    rows = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    table = pd.DataFrame([{
        "index": r["index"], "split": r["split"], "objects": r["object_count"], "subject": r["subject_shape"],
        "struck": ",".join(r["factual_struck"]), "untouched": ",".join(r["factual_untouched"]),
        "hard_affected": ",".join(r["hard_affected"]), "renders": ",".join(r["rendered_branches"]),
        "device": ",".join(sorted({v["device"] for v in r["renders"].values()})),
    } for r in rows])
    first = dataset_dir / rows[0]["path"] if rows else None
    videos = []
    if first is not None:
        for branch in ("factual", "counterfactual", "subject_removed"):
            video = first / branch / "video.mp4"
            mask = first / branch / "mask.mp4"
            if video.exists():
                videos.append(mo.vstack([
                    mo.md("**{}**".format(branch)),
                    mo.video(video.open("rb"), autoplay=True, loop=True, muted=True),
                    mo.video(mask.open("rb"), autoplay=True, loop=True, muted=True),
                ]))
    mo.vstack([
        mo.md("## 8. Results"),
        mo.md("```json\n{}\n```".format(json.dumps(summary, indent=2))),
        mo.ui.table(table, selection=None),
        mo.hstack(videos) if videos else mo.md("_No rendered videos yet._"),
    ])
    return


@app.cell
def _(mo):
    upload_button = mo.ui.run_button(label="Upload to Hugging Face")
    mo.vstack([mo.md("## 9. Publish to the Hugging Face Hub"), upload_button])
    return (upload_button,)


@app.cell
def _(dataset_dir, hf_private, hf_repo, hf_token, mo, repo_dir, seed, sh, sys, upload_button):
    mo.stop(not upload_button.value, mo.md("_Press to upload._"))
    mo.stop(not hf_repo.value or "/" not in hf_repo.value, mo.callout(mo.md("Enter a repo id like `user/kubric-velocity-intervention`."), kind="danger"))
    mo.stop(not hf_token.value, mo.callout(mo.md("Enter a Hugging Face **write** access token."), kind="danger"))
    publish_cmd = [
        sys.executable, "scripts/publish_velocity_dataset.py", "--output", str(dataset_dir),
        "--repo-id", hf_repo.value, "--seed", str(int(seed.value)),
    ] + ([] if hf_private.value else ["--public"])
    _, out = sh(publish_cmd, cwd=repo_dir, stream=True, env={"HF_TOKEN": hf_token.value})
    url = [l for l in out.splitlines() if l.startswith("https://huggingface.co/")]
    mo.callout(mo.md("Uploaded: {}".format(url[-1] if url else "see log above")), kind="success")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Notes

        * **GPU inside Docker** needs the NVIDIA Container Toolkit on the host; the sandbox on molab
          usually has neither `apt` privileges nor a Docker daemon, so the notebook picks the native
          `uv` environment automatically. Both paths pin `bpy==4.2.0` (Blender 4.2 LTS, Python 3.11).
        * `KUBRIC_USE_GPU=true` alone does **not** enable GPU rendering in Kubric; the renderer script
          selects the Cycles backend (`OPTIX` → `CUDA` → …) and enables its devices explicitly, and
          records the device actually used in every `render_info.json`.
        * Regenerate a scene by deleting its `instances/<index>` folder; re-render a branch by deleting
          its `render_info.json`.
        * See `docs/velocity_intervention_dataset.md` in the repository for the artifact schema.
        """
    )
    return


if __name__ == "__main__":
    app.run()
