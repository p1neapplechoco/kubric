`# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "pyyaml",
#     "pybullet",
#     "imageio",
#     "imageio-ffmpeg",
#     "OpenEXR",
#     "trimesh",
#     "tensorflow-cpu",
#     "huggingface-hub",
#     "bpy==4.2.0",
# ]
# ///
"""Generate a velocity-intervention dataset and publish it to HuggingFace.

Runs on https://molab.marimo.io or on any local machine with a GPU. Clones the
Kubric fork, renders three branches per scene with Blender/Cycles, and uploads
Video/Mask/Graph/Tracking to a HuggingFace dataset repo.

Read the "Rendering backend" cell before running: Docker is supported and is
used when a daemon is reachable, but hosted notebook sandboxes do not provide
one, so the notebook falls back to running Blender in-process.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Velocity-intervention dataset

        Each **instance** is one physics scene rendered **three times** from a
        single shared appearance record — same bodies, materials, textures,
        lights, background, and camera. Only the intervention differs:

        | Branch | What it is |
        | --- | --- |
        | `factual` | The subject gets one initial linear velocity at `t=0`, then is released. |
        | `counterfactual` | The subject's initial velocity is **replaced** — old velocity discarded, a new speed and turned direction applied once at `t=0`. |
        | `subject_removed` | The same scene with the subject absent. |

        Because the three branches share one `VisualSceneSpec`, any pixel
        difference between them is caused by the intervention and not by
        re-sampled appearance.

        **Scene configuration** (`configs/scene_ranges_velocity.yaml`):
        fixed mass · one-direction initial velocity only · sampled materials ·
        4–6 bodies, some struck and some untouched · both sliding and rolling ·
        camera fixed within a clip and resampled between clips.

        **Outputs per branch:** `video.mp4`, `segmentation.npz` (mask),
        `graph.json` (temporal contact graph), `tracking.npz`, `depth.npz`.
        """
    )
    return


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, os, shutil, subprocess, sys, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Environment""")
    return


@app.cell
def _(Path, os, shutil, subprocess, sys):
    def _run(command, **kwargs):
        """Runs a probe command and returns (ok, output) instead of raising."""
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=kwargs.pop("timeout", 60),
                **kwargs,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return False, str(error)
        return done.returncode == 0, (done.stdout + done.stderr).strip()

    def probe_environment():
        """What this machine can actually do, checked rather than assumed."""
        gpu_ok, gpu_out = (False, "")
        if shutil.which("nvidia-smi"):
            gpu_ok, gpu_out = _run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"]
            )
        docker_ok, docker_out = (False, "docker not on PATH")
        if shutil.which("docker"):
            # `docker info` and not `docker --version`: the CLI can be present
            # with no daemon behind it, which is exactly the hosted-notebook case.
            docker_ok, docker_out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
        return {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
            "gpu": gpu_out.splitlines()[0] if (gpu_ok and gpu_out) else None,
            "gpu_available": bool(gpu_ok and gpu_out),
            "docker_available": bool(docker_ok),
            "docker_detail": docker_out.splitlines()[-1] if docker_out else "",
            "in_molab": "MARIMO_CLOUD" in os.environ or "molab" in os.environ.get("HOSTNAME", ""),
        }

    env = probe_environment()
    return env, probe_environment


@app.cell(hide_code=True)
def _(env, mo):
    mo.md(
        f"""
        | | |
        | --- | --- |
        | Python | `{env["python"]}` on `{env["platform"]}` |
        | GPU | {"`" + env["gpu"] + "`" if env["gpu_available"] else "**none detected** — rendering will use the CPU and be roughly 3.5x slower"} |
        | Docker | {"`" + env["docker_detail"] + "`" if env["docker_available"] else "**unavailable** — " + env["docker_detail"]} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Clone the repository

        Kubric plus the `interventions/` extension that this dataset is built
        from. Cloning is idempotent: an existing checkout is left alone.
        """
    )
    return


@app.cell
def _(mo):
    repo_url = mo.ui.text(
        value="https://github.com/p1neapplechoco/kubric",
        label="Repository",
        full_width=True,
    )
    repo_branch = mo.ui.text(value="", label="Branch (blank = default)")
    repo_dir_input = mo.ui.text(value="kubric", label="Clone into")
    clone_button = mo.ui.run_button(label="Clone / verify repository")
    mo.vstack([repo_url, repo_branch, repo_dir_input, clone_button])
    return clone_button, repo_branch, repo_dir_input, repo_url


@app.cell
def _(Path, repo_dir_input):
    # Same reason as `dataset_dir` below: the cell that does the cloning starts
    # with `mo.stop`, so anything it defined would not exist until the button
    # was pressed, and every later cell would fail with a NameError instead of
    # waiting quietly. The path itself needs no clone to be known.
    repo_dir = Path(repo_dir_input.value).resolve()
    return (repo_dir,)


@app.cell
def _(clone_button, mo, repo_branch, repo_dir, repo_url, subprocess, sys):
    mo.stop(not clone_button.value, mo.md("*Press the button to clone.*"))

    _log = []
    if (repo_dir / ".git").is_dir():
        _log.append(f"Existing checkout at `{repo_dir}` — left as is.")
    else:
        _cmd = ["git", "clone", "--depth", "1"]
        if repo_branch.value.strip():
            _cmd += ["--branch", repo_branch.value.strip()]
        _cmd += [repo_url.value, str(repo_dir)]
        _done = subprocess.run(_cmd, capture_output=True, text=True)
        if _done.returncode != 0:
            raise RuntimeError(f"clone failed:\n{_done.stderr}")
        _log.append(f"Cloned into `{repo_dir}`.")

    # Importable from this kernel, so the notebook can call the repo's modules
    # directly rather than only through subprocesses.
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    _required = [
        "interventions/velocity_scenes.py",
        "scripts/generate_velocity_dataset.py",
        "scripts/render_velocity_scene.py",
        "scripts/upload_velocity_dataset.py",
        "configs/scene_ranges_velocity.yaml",
    ]
    _missing = [name for name in _required if not (repo_dir / name).is_file()]
    if _missing:
        raise RuntimeError(
            "checkout is missing the velocity pipeline: "
            + ", ".join(_missing)
            + " — is this the right branch?"
        )
    _log.append(f"All {len(_required)} pipeline files present.")
    mo.md("\n\n".join(f"- {line}" for line in _log))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Rendering backend

        The request was to run Blender **inside Docker** on the GPU. That works
        wherever a Docker daemon is reachable and `nvidia-container-toolkit` is
        installed, and the notebook uses it when it finds one.

        Hosted notebook sandboxes — molab included — do not expose a Docker
        daemon to the notebook process, so on molab the Docker option is
        unavailable and the notebook runs Blender **in-process** through the pip
        `bpy` wheel instead. Both paths execute the identical
        `scripts/generate_velocity_dataset.py` and produce byte-comparable
        outputs; the difference is only where Blender lives.

        GPU selection is handled explicitly either way. Kubric's own
        `KUBRIC_USE_GPU` sets `scene.cycles.device = "GPU"` but never sets
        `preferences.compute_device_type`, so on a fresh `bpy` the backend stays
        `NONE`, the device list is empty, and Cycles silently falls back to the
        CPU. `render_velocity_scene.enable_gpu()` selects OPTIX/CUDA *after* the
        renderer is constructed — before that point Blender's
        `read_factory_settings` discards it — and reports which device it got.
        """
    )
    return


@app.cell
def _(env, mo):
    backend_choice = mo.ui.radio(
        options={
            "Auto (Docker if a daemon is reachable, else in-process)": "auto",
            "Docker (kubricdockerhub/kubruntudev)": "docker",
            "In-process (pip bpy)": "inprocess",
        },
        value="Auto (Docker if a daemon is reachable, else in-process)",
        label="Backend",
    )
    docker_image = mo.ui.text(
        value="kubricdockerhub/kubruntudev:latest", label="Docker image", full_width=True
    )
    device_choice = mo.ui.radio(
        options=["GPU", "CPU"],
        value="GPU" if env["gpu_available"] else "CPU",
        label="Cycles device",
        inline=True,
    )
    mo.vstack([backend_choice, docker_image, device_choice])
    return backend_choice, device_choice, docker_image


@app.cell
def _(backend_choice, env, mo):
    resolved_backend = backend_choice.value
    if resolved_backend == "auto":
        resolved_backend = "docker" if env["docker_available"] else "inprocess"

    # Reported, not raised. Raising here would leave `resolved_backend`
    # undefined, and every cell below would show a NameError instead of the
    # actual problem.
    backend_error = (
        "Docker was selected but no daemon is reachable: "
        f"{env['docker_detail']}. Choose the in-process backend, or run this "
        "notebook on a machine with Docker and nvidia-container-toolkit."
        if resolved_backend == "docker" and not env["docker_available"]
        else None
    )
    mo.md(
        f"Backend: **{resolved_backend}**"
        if backend_error is None
        else f"Backend: **{resolved_backend}** — ⚠️ {backend_error}"
    )
    return backend_error, resolved_backend


@app.cell
def _(mo, resolved_backend):
    pull_button = mo.ui.run_button(
        label="Pull the Docker image",
        disabled=resolved_backend != "docker",
    )
    mo.vstack([
        pull_button,
        mo.md(
            "*The image is several GB. `docker run` would pull it implicitly, "
            "but silently and with no progress, so a first run looks like a "
            "hang; pulling here makes the download visible and separates a "
            "network failure from a render failure.*"
            if resolved_backend == "docker"
            else "*Not applicable to the in-process backend.*"
        ),
    ])
    return (pull_button,)


@app.cell
def _(docker_image, mo, pull_button, subprocess):
    mo.stop(not pull_button.value, mo.md("*Not run.*"))

    _pull = subprocess.run(
        ["docker", "pull", docker_image.value],
        capture_output=True,
        text=True,
    )
    if _pull.returncode != 0:
        raise RuntimeError(
            f"docker pull failed:\n{(_pull.stdout + _pull.stderr)[-3000:]}"
        )
    # The digest is what a rerun can be pinned to; a tag can be moved under you.
    _digest = subprocess.run(
        ["docker", "image", "inspect", docker_image.value,
         "--format", "{{index .RepoDigests 0}}"],
        capture_output=True, text=True,
    )
    mo.md(
        f"Pulled `{docker_image.value}`.\n\n"
        f"Digest: `{_digest.stdout.strip() or 'unavailable'}`\n\n"
        f"```\n{(_pull.stdout + _pull.stderr).strip()[-1500:]}\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Dependencies

        Only needed for the in-process backend; the Docker image already carries
        Blender and Kubric's runtime.
        """
    )
    return


@app.cell
def _(mo, resolved_backend):
    install_button = mo.ui.run_button(
        label="Install in-process dependencies (bpy, kubric runtime)",
        disabled=resolved_backend != "inprocess",
    )
    mo.vstack([
        install_button,
        mo.md(
            "*Not needed for the Docker backend.*"
            if resolved_backend != "inprocess"
            else "*On molab the `/// script` header at the top of this file "
                 "usually installs these already; this button is the manual "
                 "fallback.*"
        ),
    ])
    return (install_button,)


@app.cell
def _(install_button, mo, subprocess, sys):
    mo.stop(not install_button.value, mo.md("*Not run.*"))
    _packages = [
        "bpy==4.2.0", "imageio", "imageio-ffmpeg", "pybullet", "OpenEXR",
        "trimesh", "pyyaml", "scipy", "huggingface-hub",
    ]
    _done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *_packages],
        capture_output=True, text=True,
    )
    mo.md(
        f"Install exit code `{_done.returncode}`.\n\n```\n{_done.stdout[-1500:]}{_done.stderr[-1500:]}\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Batch settings""")
    return


@app.cell
def _(mo):
    num_instances = mo.ui.slider(
        1, 200, value=8, label="Instances", show_value=True
    )
    master_seed = mo.ui.number(value=20260908, label="Master seed")
    width_input = mo.ui.number(value=512, label="Width")
    height_input = mo.ui.number(value=384, label="Height")
    spp_input = mo.ui.slider(
        8, 128, value=48, step=8, label="Samples per pixel", show_value=True
    )
    output_name = mo.ui.text(value="output/velocity_dataset", label="Output directory")
    resume_toggle = mo.ui.checkbox(value=True, label="Resume (skip already-rendered instances)")

    mo.vstack([
        num_instances,
        master_seed,
        mo.hstack([width_input, height_input], justify="start"),
        spp_input,
        output_name,
        resume_toggle,
    ])
    return (
        height_input,
        master_seed,
        num_instances,
        output_name,
        resume_toggle,
        spp_input,
        width_input,
    )


@app.cell(hide_code=True)
def _(device_choice, height_input, mo, num_instances, spp_input, width_input):
    # A separate cell on purpose: a widget's .value read inside the cell that
    # creates it never updates, because that cell does not re-run when the
    # widget changes. Here the estimate tracks the sliders.
    _per_instance = 6.0 if device_choice.value == "GPU" else 20.0
    # 48 spp at 512x384 is what the 6 min was measured at; Cycles cost is close
    # enough to linear in both to make this a usable estimate, not a promise.
    _scale = (
        (int(spp_input.value) / 48.0)
        * (int(width_input.value) * int(height_input.value)) / (512.0 * 384.0)
    )
    _total = _per_instance * _scale * int(num_instances.value)
    mo.md(
        f"Budget on **{device_choice.value}**: about "
        f"**{_per_instance * _scale:.1f} min** per instance for all three "
        f"branches, so **{_total / 60:.1f} h** for "
        f"{int(num_instances.value)} instances.\n\n"
        f"The 6 min/instance baseline was measured at 512x384 / 48 spp on an "
        f"RTX 3500 Ada with OPTIX; the CPU figure scales it by the 3.5x gap "
        f"measured on the same scenes. Rendering dominates, so samples per "
        f"pixel is the lever — and resume means a stopped run is not lost."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Generate""")
    return


@app.cell
def _(
    device_choice,
    height_input,
    master_seed,
    num_instances,
    output_name,
    resume_toggle,
    spp_input,
    width_input,
):
    def generator_arguments():
        """The CLI the two backends both run, so they cannot drift apart."""
        args = [
            "-m", "scripts.generate_velocity_dataset",
            "--config", "configs/scene_ranges_velocity.yaml",
            "--output", output_name.value,
            "--seed", str(int(master_seed.value)),
            "--num-instances", str(int(num_instances.value)),
            "--resolution", str(int(width_input.value)), str(int(height_input.value)),
            "--samples-per-pixel", str(int(spp_input.value)),
            "--device", device_choice.value,
        ]
        if resume_toggle.value:
            args.append("--resume")
        return args

    return (generator_arguments,)


@app.cell
def _(docker_image, generator_arguments, repo_dir, resolved_backend, sys):
    def build_command():
        """The exact command that will run, shown before it runs."""
        if resolved_backend == "docker":
            return [
                "docker", "run", "--rm",
                "--gpus", "all",
                "-v", f"{repo_dir}:/workspace",
                "-w", "/workspace",
                # Blender's scratch and OptiX cache want a writable HOME.
                "-e", "HOME=/tmp",
                "-e", "KUBRIC_USE_GPU=true",
                "--user", "root",
                docker_image.value,
                "python3", *generator_arguments(),
            ]
        return [sys.executable, *generator_arguments()]

    generation_command = build_command()
    return build_command, generation_command


@app.cell(hide_code=True)
def _(generation_command, mo, repo_dir):
    mo.md(
        f"""
        Working directory: `{repo_dir}`

        ```bash
        {" ".join(generation_command)}
        ```
        """
    )
    return


@app.cell
def _(output_name, repo_dir):
    # Defined here rather than inside the run cell below. That cell begins with
    # `mo.stop`, which raises, so anything it defines does not exist until the
    # button is pressed -- and every cell downstream would fail with "name
    # dataset_dir is not defined" on a freshly opened notebook. The path is a
    # pure function of two widgets, so it can always be known. It also means
    # sections 7 and 8 work against a batch generated in an earlier session,
    # without re-rendering it.
    dataset_dir = (repo_dir / output_name.value).resolve()
    return (dataset_dir,)


@app.cell
def _(mo):
    generate_button = mo.ui.run_button(label="Generate dataset")
    generate_button
    return (generate_button,)


@app.cell
def _(backend_error, dataset_dir, generate_button, generation_command, mo, repo_dir, subprocess, time):
    mo.stop(not generate_button.value, mo.md("*Press **Generate dataset** to start.*"))
    mo.stop(backend_error is not None, mo.md(f"**Cannot run:** {backend_error}"))
    mo.stop(
        not (repo_dir / "scripts" / "generate_velocity_dataset.py").is_file(),
        mo.md(f"**Cannot run:** no checkout at `{repo_dir}` — run section 2 first."),
    )

    _started = time.perf_counter()
    _process = subprocess.run(
        generation_command,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    _elapsed = time.perf_counter() - _started

    if _process.returncode not in (0, 2):
        raise RuntimeError(
            f"generation failed (exit {_process.returncode}):\n"
            f"{_process.stderr[-4000:]}"
        )

    mo.md(
        f"""
        Finished in **{_elapsed / 60:.1f} min** (exit `{_process.returncode}`
        {"— batch incomplete, see the log" if _process.returncode == 2 else ""}).

        Output: `{dataset_dir}`

        ```
        {_process.stderr[-3000:]}
        ```

        ```json
        {_process.stdout[-2000:]}
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Inspect what was produced""")
    return


@app.cell
def _(dataset_dir, json):
    # ``None`` rather than a raise or an ``mo.stop``: this name is read by every
    # cell below, so it has to exist even when there is nothing to read yet.
    _manifest_path = dataset_dir / "manifest.json"
    manifest, manifest_error = None, None
    if _manifest_path.is_file():
        try:
            manifest = json.loads(_manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as _error:
            # The generator writes this file in one non-atomic call, so a run
            # killed at the wrong moment leaves a truncated one. Reporting that
            # beats a raw decode traceback in a cell nobody pressed.
            manifest_error = f"`{_manifest_path}` is not readable JSON: {_error}"
    return manifest, manifest_error


@app.cell
def _(dataset_dir, manifest, manifest_error, mo):
    mo.stop(
        manifest_error is not None,
        mo.md(
            f"**Corrupt manifest.** {manifest_error}\n\nRe-run section 6; "
            "`--resume` keeps every instance already rendered."
        ),
    )
    mo.stop(
        manifest is None,
        mo.md(
            f"*No `manifest.json` under `{dataset_dir}` yet — run section 6, or "
            "point **Output directory** at a batch generated earlier.*"
        ),
    )
    mo.stop(
        not manifest["instances"],
        mo.md(
            "*The batch was written but accepted no instances. "
            f"{len(manifest.get('rejections') or [])} attempts were rejected; "
            "`manifest.json` records why.*"
        ),
    )

    _rows = [
        {
            "instance": entry["instance_id"][-8:],
            "split": entry["split"],
            "bodies": entry["num_objects"] - 1,
            "subject": entry["subject_id"],
            "render_s": entry.get("render_seconds"),
            "contact_edges": (entry.get("contact_edges") or {}).get("factual"),
        }
        for entry in manifest["instances"]
    ]
    mo.vstack([
        mo.md(
            f"**{manifest['accepted']}** instances from {manifest['attempts']} "
            f"attempts · splits {manifest['splits']['counts']} · "
            f"{manifest['wall_seconds'] / 60:.1f} min"
        ),
        mo.ui.table(_rows, selection=None, page_size=10),
    ])
    return


@app.cell
def _(manifest, mo):
    _ids = [entry["instance_id"] for entry in (manifest or {}).get("instances", ())]
    # An empty dropdown is still a dropdown, so `preview_instance` exists for
    # the cells below even before anything has been generated.
    preview_instance = mo.ui.dropdown(
        options=_ids,
        value=_ids[0] if _ids else None,
        label="Instance to preview",
    )
    preview_instance if _ids else mo.md("*Nothing to preview yet.*")
    return (preview_instance,)


@app.cell
def _(dataset_dir, mo, preview_instance):
    mo.stop(preview_instance.value is None, mo.md(""))
    _root = dataset_dir / "instances" / preview_instance.value

    def _panel(branch):
        video = _root / branch / "video.mp4"
        mask = _root / branch / "segmentation_preview.mp4"
        items = [mo.md(f"**{branch}**")]
        if video.is_file():
            items.append(mo.video(src=video.read_bytes(), controls=True))
        if mask.is_file():
            items.append(mo.video(src=mask.read_bytes(), controls=True))
        return mo.vstack(items)

    mo.hstack(
        [_panel(name) for name in ("factual", "counterfactual", "subject_removed")
         if (_root / name).is_dir()],
        widths="equal",
    )
    return


@app.cell
def _(dataset_dir, json, mo, preview_instance):
    import numpy as np

    mo.stop(preview_instance.value is None, mo.md(""))
    _root = dataset_dir / "instances" / preview_instance.value
    _summary = json.loads((_root / "instance.json").read_text(encoding="utf-8"))

    _lines = []
    for _branch in ("factual", "counterfactual", "subject_removed"):
        _dir = _root / _branch
        if not _dir.is_dir():
            continue
        _seg = np.load(_dir / "segmentation.npz")["segmentation"]
        _trk = np.load(_dir / "tracking.npz")
        _graph = json.loads((_dir / "graph.json").read_text(encoding="utf-8"))
        _ids = [str(x) for x in _trk["object_ids"]]
        _never = [
            _ids[row] for row in range(len(_ids))
            if int(_trk["visible_pixels"][:, row].sum()) == 0
        ]
        _lines.append(
            f"| `{_branch}` | {_seg.shape[0]} | {len(_ids)} | "
            f"{len(_graph['graph']['edges'])} | {_graph['contact_count']} | "
            f"{', '.join(_never) if _never else 'none'} |"
        )
    mo.stop(
        not _lines,
        mo.md("*This instance has no rendered branches — was it run with `--no-render`?*"),
    )

    def _truth(name):
        truth = _summary["ground_truth"][name]
        delta = truth.get("graph_delta") or {}
        return (
            f"| `{name}` | `{truth.get('hard_affected')}` | "
            f"`{truth.get('soft_affected')}` | "
            f"+{len(delta.get('added') or [])} "
            f"-{len(delta.get('removed') or [])} "
            f"~{len(delta.get('changed') or [])} |"
        )

    mo.md(
        "### Annotations\n\n"
        "| Branch | Frames | Tracked bodies | Contact edges | Contacts | Never visible |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(_lines)
        + "\n\n### Ground truth against the factual branch\n\n"
        "Contact-graph delta is `+added -removed ~changed` edges.\n\n"
        "| Branch | Hard affected | Soft affected | Edge delta |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(_truth(name) for name in ("counterfactual", "subject_removed"))
        + "\n\nPropagation paths (how the intervention reached each affected "
        "body):\n\n```json\n"
        + json.dumps(
            {
                name: _summary["ground_truth"][name].get("propagation_path")
                for name in ("counterfactual", "subject_removed")
            },
            indent=2,
        )
        + "\n```"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Upload to HuggingFace

        The token is read from the field below or from `HF_TOKEN` in the
        environment. It is never written to disk, never printed, and never put
        in the dataset card or the commit message.

        **Uploading publishes this data to a third party.** A public repo can be
        mirrored or indexed before it can be deleted, so the notebook creates a
        **private** repo unless you tick the box. Run the dry run first — it
        writes the card and the split index locally and reports exactly what
        would be sent, without sending anything.
        """
    )
    return


@app.cell
def _(mo, os):
    token_input = mo.ui.text(
        value="",
        label="HF access token",
        kind="password",
        full_width=True,
        placeholder=(
            "found in HF_TOKEN" if os.environ.get("HF_TOKEN") else "hf_..."
        ),
    )
    repo_id_input = mo.ui.text(
        value="", label="Dataset repo id", placeholder="username/dataset-name",
        full_width=True,
    )
    public_toggle = mo.ui.checkbox(value=False, label="Make the repo public")
    dry_run_button = mo.ui.run_button(label="Dry run (uploads nothing)")
    upload_button = mo.ui.run_button(label="Upload")
    mo.vstack([
        token_input, repo_id_input, public_toggle,
        mo.hstack([dry_run_button, upload_button], justify="start"),
    ])
    return (
        dry_run_button,
        public_toggle,
        repo_id_input,
        token_input,
        upload_button,
    )


@app.cell
def _(
    dataset_dir,
    dry_run_button,
    manifest,
    mo,
    public_toggle,
    repo_id_input,
    token_input,
    upload_button,
):
    mo.stop(
        not (dry_run_button.value or upload_button.value),
        mo.md("*Press **Dry run** or **Upload**.*"),
    )
    mo.stop(
        manifest is None,
        mo.md(f"*Nothing to upload: no `manifest.json` under `{dataset_dir}`.*"),
    )
    if not repo_id_input.value.strip():
        raise ValueError("a dataset repo id is required, e.g. username/dataset-name")

    # Imported here and not at the top of the cell: the module lives in the
    # clone, which is only on sys.path after section 2 has run. An import above
    # the guards would fail on a freshly opened notebook, before the guards get
    # a chance to explain why.
    from scripts.upload_velocity_dataset import upload_dataset

    # Dry run wins when both buttons have been pressed. Erring toward not
    # transmitting is the right way round: a dry run costs a rerun, an upload
    # cannot be taken back.
    _dry = bool(dry_run_button.value) or not upload_button.value
    _result = upload_dataset(
        dataset_dir,
        repo_id_input.value.strip(),
        # Empty string means "fall back to the environment", which is what
        # resolve_token does with None.
        token=token_input.value.strip() or None,
        private=not public_toggle.value,
        dry_run=_dry,
    )

    mo.md(
        ("### Dry run — nothing was uploaded\n\n" if _dry else "### Uploaded\n\n")
        + f"- Repo: `{_result['repo_id']}` "
        + ("(private)" if _result["private"] else "**(public)**")
        + f"\n- Instances: {_result['instances']}"
        + f"\n- Files: {_result['files']} ({_result['total_mib']} MiB)"
        + f"\n- Splits: `{_result['splits']}`"
        + (f"\n- URL: {_result['url']}" if _result.get("url") else "")
    )
    return


if __name__ == "__main__":
    app.run()
