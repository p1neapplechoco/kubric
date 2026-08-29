# The `thesis` environment (Windows, Docker-free)

Every command in this repository's intervention and rendering pipeline runs in a
single Conda environment named `thesis`. Docker is not used anywhere, including
for rendering: Blender is provided by the pip-installed `bpy` module rather than
by the `kubricdockerhub/kubruntudev` image.

## Interpreter

```
C:\Users\uya7hc\.conda\envs\thesis\python.exe
```

Python 3.11.16 (conda-forge). Python 3.11 is required because `bpy==4.2.0` ships
only a `cp311` wheel.

## Creating the environment

```powershell
conda create -y -n thesis -c conda-forge python=3.11
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
& $py -m pip install bpy==4.2.0
& $py -m pip install --retries 10 --timeout 120 pybullet
& $py -m pip install pyquaternion traitlets munch "etils[epath]" imageio `
    imageio-ffmpeg pypng trimesh absl-py PyYAML scipy OpenEXR Pillow bidict `
    pytest hypothesis
& $py -m pip install tensorflow-cpu scikit-learn pandas
```

`pybullet` has no Windows `cp311` wheel and is built from source; the build is
slow but succeeds with the MSVC build tools already present. `tensorflow-cpu`
and `scikit-learn` are not optional: `kubric.file_io` imports TensorFlow for
`tf.io.gfile`, and `kubric.renderer.blender_utils` imports `sklearn.utils`, both
at module import time.

## Required machine settings

Two Windows settings must be enabled once, from an elevated PowerShell, followed
by a sign-out or reboot.

```powershell
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force

New-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock' `
  -Name AllowDevelopmentWithoutDevLicense -Value 1 -PropertyType DWORD -Force
```

`LongPathsEnabled` is mandatory, not a convenience. Published artifacts nest a
64-character pair generation inside a 64-character branch generation, which
produces paths of roughly 276 characters. Without long-path support these fail
partway through publication and surface as misleading `incomplete simulation
log` errors rather than as path-length errors.

The second setting grants the symlink-creation privilege that several tests rely
on to build rejection fixtures.

## Command prelude

The agent terminal is a persistent session, so clear `PYTHONPATH` before running
anything and give Matplotlib a writable config directory.

```powershell
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = "$env:TEMP\kubric-mpl"
$py = "C:\Users\uya7hc\.conda\envs\thesis\python.exe"
```

Scripts executed from a path outside the repository need the repository root on
`PYTHONPATH`, because `sys.path[0]` is the script's own directory:

```powershell
$env:PYTHONPATH = "$PWD"
& $py <script>
Remove-Item Env:PYTHONPATH
```

## Verifying the environment

```powershell
& $py -c "import kubric as kb; import kubric.renderer.blender; import kubric.simulator.pybullet; import bpy; print(bpy.app.version_string)"
```

A one-frame render is the real gate, because it is the only check that exercises
Cycles and the annotation-layer wiring:

```python
import pathlib, tempfile
import kubric as kb
from kubric.renderer.blender import Blender

scene = kb.Scene(resolution=(64, 64), frame_start=1, frame_end=1)
renderer = Blender(scene, scratch_dir=pathlib.Path(tempfile.mkdtemp()),
                   samples_per_pixel=1, adaptive_sampling=False,
                   use_denoising=False)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=2.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -3, 3),
                              look_at=(0, 0, 0))
scene += kb.Cube(name="floor", scale=(4, 4, 0.1), position=(0, 0, -0.1),
                 static=True)
scene += kb.Sphere(name="ball", scale=0.7, position=(0, 0, 0.7))
frame = renderer.render_still()
print(sorted(frame.keys()))
```

This must print all seven layers: `backward_flow`, `depth`, `forward_flow`,
`normal`, `object_coordinates`, `rgba`, `segmentation`.

## Platform differences that the code accounts for

`interventions/_portability.py` isolates the filesystem primitives that
differ between POSIX and Windows, so the publication paths in
`interventions/logging.py`, `interventions/dataset.py`, and
`interventions/twin_runner.py` share one implementation.

- **Directory fsync.** POSIX opens the directory and calls `os.fsync`. Windows
  cannot: `os.open` on a directory raises `PermissionError`, and no CRT handle
  exposes the entry. `fsync_directory` is therefore a no-op on Windows.
  Publication remains atomic there through `os.replace`, and payload contents are
  still durable through their own `os.fsync`, but the directory entry itself is
  not separately flushed. `DIRECTORY_FSYNC_SUPPORTED` exposes this difference so
  tests can assert the guarantee their platform actually provides.
- **Advisory locking.** POSIX uses `fcntl.flock`. Windows uses
  `msvcrt.locking`, which has no blocking mode that waits indefinitely, so
  contention is handled by polling.
- **Publication renames.** Windows refuses to rename a directory while any other
  process holds an open handle anywhere beneath it, which real-time virus
  scanners and the search indexer routinely do for a few milliseconds after files
  are written. Left unhandled this surfaced as a random
  `PermissionError: [WinError 5] Access is denied` from `os.rename` in roughly
  one to four tests per full-suite run, always in a different test. Publishers
  therefore call `publish_rename` and `publish_replace`, which retry on the
  delays in `PUBLISH_RETRY_DELAYS` and then make one final unguarded attempt, so
  a genuinely unwritable destination still raises its original exception.
  `PUBLISH_RETRY_DELAYS` is empty on POSIX, where `EACCES` is never transient.

## Test baseline

```powershell
& $py -m pytest tests/ test/ -q
```

Expected on Windows: **8 failed, 7 skipped**, everything else passing. All eight
failures are pre-existing platform gaps in demo tooling that the render pipeline
does not use, and none of them touch a module under active development:

| Tests | Cause |
| --- | --- |
| 5 in `tests/test_demo_collision_intervention.py` | Drive `run_demo.sh` through a POSIX shell sandbox; `shutil.which` finds no `dirname`, `id`, or `ls` on Windows. |
| 2 in `tests/test_compose_intervention_demo.py` | The ffmpeg `filter_complex` `textfile=` option is colon-delimited, so Windows drive-letter paths do not parse. |
| 1 in `tests/test_intervention_packaging.py` | `core.autocrlf=true` gives the working tree CRLF, so the built wheel fails an anchored `^...$` version match. |

Any other failure is a regression and should be treated as one.
