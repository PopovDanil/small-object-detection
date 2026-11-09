"""Shared commands for the seven Kaggle experiment notebooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path("/kaggle/working/manual_experiments")
INPUT = Path("/kaggle/input")
CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
EXPERIMENTS = {
    f"{model}_{size}_seed{seed}": {
        "model": model,
        "imgsz": size,
        "seed": seed,
        "batch": (16 if model == "yolo26m" else 8) if size == 640 else None,
    }
    for model in ("yolo26m", "yolo26x")
    for size, seeds in ((640, (42, 43, 44)), (1280, (42,)))
    for seed in seeds
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def environment():
    import torch

    packages = {}
    for name in ("ultralytics", "torch", "torchvision", "pycocotools", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def runtime(gpu=True):
    import torch

    if importlib.metadata.version("ultralytics") != "8.4.175":
        raise RuntimeError("Install ultralytics==8.4.175 (baseline version).")
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before this command.")


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return p


def verify_files(root, manifest):
    if not manifest:
        raise ValueError("Empty integrity manifest")
    for name, digest in manifest.items():
        safe_member(name)
        path = Path(root) / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing/corrupt file: {path}")


@contextmanager
def run_lock(work, experiment):
    """The same OS lock protects training, restore, backup and evaluation."""
    directory = Path(work) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{experiment}.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Run is busy. Wait for Training/Evaluation to stop before Backup/Resume."
            ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_data(data, counts=None):
    import yaml

    data = Path(data).resolve()
    cfg = yaml.safe_load((data / "visdrone.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(10)]
    if tuple(names) != CLASSES:
        raise ValueError("Expected the original ten VisDrone classes in the original order.")
    for split, count in (counts or {"train": 6471, "val": 548, "test": 1610}).items():
        if cfg.get(split) != f"images/{split}":
            raise ValueError(f"Unexpected split layout: {split}")
        with (data / "manifests" / f"{split}.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        images = [row["image"] for row in rows]
        if len(images) != count or len(set(images)) != count:
            raise ValueError(f"{split}: expected {count} distinct official split images")
        for name in images:
            if Path(name).name != name:
                raise ValueError(f"Invalid image name: {name}")
            for path in (
                data / "images" / split / name,
                data / "labels" / split / f"{Path(name).stem}.txt",
                data / "ignore" / split / f"{Path(name).stem}.txt",
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
    marker = data / "prepared_manifest.json"
    if marker.is_file():
        verify_files(data, json.loads(marker.read_text(encoding="utf-8")))
    cfg["path"] = str(data)
    return cfg


def find_data(inputs=INPUT, prepare_missing=False):
    """Prefer this session, then prepared Inputs (folder or ZIP), then original preparation."""
    local = ROOT / "data"
    if (local / "visdrone.yaml").is_file():
        validate_data(local)
        return local
    candidates = sorted(
        {
            p.parent
            for p in Path(inputs).rglob("visdrone.yaml")
            if (p.parent / "manifests/train.csv").is_file()
        }
    )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple prepared VisDrone Datasets: {candidates}. Attach one.")
    if candidates:
        validate_data(candidates[0])
        return candidates[0]
    archives = []
    for path in sorted(Path(inputs).rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for name in sorted(names):
                if name == "visdrone.yaml" or name.endswith("/visdrone.yaml"):
                    prefix = name[: -len("visdrone.yaml")]
                    if prefix + "manifests/train.csv" in names:
                        archives.append((path, prefix))
    if len(archives) > 1:
        raise RuntimeError("Multiple prepared VisDrone ZIPs. Attach exactly one prepared Dataset.")
    if archives:
        path, prefix = archives[0]
        stage = ROOT / f".prepared-{uuid.uuid4()}"
        stage.mkdir()
        try:
            with zipfile.ZipFile(path) as z:
                if z.testzip():
                    raise ValueError(f"Prepared Dataset ZIP CRC error: {path}")
                for entry in z.infolist():
                    if entry.is_dir() or not entry.filename.startswith(prefix):
                        continue
                    name = entry.filename[len(prefix) :]
                    safe_member(name)
                    if PurePosixPath(name).parts[0] not in (
                        "images",
                        "labels",
                        "ignore",
                        "manifests",
                        "visdrone.yaml",
                        "prepared_manifest.json",
                    ):
                        continue
                    dest = stage / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            validate_data(stage)
            shutil.copytree(stage, local, dirs_exist_ok=True)
        finally:
            shutil.rmtree(stage)
        validate_data(local)
        return local
    if prepare_missing:
        print(
            "No prepared Dataset attached. Preparing official VisDrone with the original CPU pipeline.",
            flush=True,
        )
        prepare(export=False)
        return local
    raise RuntimeError("VisDrone is not ready. Execute the Setup cell in this notebook first.")


def data_yaml(data, directory):
    import yaml

    # find_data already validated the complete Dataset in this command.
    cfg = yaml.safe_load((Path(data) / "visdrone.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(Path(data).resolve())
    path = Path(directory) / "visdrone.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def prepare(export=True, data=None):
    """Reuse the project's original CPU preparation, then export one reusable Dataset."""
    if data is None:
        data = ROOT / "data"
        if not (data / "visdrone.yaml").is_file():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/prepare_data.py")],
                cwd=ROOT,
                env=env,
                check=True,
            )
    data = Path(data)
    validate_data(data)
    files = [
        p
        for folder in ("images", "labels", "ignore", "manifests")
        for p in (data / folder).rglob("*")
        if p.is_file()
    ]
    files.append(data / "visdrone.yaml")
    manifest = {p.relative_to(data).as_posix(): sha256(p) for p in files}
    if data == ROOT / "data":
        write_json(data / "prepared_manifest.json", manifest)
    if not export:
        print(f"DATA READY: {data}")
        return data
    output = Path("/kaggle/working/prepared_visdrone.zip")
    if output.exists():
        with zipfile.ZipFile(output) as archive:
            if archive.testzip():
                raise ValueError("Existing prepared ZIP is corrupt; it has not been overwritten.")
            saved = json.loads(archive.read("prepared_visdrone/prepared_manifest.json"))
            if saved != manifest:
                raise ValueError(
                    "Existing prepared ZIP belongs to different data; it has not been overwritten."
                )
        print(output)
        return output
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for p in files:
            archive.write(p, "prepared_visdrone/" + p.relative_to(data).as_posix())
        archive.writestr("prepared_visdrone/prepared_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip():
            raise ValueError("Prepared data ZIP failed CRC validation")
    print(output)
    return output


def coco_weights(model):
    from ultralytics import YOLO

    directory = Path("/kaggle/working/coco_weights")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model}.pt"
    if path.resolve().parent != directory.resolve():
        raise ValueError("COCO checkpoint must be loaded from the expected coco_weights directory.")
    # A separate directory prevents accidentally loading an attached VisDrone best.pt.
    previous = Path.cwd()
    try:
        os.chdir(directory)
        loaded = YOLO(str(path))
    finally:
        os.chdir(previous)
    if loaded.task != "detect" or len(loaded.names) != 80:
        raise ValueError("Fresh runs require the official 80-class COCO checkpoint.")
    return path


def setup(model=None, failure=False):
    runtime()
    if failure:
        print(f"BASELINE READY: {failure_weights()}", flush=True)
    data = find_data(prepare_missing=True)
    print(f"DATA READY: {data}")
    if model:
        print(f"COCO READY: {coco_weights(model)}")
        print(environment())
    return data


def config(experiment, work):
    spec = dict(EXPERIMENTS[experiment])
    if spec["batch"] is None:
        path = Path(work) / f"{experiment}_pilot.json"
        if not path.is_file():
            raise RuntimeError("Run the separate 1280 Memory pilot cell before fresh Training.")
        pilot = json.loads(path.read_text(encoding="utf-8"))
        if pilot["experiment"] != experiment or pilot["spec"] != spec:
            raise ValueError("Memory pilot belongs to a different experiment")
        spec["batch"] = pilot["batch"]
        allowed = (2, 4) if spec["model"] == "yolo26m" else (1, 2)
        if spec["batch"] not in allowed:
            raise ValueError("Invalid pilot batch decision")
    return spec


def baseline_args(spec):
    import yaml

    args = yaml.safe_load((ROOT / f"configs/manual/{spec['model']}_baseline.yaml").read_text())
    for key in ("model", "data", "project", "name", "exist_ok", "resume", "save_dir"):
        args.pop(key, None)
    args.update(
        imgsz=spec["imgsz"],
        seed=spec["seed"],
        batch=spec["batch"],
        epochs=50,
        patience=15,
        device="0",
        workers=2,
        save=True,
    )
    return args


def checkpoint(path, spec, resumable=True):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("train_args", {})
    for key in ("seed", "imgsz", "batch"):
        if args.get(key) != spec[key]:
            raise ValueError(f"Checkpoint {key}={args.get(key)} differs from requested {spec[key]}")
    model = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt.get("model")
    scale = getattr(model, "yaml", {}).get("scale")
    architecture = Path(getattr(model, "yaml", {}).get("yaml_file", "")).stem
    if scale != spec["model"][-1] or (
        "yolo26" not in architecture and "yolo26" not in str(getattr(model, "yaml", {}))
    ):
        raise ValueError("Checkpoint architecture does not match requested YOLO26 model")
    if resumable and (ckpt.get("optimizer") is None or ckpt.get("epoch", -1) < 0):
        raise ValueError(
            "last.pt has no optimizer/epoch: weights only, cannot Resume. Use an unfinished-run Backup."
        )
    if resumable and ckpt["epoch"] + 1 >= args.get("epochs", 50):
        raise ValueError("Training already reached its final epoch; nothing to Resume.")
    return ckpt


class ManualStop(Exception):
    """Exit only after the epoch checkpoint and scheduler state have been saved."""


class PilotComplete(Exception):
    """Exit a disposable memory probe before epoch-end evaluation."""


def train(experiment, work=WORK):
    import torch
    from ultralytics import YOLO

    runtime()
    work = Path(work)
    with run_lock(work, experiment):
        directory = work / experiment
        resume = directory / "resume.json"
        if (directory / "completed.json").is_file():
            req = required_run(directory, resumable=False)
            if req["experiment"] != experiment or req["spec"] != config(experiment, work):
                raise ValueError("Completed run identity differs from this notebook")
            print(f"TRAINING ALREADY COMPLETE: {directory}")
            return
        if directory.exists() and not resume.is_file():
            raise FileExistsError(
                f"{directory}: refusing to overwrite. Use Resume in a new session."
            )
        spec = config(experiment, work)
        data = find_data()
        state = None
        if resume.is_file():
            req = json.loads((directory / "request.json").read_text())
            marker = json.loads(resume.read_text())
            if req["spec"] != spec or marker["run_id"] != req["run_id"]:
                raise ValueError("Resume run identity mismatch")
            if req["evaluator_sha256"] != sha256(ROOT / "src/evaluation/metrics.py"):
                raise ValueError("Evaluator changed since Backup")
            last = directory / "run/weights/last.pt"
            ckpt = checkpoint(last, spec)
            state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
            if state["epoch"] != ckpt["epoch"]:
                raise ValueError("Scheduler and last.pt refer to different epochs")
            if state["finished"]:
                raise ValueError("Run reached early stopping or max epochs; nothing to Resume.")
            model = YOLO(str(last))
            kwargs = {
                "resume": str(last),
                "data": str(data_yaml(data, directory)),
                "device": "0",
                "workers": 2,
                "save_dir": str(directory / "run"),
            }
        else:
            weights = coco_weights(spec["model"])
            directory.mkdir(parents=True, exist_ok=False)
            pilot = work / f"{experiment}_pilot.json"
            if pilot.exists():
                shutil.copy2(pilot, directory / "pilot.json")
            req = {
                "experiment": experiment,
                "run_id": str(uuid.uuid4()),
                "spec": spec,
                "initial_weights_sha256": sha256(weights),
                "initial_weights": str(weights),
                "baseline_args_sha256": sha256(
                    ROOT / f"configs/manual/{spec['model']}_baseline.yaml"
                ),
                "evaluator_sha256": sha256(ROOT / "src/evaluation/metrics.py"),
                "environment": environment(),
                "differences": {
                    "seed": [42, spec["seed"]],
                    "imgsz": [640, spec["imgsz"]],
                    "physical_batch": [16 if spec["model"] == "yolo26m" else 8, spec["batch"]],
                    "note": "x baseline actually used batch8 after OOM at batch16; nbs64 retained.",
                },
                "manifests_sha256": {
                    s: sha256(data / f"manifests/{s}.csv") for s in ("train", "val", "test")
                },
            }
            write_json(directory / "request.json", req)
            model = YOLO(str(weights))
            kwargs = baseline_args(spec)
            kwargs.update(
                model=str(weights),
                data=str(data_yaml(data, directory)),
                project=str(directory),
                name="run",
                exist_ok=False,
            )
        current = {s: sha256(data / f"manifests/{s}.csv") for s in ("train", "val", "test")}
        if current != req["manifests_sha256"]:
            raise ValueError("Dataset split manifests differ from the original run")
        stop = [False]

        def ask_stop(_signum, _frame):
            stop[0] = True
            print(
                "STOP REQUESTED: waiting for the end of this epoch and checkpoint write.",
                flush=True,
            )

        def start(trainer):
            if trainer.batch_size != spec["batch"]:
                raise ValueError("Physical batch changed; refusing automatic OOM fallback")
            if state:
                trainer.scheduler.load_state_dict(state["scheduler"])

        def guard(trainer):
            if trainer.batch_size != spec["batch"]:
                raise ValueError(
                    "Ultralytics reduced the physical batch after OOM. Stop and review the batch."
                )

        def saved(trainer):
            tmp = directory / "scheduler.pt.tmp"
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "scheduler": trainer.scheduler.state_dict(),
                    "finished": bool(trainer.stop),
                },
                tmp,
            )
            tmp.replace(directory / "scheduler.pt")
            if stop[0] and not trainer.stop:
                write_json(resume, {"run_id": req["run_id"]})
                raise ManualStop()

        model.add_callback("on_train_start", start)
        model.add_callback("on_train_batch_start", guard)
        model.add_callback("on_model_save", saved)
        previous = signal.signal(signal.SIGINT, ask_stop)
        try:
            model.train(**kwargs)
            resume.unlink(missing_ok=True)
            write_json(directory / "completed.json", {"run_id": req["run_id"]})
        except ManualStop:
            print(
                "STOPPED SAFELY. Run the independent Backup cell and download its ZIP.", flush=True
            )
        finally:
            signal.signal(signal.SIGINT, previous)


def run_notebook(command, log):
    """Show subprocess output; one notebook Interrupt requests a safe epoch boundary."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        while True:
            try:
                for line in process.stdout:
                    print(line, end="")
                    handle.write(line)
                    handle.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break
            except KeyboardInterrupt:
                print(
                    "Interrupt received. Waiting for a complete epoch/checkpoint; do not terminate the session."
                )
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)


def probe(experiment, batch, output):
    import torch
    from ultralytics import YOLO

    runtime()
    spec = dict(EXPERIMENTS[experiment], batch=batch)
    data = find_data()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(coco_weights(spec["model"])))
    kwargs = baseline_args(spec)
    kwargs.update(
        data=str(data_yaml(data, output)),
        project=str(output),
        name="probe",
        epochs=1,
        fraction=0.02,
        nbs=batch,
        warmup_epochs=0,
        val=False,
        plots=False,
        save=False,
        close_mosaic=0,
        exist_ok=False,
    )
    batches = [0]

    def end(trainer):
        if trainer.batch_size != batch:
            raise RuntimeError("CUDA out of memory: automatic batch fallback rejected")
        batches[0] += 1
        if batches[0] >= 8:
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            write_json(
                output / "memory.json",
                {
                    "batch": batch,
                    "batches": batches[0],
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "used_fraction": (total - free) / total,
                },
            )
            raise PilotComplete()

    torch.cuda.reset_peak_memory_stats()
    model.add_callback("on_train_batch_end", end)
    try:
        model.train(**kwargs)
    except PilotComplete:
        return
    raise RuntimeError("Pilot did not complete eight batches")


def pilot(experiment, work=WORK):
    spec = EXPERIMENTS[experiment]
    if spec["imgsz"] != 1280:
        raise ValueError("Only 1280 runs need a memory pilot")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    record = work / f"{experiment}_pilot.json"
    if record.exists() or (work / experiment).exists():
        raise FileExistsError("Pilot/run already exists; refusing to change the batch decision")
    rows = []
    for batch in (2, 4) if spec["model"] == "yolo26m" else (1, 2):
        out = work / f"{experiment}_pilot_batch{batch}"
        command = [
            sys.executable,
            str(Path(__file__)),
            "probe",
            "--experiment",
            experiment,
            "--batch",
            str(batch),
            "--output",
            str(out),
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        (work / f"{experiment}_pilot_batch{batch}.log").write_text(
            proc.stdout + proc.stderr, encoding="utf-8"
        )
        if proc.returncode:
            message = proc.stdout + proc.stderr
            if "out of memory" not in message.lower():
                raise RuntimeError(
                    f"Pilot failed for a non-memory reason. See {out}.log\n{message[-2000:]}"
                )
            rows.append({"batch": batch, "fits": False, "reason": "OOM"})
            break
        row = json.loads((out / "memory.json").read_text())
        row["fits"] = row["used_fraction"] <= 0.85
        rows.append(row)
    fitting = [r["batch"] for r in rows if r["fits"]]
    if not fitting:
        raise RuntimeError("No candidate batch fits with 15% headroom. No training was started.")
    write_json(
        record,
        {
            "experiment": experiment,
            "spec": spec,
            "batch": max(fitting),
            "probes": rows,
            "pilot_only_changes": {
                "epochs": 1,
                "fraction": 0.02,
                "nbs": "physical batch",
                "warmup_epochs": 0,
                "stop_after_batches": 8,
                "val": False,
                "save": False,
            },
            "training": "Independent fresh COCO weights; original nbs64, warmup3, full split.",
            "environment": environment(),
        },
    )
    print(record.read_text())


def required_run(directory, resumable):
    directory = Path(directory)
    req = json.loads((directory / "request.json").read_text())
    for name in (
        "request.json",
        "visdrone.yaml",
        "scheduler.pt",
        "run/weights/last.pt",
        "run/weights/best.pt",
        "run/args.yaml",
        "run/results.csv",
    ):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    import yaml

    args = yaml.safe_load((directory / "run/args.yaml").read_text())
    for key in ("seed", "imgsz", "batch"):
        if args[key] != req["spec"][key]:
            raise ValueError(f"Run args differ from request: {key}")
    checkpoint(directory / "run/weights/best.pt", req["spec"], resumable=False)
    ckpt = checkpoint(directory / "run/weights/last.pt", req["spec"], resumable=resumable)
    with (directory / "run/results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or (resumable and int(float(rows[-1]["epoch"])) != ckpt["epoch"] + 1):
        raise ValueError("results.csv is empty or does not match the last completed epoch")
    if resumable:
        import torch

        state = torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=False)
        if state["epoch"] != ckpt["epoch"] or state["finished"]:
            raise ValueError(
                "Backup is finished or scheduler does not match last.pt; cannot Resume."
            )
    return req


def export_run(experiment, work=WORK, final=False):
    work = Path(work)
    with run_lock(work, experiment):
        directory = work / experiment
        req = required_run(directory, resumable=not final)
        logs = list(work.glob(f"{experiment}*.log"))
        for log in logs:
            target = directory / "logs" / log.name
            target.parent.mkdir(exist_ok=True)
            if target.exists() and sha256(target) != sha256(log):
                # Keep training logs downloaded from earlier sessions as well.
                target = target.with_name(f"{target.stem}_{sha256(log)[:12]}{target.suffix}")
            if not target.exists():
                shutil.copy2(log, target)
        files = [
            p for p in directory.rglob("*") if p.is_file() and p.name != "backup_manifest.json"
        ]
        manifest = {
            "experiment": experiment,
            "run_id": req["run_id"],
            "spec": req["spec"],
            "resumable": not final,
            "files": {p.relative_to(directory).as_posix(): sha256(p) for p in files},
        }
        write_json(directory / "backup_manifest.json", manifest)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = work / f"{experiment}_{'final' if final else 'backup'}_{stamp}.zip"
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
            for p in [*files, directory / "backup_manifest.json"]:
                archive.write(p, f"{experiment}/" + p.relative_to(directory).as_posix())
        with zipfile.ZipFile(path) as archive:
            if archive.testzip():
                raise ValueError("Backup ZIP failed CRC validation")
            for name, digest in manifest["files"].items():
                with archive.open(f"{experiment}/{name}") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                        raise ValueError(f"Backup SHA256 mismatch: {name}")
        print(path)
        return path


def restore(experiment, work=WORK, inputs=INPUT):
    """Accept both manually attached ZIPs and Kaggle's already expanded folders."""
    work = Path(work)
    with run_lock(work, experiment):
        target = work / experiment
        if target.exists():
            raise FileExistsError(f"{target}: restore will not overwrite an existing run")
        candidates = []
        for p in Path(inputs).rglob("backup_manifest.json"):
            meta = json.loads(p.read_text())
            if meta.get("experiment") == experiment and meta.get("resumable"):
                candidates.append(("folder", p.parent, meta, None))
        for p in Path(inputs).rglob("*.zip"):
            with zipfile.ZipFile(p) as archive:
                names = [n for n in archive.namelist() if n.endswith("/backup_manifest.json")]
                for name in names:
                    meta = json.loads(archive.read(name))
                    if meta.get("experiment") == experiment and meta.get("resumable"):
                        if archive.testzip():
                            raise ValueError(f"ZIP CRC error: {p}")
                        candidates.append(("zip", p, meta, name.rsplit("/", 1)[0]))
        if len(candidates) != 1:
            raise ValueError(
                f"Attach exactly one resumable backup for {experiment}; found {len(candidates)}"
            )
        kind, source, meta, prefix = candidates[0]
        expected = EXPERIMENTS[experiment]
        if any(meta["spec"][k] != v for k, v in expected.items() if v is not None):
            raise ValueError("Backup model/seed/imgsz/batch mismatch")
        staging = work / f".restore-{uuid.uuid4()}"
        staging.mkdir()
        try:
            if kind == "folder":
                verify_files(source, meta["files"])
                for name in meta["files"]:
                    dest = staging / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / name, dest)
            else:
                with zipfile.ZipFile(source) as archive:
                    for name in meta["files"]:
                        safe_member(name)
                        dest = staging / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(f"{prefix}/{name}") as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            verify_files(staging, meta["files"])
            req = required_run(staging, resumable=True)
            if req["run_id"] != meta["run_id"] or req["spec"] != meta["spec"]:
                raise ValueError("Backup run identity mismatch")
            if req["evaluator_sha256"] != sha256(ROOT / "src/evaluation/metrics.py"):
                raise ValueError("Evaluator differs from this backup")
            pilot_record = staging / "pilot.json"
            if expected["batch"] is None:
                if not pilot_record.exists():
                    raise ValueError("1280 backup missing its batch pilot decision")
                decision = json.loads(pilot_record.read_text())
                if (
                    decision["experiment"] != experiment
                    or decision["spec"] != expected
                    or decision["batch"] != req["spec"]["batch"]
                ):
                    raise ValueError("1280 pilot decision differs from the backup identity")
                existing_pilot = work / f"{experiment}_pilot.json"
                if existing_pilot.exists():
                    raise FileExistsError(existing_pilot)
            write_json(staging / "resume.json", {"run_id": req["run_id"]})
            if expected["batch"] is None:
                shutil.copy2(pilot_record, work / f"{experiment}_pilot.json")
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        print(f"RESUME READY: {target}. Run only the Training cell next.")
        return target


def evaluate(experiment, mode, work=WORK):
    import matplotlib
    from ultralytics import YOLO

    from src.evaluation import metrics as m

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runtime()
    work = Path(work)
    with run_lock(work, experiment):
        directory = work / experiment
        req = json.loads((directory / "request.json").read_text())
        if not (directory / "completed.json").exists():
            raise ValueError("Finish Training before full evaluation.")
        weights = directory / "run/weights/best.pt"
        checkpoint(weights, req["spec"], resumable=False)
        data = find_data()
        images, labels = m.load_split_paths(
            "test", data / "manifests", data / "images", data / "labels"
        )
        output = directory / "evaluation" / mode
        if sha256(data / "manifests/test.csv") != req["manifests_sha256"]["test"]:
            raise ValueError("Evaluation test split differs from the training run")
        identity = {
            "run_id": req["run_id"],
            "mode": mode,
            "weights": sha256(weights),
            "evaluator": sha256(Path(m.__file__)),
            "test_manifest": sha256(data / "manifests/test.csv"),
        }
        if output_complete(output, identity):
            print(f"EVALUATION ALREADY COMPLETE: {output}")
            return
        output.mkdir(parents=True, exist_ok=False)
        tiled = mode == "tiled-ov20"
        size = 640 if tiled else int(mode.split("-")[1])
        model = YOLO(str(weights))
        if tuple(model.names.values()) != m.CLASSES:
            raise ValueError("Unexpected model classes")
        scores = m.evaluate_split(
            labels,
            images,
            model,
            imgsz=size,
            batch=1,
            device="0",
            ignore_dir=data / "ignore",
            inference="tiled" if tiled else "full",
            tile=640,
            overlap=0.20,
            tile_batch=1,
            runtime_images=200,
            warmup=10,
        )
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["experiment", "mode", *scores])
            writer.writeheader()
            writer.writerow({"experiment": experiment, "mode": mode, **scores})
        write_json(
            output / "provenance.json",
            {
                "run": req,
                "mode": mode,
                "weights_sha256": sha256(weights),
                "evaluator_sha256": sha256(Path(m.__file__)),
                "environment": environment(),
                "test_manifest_sha256": sha256(data / "manifests/test.csv"),
                "runtime_images": 200,
                "warmup": 10,
                "conf_ap": m.CONF_EVAL,
                "conf_recall": m.CONF_RECALL,
                "iou_recall": m.IOU_RECALL,
                "iou_nms": m.IOU_NMS,
                "max_det": m.MAX_DET,
                "ignore_iou": m.IGNORE_IOU_THR,
                "tile": 640,
                "overlap": 0.20,
            },
        )
        for kind in ("ap", "recall"):
            groups = ("tiny", "small_only", "medium", "large")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(groups, [scores[f"{kind}_{g}"] for g in groups])
            ax.set(ylim=(0, 1), title=f"{experiment}: {mode}", ylabel=kind)
            fig.tight_layout()
            fig.savefig(output / f"{kind}_sizes.png", dpi=150)
            plt.close(fig)
        seal_output(output, identity)
        print(output)


def failure_weights(inputs=INPUT):
    expected = "72239d4f020080b7b8fe862898ba1fbd74f378210dd9c09b1496239cc1466c3d"
    matches = [
        p
        for p in Path(inputs).rglob("best.pt")
        if p.stat().st_size == 44026393 and sha256(p) == expected
    ]
    if matches:
        return sorted(matches)[0]
    for path in sorted(Path(inputs).rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            for entry in z.infolist():
                if PurePosixPath(entry.filename).name != "best.pt" or entry.file_size != 44026393:
                    continue
                with z.open(entry) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest == expected:
                    target = ROOT.parent / "baseline_checkpoints/yolo26m_s640/best.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if sha256(target) != expected:
                            raise ValueError(f"Conflicting original checkpoint: {target}")
                    else:
                        with z.open(entry) as src, target.open("xb") as dst:
                            shutil.copyfileobj(src, dst)
                    if sha256(target) != expected:
                        raise ValueError("Checkpoint ZIP extraction SHA256 mismatch")
                    return target
    raise ValueError(
        "Failure analysis needs the ORIGINAL yolo26m_s640/best.pt as an additional Input. "
        "The code ZIP contains no model weights. Attach best.pt or its checkpoint ZIP."
    )


def output_complete(output, identity):
    marker = Path(output) / "output_manifest.json"
    if not marker.is_file():
        if Path(output).exists():
            raise FileExistsError(
                f"Incomplete existing output: {output}; it will not be overwritten."
            )
        return False
    saved = json.loads(marker.read_text())
    if saved["identity"] != identity:
        raise ValueError(f"Existing results belong to different inputs: {output}")
    verify_files(output, saved["files"])
    return True


def seal_output(output, identity):
    write_json(
        Path(output) / "output_manifest.json",
        {
            "identity": identity,
            "files": {
                p.relative_to(output).as_posix(): sha256(p)
                for p in Path(output).rglob("*")
                if p.is_file() and p.name != "output_manifest.json"
            },
        },
    )


def failure_files():
    from scripts.failure_analysis_addendum import CASES

    names = ["provenance.json"]
    for name in CASES:
        stem = Path(name).stem
        names.extend([f"{stem}.json", f"{stem}_predictions.npz", f"{stem}_comparison.png"])
    return [*names, f"{Path(CASES[0]).stem}_tiny_roi.png", f"{Path(CASES[2]).stem}_nms.png"]


def run_failure(inputs=INPUT, work=WORK):
    runtime()
    weights = failure_weights(inputs)
    data = find_data()
    output = Path(work) / "failure_analysis"
    from scripts.failure_analysis_addendum import CASES

    identity = {
        "weights": sha256(weights),
        "evaluator": sha256(ROOT / "src/evaluation/metrics.py"),
        "cases": {
            f"{folder}/test/{Path(name).stem}{'.jpg' if folder == 'images' else '.txt'}": sha256(
                data / folder / "test" / (name if folder == "images" else f"{Path(name).stem}.txt")
            )
            for name in CASES
            for folder in ("images", "labels", "ignore")
        },
    }
    with run_lock(work, "failure_analysis"):
        if output_complete(output, identity):
            print(f"FAILURE COMPARISONS ALREADY COMPLETE: {output}")
            return
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/failure_analysis_addendum.py"),
                "--weights",
                str(weights),
                "--data",
                str(data),
                "--output",
                str(output),
                "--device",
                "0",
            ],
            check=True,
        )
        for name in failure_files():
            if not (output / name).is_file():
                raise FileNotFoundError(output / name)
        seal_output(output, identity)
        print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "setup",
            "prepare",
            "train",
            "pilot",
            "probe",
            "backup",
            "resume",
            "evaluate",
            "export",
            "failure",
        ],
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--model", choices=["yolo26m", "yolo26x"])
    parser.add_argument("--failure-checkpoint", action="store_true")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--inputs", type=Path, default=INPUT)
    parser.add_argument("--mode", choices=["full-640", "full-1280", "tiled-ov20"])
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(data=find_data(prepare_missing=True))
    elif args.command == "setup":
        setup(args.model, failure=args.failure_checkpoint)
    elif args.command == "failure":
        run_failure(args.inputs, args.work)
    else:
        if not args.experiment:
            parser.error("--experiment is required")
        if args.command == "train":
            train(args.experiment, args.work)
        elif args.command == "pilot":
            pilot(args.experiment, args.work)
        elif args.command == "probe":
            if not args.batch or not args.output:
                parser.error("probe requires --batch and --output")
            probe(args.experiment, args.batch, args.output)
        elif args.command == "backup":
            export_run(args.experiment, args.work)
        elif args.command == "resume":
            restore(args.experiment, args.work, args.inputs)
        elif args.command == "export":
            export_run(args.experiment, args.work, final=True)
        elif args.command == "evaluate":
            if not args.mode:
                parser.error("evaluate requires --mode")
            evaluate(args.experiment, args.mode, args.work)


if __name__ == "__main__":
    main()
