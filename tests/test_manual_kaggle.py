"""CPU-only checks of manual delivery, checkpoint integrity and explicit resume."""

from __future__ import annotations

# ruff: noqa: E402 -- isolate CPU dependencies before importing the runtime
import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Optional isolated dependencies used by the local CPU check; never packaged for Kaggle.
if (ROOT / "runs/cpu_check_deps").exists():
    sys.path.insert(0, str(ROOT / "runs/cpu_check_deps"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config_dir = ROOT / "runs/cpu_yolo_config"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

import torch
import yaml

from scripts import manual_experiment as m
from scripts.pack_manual_kaggle import build


class CocoWeightsChecks(unittest.TestCase):
    def test_both_models_accept_different_historical_training_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "coco_weights"
            previous = Path.cwd()
            for model in ("yolo26m", "yolo26x"):
                for historical in (f"{model}.pt", "/historical/training/another_model.yaml"):
                    with self.subTest(model=model, historical=historical):
                        loaded = SimpleNamespace(
                            task="detect",
                            names=dict.fromkeys(range(80)),
                            ckpt={"train_args": {"model": historical}},
                        )
                        loader = Mock(return_value=loaded)
                        with (
                            patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=loader)}),
                            patch.object(m, "Path", return_value=directory) as paths,
                        ):
                            paths.cwd.return_value = previous
                            actual = m.coco_weights(model)
                        self.assertEqual(actual, directory / f"{model}.pt")
                        loader.assert_called_once_with(str(directory / f"{model}.pt"))
                        self.assertEqual(Path.cwd(), previous)

    def test_rejects_wrong_task_class_count_and_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "coco_weights"
            previous = Path.cwd()
            for model in ("yolo26m", "yolo26x"):
                for task, count in (("classify", 80), ("detect", 10)):
                    with self.subTest(model=model, task=task, classes=count):
                        loader = Mock(return_value=SimpleNamespace(task=task, names=range(count)))
                        with (
                            patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=loader)}),
                            patch.object(m, "Path", return_value=directory) as paths,
                        ):
                            paths.cwd.return_value = previous
                            with self.assertRaisesRegex(ValueError, "80-class COCO"):
                                m.coco_weights(model)
                        self.assertEqual(Path.cwd(), previous)
            loader = Mock()
            with (
                patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=loader)}),
                patch.object(m, "Path", return_value=directory),
                self.assertRaisesRegex(ValueError, "expected coco_weights directory"),
            ):
                m.coco_weights("../yolo26x")
            loader.assert_not_called()


class ManualChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exp = "yolo26m_640_seed43"
        self.spec = dict(m.EXPERIMENTS[self.exp])
        self.work = self.root / "work"
        self.run = self.work / self.exp

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, stripped=False):
        (self.run / "run/weights").mkdir(parents=True)
        req = {
            "experiment": self.exp,
            "run_id": "fixture-run-uuid",
            "spec": self.spec,
            "evaluator_sha256": m.sha256(ROOT / "src/evaluation/metrics.py"),
            "manifests_sha256": {
                s: m.sha256(ROOT / "data/.gitkeep") for s in ("train", "val", "test")
            },
        }
        m.write_json(self.run / "request.json", req)
        (self.run / "visdrone.yaml").write_text("path: /old/session\n")
        args = dict(
            self.spec, epochs=50, save_dir=f"/kaggle/working/manual_experiments/{self.exp}/run"
        )
        (self.run / "run/args.yaml").write_text(yaml.safe_dump(args))
        (self.run / "run/results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.1\n2,0.2\n")
        c = {
            "epoch": -1 if stripped else 1,
            "optimizer": None if stripped else {"state": {1: {"step": torch.tensor(2)}}},
            "train_args": args,
            "model": SimpleNamespace(yaml={"scale": "m", "yaml_file": "yolo26m.yaml"}),
        }
        for name in ("last", "best"):
            torch.save(c, self.run / f"run/weights/{name}.pt")
        torch.save(
            {"epoch": 1, "scheduler": {"last_epoch": 1}, "finished": False},
            self.run / "scheduler.pt",
        )
        return req

    def export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return m.export_run(self.exp, self.work)

    def test_zip_roundtrip_and_expanded_kaggle_folder(self):
        self.make_run()
        archive = self.export()
        inputs = self.root / "inputs"
        inputs.mkdir()
        shutil.copy2(archive, inputs)
        with contextlib.redirect_stdout(io.StringIO()):
            restored = m.restore(self.exp, self.root / "new-session", inputs)
        m.required_run(restored, True)
        original = json.loads((self.run / "request.json").read_text())
        self.assertEqual(
            json.loads((restored / "resume.json").read_text())["run_id"], original["run_id"]
        )
        expanded = self.root / "expanded-inputs/dataset/backup"
        with zipfile.ZipFile(archive) as z:
            z.extractall(expanded)
        with contextlib.redirect_stdout(io.StringIO()):
            restored_folder = m.restore(
                self.exp, self.root / "third-session", self.root / "expanded-inputs"
            )
        self.assertEqual(
            m.sha256(restored_folder / "run/weights/last.pt"),
            m.sha256(self.run / "run/weights/last.pt"),
        )
        with self.assertRaises(FileExistsError):
            m.restore(self.exp, self.root / "third-session", self.root / "expanded-inputs")

    def test_sha_corruption_rejected_before_restore(self):
        self.make_run()
        archive = self.export()
        inputs = self.root / "inputs"
        with zipfile.ZipFile(archive) as z:
            z.extractall(inputs)
        (inputs / self.exp / "run/weights/last.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            m.restore(self.exp, self.root / "new", inputs)
        self.assertFalse((self.root / "new" / self.exp).exists())

    def test_optimizer_is_required_and_finished_export_still_available(self):
        self.make_run(stripped=True)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            self.export()
        with contextlib.redirect_stdout(io.StringIO()):
            path = m.export_run(self.exp, self.work, final=True)
        with zipfile.ZipFile(path) as z:
            self.assertFalse(json.loads(z.read(f"{self.exp}/backup_manifest.json"))["resumable"])

    def test_wrong_seed_batch_model_rejected(self):
        self.make_run()
        for key, value in [("seed", 44), ("batch", 8), ("imgsz", 1280), ("model", "yolo26x")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.checkpoint(self.run / "run/weights/last.pt", dict(self.spec, **{key: value}))

    def test_scheduler_epoch_and_csv_must_match(self):
        self.make_run()
        torch.save({"epoch": 0, "scheduler": {}, "finished": False}, self.run / "scheduler.pt")
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.export()
        torch.save({"epoch": 1, "scheduler": {}, "finished": False}, self.run / "scheduler.pt")
        (self.run / "run/results.csv").write_text("epoch\n1\n")
        with self.assertRaisesRegex(ValueError, "results.csv"):
            self.export()

    def test_backups_keep_logs_from_both_sessions(self):
        self.make_run()
        log = self.work / f"{self.exp}_training.log"
        log.write_text("session one")
        self.export()
        log.write_text("session two")
        archive = self.export()
        with zipfile.ZipFile(archive) as z:
            logs = [z.read(n).decode() for n in z.namelist() if "/logs/" in n]
        self.assertCountEqual(logs, ["session one", "session two"])

    def test_completed_training_is_validated_and_not_restarted(self):
        self.make_run(stripped=True)
        (self.run / "completed.json").write_text("{}")
        with (
            patch.object(m, "runtime"),
            patch("ultralytics.YOLO") as model,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.train(self.exp, self.work)
        model.assert_not_called()

    def test_active_writer_blocks_backup(self):
        self.make_run()
        with m.run_lock(self.work, self.exp), self.assertRaisesRegex(RuntimeError, "busy"):
            self.export()

    def test_expanded_backup_identity_mismatch_rejected(self):
        self.make_run()
        archive = self.export()
        inputs = self.root / "inputs"
        with zipfile.ZipFile(archive) as z:
            z.extractall(inputs)
        p = inputs / self.exp / "backup_manifest.json"
        meta = json.loads(p.read_text())
        meta["run_id"] = "different"
        m.write_json(p, meta)
        with self.assertRaisesRegex(ValueError, "identity"):
            m.restore(self.exp, self.root / "new", inputs)

    def test_1280_pilot_is_restored_without_rerunning(self):
        self.exp = "yolo26m_1280_seed42"
        original = dict(m.EXPERIMENTS[self.exp])
        self.spec = dict(original, batch=2)
        self.run = self.work / self.exp
        self.make_run()
        m.write_json(
            self.run / "pilot.json", {"experiment": self.exp, "spec": original, "batch": 2}
        )
        archive = self.export()
        inputs = self.root / "inputs"
        inputs.mkdir()
        shutil.copy2(archive, inputs)
        with contextlib.redirect_stdout(io.StringIO()):
            m.restore(self.exp, self.root / "new", inputs)
        self.assertEqual(m.config(self.exp, self.root / "new")["batch"], 2)

    def test_resume_passes_explicit_checkpoint_and_scheduler_to_ultralytics(self):
        req = self.make_run()
        m.write_json(self.run / "resume.json", {"run_id": req["run_id"]})
        data = self.root / "data"
        (data / "manifests").mkdir(parents=True)
        for split in ("train", "val", "test"):
            (data / f"manifests/{split}.csv").write_bytes(b"")
        callbacks = {}
        scheduler = SimpleNamespace(load_state_dict=Mock())
        trainer = SimpleNamespace(batch_size=16, scheduler=scheduler)
        model = Mock()
        model.add_callback.side_effect = lambda event, callback: callbacks.update({event: callback})
        model.train.side_effect = lambda **kwargs: callbacks["on_train_start"](trainer)
        with (
            patch("ultralytics.YOLO", return_value=model) as factory,
            patch.object(m, "runtime"),
            patch.object(m, "find_data", return_value=data),
            patch.object(m, "data_yaml", return_value=data / "visdrone.yaml"),
        ):
            m.train(self.exp, self.work)
        last = str(self.run / "run/weights/last.pt")
        factory.assert_called_once_with(last)
        self.assertEqual(model.train.call_args.kwargs["resume"], last)
        self.assertEqual(model.train.call_args.kwargs["save_dir"], str(self.run / "run"))
        scheduler.load_state_dict.assert_called_once_with({"last_epoch": 1})

    def test_interrupt_saves_epoch_before_manual_stop(self):
        data = self.root / "data"
        (data / "manifests").mkdir(parents=True)
        for split in ("train", "val", "test"):
            (data / f"manifests/{split}.csv").write_bytes(b"")
        weights = self.root / "yolo26m.pt"
        weights.write_bytes(b"CPU fixture only; never used by a real trainer")
        callbacks = {}
        handlers = {}
        model = Mock()
        model.add_callback.side_effect = lambda event, callback: callbacks.update({event: callback})
        trainer = SimpleNamespace(
            batch_size=16,
            epoch=1,
            stop=False,
            scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 1}),
        )

        def fake_train(**kwargs):
            callbacks["on_train_start"](trainer)
            # A signal during an epoch only records a stop request, then returns.
            handlers[m.signal.SIGINT](None, None)
            self.assertFalse((self.run / "resume.json").exists())
            callbacks["on_model_save"](trainer)

        def handler(signum, callback):
            previous = handlers.get(signum)
            handlers[signum] = callback
            return previous

        model.train.side_effect = fake_train
        with (
            patch("ultralytics.YOLO", return_value=model),
            patch.object(m, "runtime"),
            patch.object(m, "find_data", return_value=data),
            patch.object(m, "data_yaml", return_value=data / "visdrone.yaml"),
            patch.object(m, "coco_weights", return_value=weights),
            patch.object(m.signal, "signal", side_effect=handler),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.train(self.exp, self.work)
        saved = torch.load(self.run / "scheduler.pt", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["scheduler"], {"last_epoch": 1})
        self.assertTrue((self.run / "resume.json").is_file())
        self.assertFalse((self.run / "completed.json").exists())

    def test_baseline_args_and_fixed_notebooks(self):
        books = sorted((ROOT / "notebooks/kaggle").glob("*.ipynb"))
        self.assertEqual(len(books), 7)
        for book in books:
            nb = json.loads(book.read_text(encoding="utf-8"))
            for c in nb["cells"]:
                if c["cell_type"] == "code":
                    ast.parse("".join(c["source"]))
                    self.assertEqual(c["outputs"], [])
            if book.name.startswith("01_"):
                continue
            exp = book.stem[3:]
            self.assertIn(exp, m.EXPERIMENTS)
            source = "\n".join("".join(c["source"]) for c in nb["cells"])
            for action in ("train", "backup", "resume", "evaluate", "export"):
                self.assertIn(repr(action), source.replace('"', "'"))
            for action in ("backup", "resume"):
                cell = next(
                    "".join(c["source"])
                    for c in nb["cells"]
                    if c["cell_type"] == "code"
                    and any(
                        isinstance(node, ast.Constant) and node.value == action
                        for node in ast.walk(ast.parse("".join(c["source"])))
                    )
                )
                self.assertNotIn("run_notebook", cell)
            spec = (
                dict(m.EXPERIMENTS[exp], batch=2)
                if m.EXPERIMENTS[exp]["batch"] is None
                else m.EXPERIMENTS[exp]
            )
            args = m.baseline_args(spec)
            for key, value in [
                ("epochs", 50),
                ("patience", 15),
                ("nbs", 64),
                ("warmup_epochs", 3),
                ("cos_lr", True),
            ]:
                self.assertEqual(args[key], value)

    def test_first_cells_execute_on_kaggle_like_expanded_dataset(self):
        with contextlib.redirect_stdout(io.StringIO()):
            archive = build(self.root / "code.zip")
        for layout in ("dataset", "dataset/kaggle_launch_bundle/baseline_args"):
            inputs = self.root / layout
            with zipfile.ZipFile(archive) as z:
                z.extractall(inputs)
            for book in sorted((ROOT / "notebooks/kaggle").glob("*.ipynb")):
                output = self.root / (book.stem + layout.replace("/", "_"))
                nb = json.loads(book.read_text(encoding="utf-8"))
                source = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
                source = source.replace(
                    "/kaggle/working/small-object-detection-manual", output.as_posix()
                )
                source = source.replace("/kaggle/input", inputs.as_posix()).replace(
                    "/kaggle/working", self.root.as_posix()
                )
                previous = Path.cwd()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(source, book.name, "exec"), {})
                finally:
                    os.chdir(previous)
                self.assertTrue((output / "configs/manual/yolo26m_baseline.yaml").exists())
            # Integrity guards stay active for already expanded Inputs.
            bad = inputs / "configs/manual/yolo26m_baseline.yaml"
            bad.write_text("changed")
            with (
                self.assertRaisesRegex(ValueError, "corrupt"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exec(compile(source, "bad-input", "exec"), {})

    def test_prepared_dataset_relocation_and_ignore_presence(self):
        data = self.root / "prepared_visdrone"
        cfg = {"path": "/old/session", "names": dict(enumerate(m.CLASSES))}
        files = []
        for split in ("train", "val", "test"):
            cfg[split] = f"images/{split}"
            for folder, name, content in [
                ("images", f"{split}/frame.jpg", b"fixture"),
                ("labels", f"{split}/frame.txt", b""),
                ("ignore", f"{split}/frame.txt", b""),
                ("manifests", f"{split}.csv", b"image\nframe.jpg\n"),
            ]:
                p = data / folder / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
                files.append(p)
        (data / "visdrone.yaml").write_text(yaml.safe_dump(cfg))
        m.write_json(
            data / "prepared_manifest.json",
            {p.relative_to(data).as_posix(): m.sha256(p) for p in files},
        )
        self.assertEqual(
            m.validate_data(data, {"train": 1, "val": 1, "test": 1})["path"], str(data.resolve())
        )
        (data / "ignore/test/frame.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            m.validate_data(data, {"train": 1, "val": 1, "test": 1})

    def test_archive_paths_and_cli(self):
        for path in ("../x", "/x", "a\\b", "C:/x"):
            with self.assertRaises(ValueError):
                m.safe_member(path)
        for name in ("manual_experiment.py", "failure_analysis_addendum.py"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
