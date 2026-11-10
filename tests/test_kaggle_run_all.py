"""CPU integration checks for Run All, Input layouts and automatic data preparation."""

from __future__ import annotations

# ruff: noqa: E402 -- isolate local check dependencies before runtime imports
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / "runs/cpu_check_deps").is_dir():
    sys.path.insert(0, str(ROOT / "runs/cpu_check_deps"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
settings = ROOT / "runs/cpu_yolo_config"
settings.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(settings)
import yaml

from scripts import manual_experiment as m
from scripts.pack_manual_kaggle import build


class RunAllChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.real_validate = m.validate_data

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, path):
        cfg = {"path": "/old/session/data", "names": dict(enumerate(m.CLASSES))}
        for split in ("train", "val", "test"):
            cfg[split] = f"images/{split}"
            for folder, name, data in [
                ("images", f"{split}/frame.jpg", b"CPU fixture"),
                ("labels", f"{split}/frame.txt", b""),
                ("ignore", f"{split}/frame.txt", b""),
                ("manifests", f"{split}.csv", b"image\nframe.jpg\n"),
            ]:
                p = path / folder / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        (path / "visdrone.yaml").write_text(yaml.safe_dump(cfg))
        files = {
            p.relative_to(path).as_posix(): m.sha256(p) for p in path.rglob("*") if p.is_file()
        }
        m.write_json(path / "prepared_manifest.json", files)
        return path

    def tiny_validate(self, path):
        return self.real_validate(path, {"train": 1, "val": 1, "test": 1})

    def bundle(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.root / "manual_kaggle_code.zip")

    def test_setup_reuses_local_and_attached_data_without_download(self):
        project = self.root / "working/project"
        inputs = self.root / "input/datasets/user/data"
        attached = self.dataset(inputs / "prepared_visdrone")
        with (
            patch.object(m, "ROOT", project),
            patch.object(m, "validate_data", side_effect=self.tiny_validate),
            patch("subprocess.run") as run,
        ):
            self.assertEqual(m.find_data(self.root / "input", prepare_missing=True), attached)
            local = self.dataset(project / "data")
            self.assertEqual(m.find_data(self.root / "input", prepare_missing=True), local)
        run.assert_not_called()

    def test_prepared_zip_is_extracted_and_reused(self):
        original = self.dataset(self.root / "source")
        inputs = self.root / "inputs"
        inputs.mkdir()
        with zipfile.ZipFile(inputs / "prepared_visdrone.zip", "w") as z:
            for p in original.rglob("*"):
                if p.is_file():
                    z.write(p, "nested/prepared_visdrone/" + p.relative_to(original).as_posix())
        project = self.root / "project"
        project.mkdir()
        with (
            patch.object(m, "ROOT", project),
            patch.object(m, "validate_data", side_effect=self.tiny_validate),
            patch("subprocess.run") as run,
        ):
            found = m.find_data(inputs, prepare_missing=True)
            self.assertEqual(found, project / "data")
            self.assertEqual(m.find_data(inputs, prepare_missing=True), found)
            self.assertEqual(self.tiny_validate(found)["path"], str(found.resolve()))
        run.assert_not_called()

    def test_no_dataset_uses_original_cpu_preparation_once(self):
        project = self.root / "project"
        project.mkdir()
        inputs = self.root / "inputs"
        inputs.mkdir()

        def cpu_prepare(command, **kwargs):
            self.assertEqual(command[1], str(project / "scripts/prepare_data.py"))
            self.assertEqual(kwargs["cwd"], project)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.dataset(project / "data")

        with (
            patch.object(m, "ROOT", project),
            patch.object(m, "validate_data", side_effect=self.tiny_validate),
            patch("subprocess.run", side_effect=cpu_prepare) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.find_data(inputs, prepare_missing=True)
            m.find_data(inputs, prepare_missing=True)
        run.assert_called_once()
        self.assertTrue((project / "data/prepared_manifest.json").is_file())
        self.assertFalse((project.parent / "prepared_visdrone.zip").exists())

    def test_invalid_prepared_data_never_falls_back_to_different_data(self):
        prepared = self.dataset(self.root / "inputs/prepared")
        (prepared / "labels/train/frame.txt").write_text("changed")
        project = self.root / "project"
        with (
            patch.object(m, "ROOT", project),
            patch.object(m, "validate_data", side_effect=self.tiny_validate),
            patch("subprocess.run") as run,
            self.assertRaisesRegex(ValueError, "corrupt"),
        ):
            m.find_data(self.root / "inputs", prepare_missing=True)
        run.assert_not_called()

    def test_setup_checks_failure_checkpoint_before_preparing_data(self):
        with (
            patch.object(m, "runtime"),
            patch.object(m, "failure_weights", side_effect=ValueError("missing best.pt")),
            patch.object(m, "find_data") as data,
            self.assertRaisesRegex(ValueError, "missing best.pt"),
        ):
            m.setup(failure=True)
        data.assert_not_called()

    def test_completed_failure_is_reused_without_inference(self):
        data = self.root / "data"
        from scripts.failure_analysis_addendum import CASES

        for name in CASES:
            for folder in ("images", "labels", "ignore"):
                path = (
                    data
                    / folder
                    / "test"
                    / (name if folder == "images" else f"{Path(name).stem}.txt")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"CPU fixture")
        weights = self.root / "best.pt"
        weights.write_bytes(b"CPU fixture")
        work = self.root / "work"

        def fake_inference(command, **kwargs):
            out = Path(command[command.index("--output") + 1])
            out.mkdir(parents=True)
            for name in m.failure_files():
                (out / name).write_bytes(b"CPU fixture")

        with (
            patch.object(m, "runtime"),
            patch.object(m, "find_data", return_value=data),
            patch.object(m, "failure_weights", return_value=weights),
            patch("subprocess.run", side_effect=fake_inference) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.run_failure(work=work)
            m.run_failure(work=work)
            run.assert_called_once()
            (work / "failure_analysis/provenance.json").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                m.run_failure(work=work)
            run.assert_called_once()

    def test_completed_evaluation_is_reused_without_model_loading(self):
        from src.evaluation import metrics

        exp = "yolo26m_640_seed43"
        work = self.root / "work"
        directory = work / exp
        (directory / "run/weights").mkdir(parents=True)
        weights = directory / "run/weights/best.pt"
        weights.write_bytes(b"CPU fixture")
        data = self.dataset(self.root / "data")
        req = {
            "run_id": "fixture",
            "spec": m.EXPERIMENTS[exp],
            "manifests_sha256": {"test": m.sha256(data / "manifests/test.csv")},
        }
        m.write_json(directory / "request.json", req)
        (directory / "completed.json").write_text("{}")
        out = directory / "evaluation/full-640"
        out.mkdir(parents=True)
        (out / "metrics.csv").write_bytes(b"CPU fixture")
        m.seal_output(
            out,
            {
                "run_id": "fixture",
                "mode": "full-640",
                "weights": m.sha256(weights),
                "evaluator": m.sha256(Path(metrics.__file__)),
                "test_manifest": m.sha256(data / "manifests/test.csv"),
            },
        )
        with (
            patch.object(m, "runtime"),
            patch.object(m, "checkpoint"),
            patch.object(m, "find_data", return_value=data),
            patch("ultralytics.YOLO") as model,
            patch.object(metrics, "evaluate_split") as inference,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            m.evaluate(exp, "full-640", work)
        model.assert_not_called()
        inference.assert_not_called()
        self.assertEqual((out / "metrics.csv").read_bytes(), b"CPU fixture")

    def test_complete_outputs_reused_and_corruption_rejected(self):
        out = self.root / "evaluation"
        self.assertFalse(m.output_complete(out, {"run_id": "one"}))
        out.mkdir()
        (out / "metrics.csv").write_text("CPU fixture")
        m.seal_output(out, {"run_id": "one"})
        self.assertTrue(m.output_complete(out, {"run_id": "one"}))
        with self.assertRaisesRegex(ValueError, "different inputs"):
            m.output_complete(out, {"run_id": "two"})
        (out / "metrics.csv").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            m.output_complete(out, {"run_id": "one"})

    def test_first_cell_supports_unexpanded_code_zip_and_rejects_old_code(self):
        archive = self.bundle()
        inputs = self.root / "input"
        inputs.mkdir()
        shutil.copy2(archive, inputs / "code.zip")
        nb = json.loads((ROOT / "notebooks/kaggle/01_failure_analysis.ipynb").read_text())
        source = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
        source = source.replace("/kaggle/input", inputs.as_posix()).replace(
            "/kaggle/working", (self.root / "working").as_posix()
        )
        previous = Path.cwd()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "code-cell", "exec"), {})
        finally:
            os.chdir(previous)
        copied = self.root / "working/small-object-detection-manual/scripts/manual_experiment.py"
        self.assertTrue(copied.is_file())
        with zipfile.ZipFile(inputs / "code.zip") as z:
            contents = {name: z.read(name) for name in z.namelist()}
        meta = json.loads(contents["manual_code_manifest.json"])
        meta.pop("notebook_mode")
        contents["manual_code_manifest.json"] = json.dumps(meta).encode()
        with zipfile.ZipFile(inputs / "code.zip", "w") as z:
            for name, data in contents.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(RuntimeError, "old code Dataset"):
            exec(compile(source, "old-cell", "exec"), {})

    def run_all(self, book, archive, layout="folder", pause=False):
        base = self.root / (book.stem + layout + str(pause))
        inputs = base / "inputs"
        working = base / "working"
        inputs.mkdir(parents=True)
        working.mkdir()
        if layout == "folder":
            with zipfile.ZipFile(archive) as z:
                z.extractall(inputs / "datasets/user/code/manual_kaggle_code")
        else:
            shutil.copy2(archive, inputs / "manual_kaggle_code.zip")
        nb = json.loads(book.read_text())
        calls = []

        def dispatch(command, **kwargs):
            if command[1:3] == ["-m", "pip"]:
                calls.append("install")
                return subprocess.CompletedProcess(command, 0)
            script = Path(command[1])
            self.assertTrue(script.is_file(), script)
            self.assertTrue(script.is_relative_to(working))
            action = command[2]
            calls.append((action, command[3:]))
            if action == "train" and not pause:
                exp = command[command.index("--experiment") + 1]
                directory = working / "manual_experiments" / exp
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "completed.json").write_text("{}")
            elif action == "pilot":
                exp = command[command.index("--experiment") + 1]
                out = working / "manual_experiments" / f"{exp}_pilot.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("{}")
            elif action == "failure":
                out = working / "manual_experiments/failure_analysis"
                out.mkdir(parents=True, exist_ok=True)
                for name in m.failure_files():
                    (out / name).write_bytes(b"CPU fixture")
                (out / "output_manifest.json").write_text("{}")
            elif action in ("backup", "resume", "prepare"):
                self.fail(f"Run All must skip manual action: {action}")
            return subprocess.CompletedProcess(command, 0)

        def exported(command, **kwargs):
            dispatch(command)
            out = working / "final.zip"
            with zipfile.ZipFile(out, "w") as z:
                z.writestr("test", "CPU fixture")
            return str(out) + "\n"

        previous = Path.cwd()
        namespace = {}
        try:
            with (
                patch("subprocess.run", side_effect=dispatch),
                patch("subprocess.check_output", side_effect=exported),
                patch.object(m, "run_notebook", side_effect=lambda command, log: dispatch(command)),
                patch("IPython.display.display"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code":
                        source = (
                            "".join(cell["source"])
                            .replace("/kaggle/input", inputs.as_posix())
                            .replace("/kaggle/working", working.as_posix())
                        )
                        exec(compile(source, book.name, "exec"), namespace)
        finally:
            os.chdir(previous)
        return calls, working

    def test_all_seven_notebooks_run_all_in_order_from_folders_and_zips(self):
        archive = self.bundle()
        for book in sorted((ROOT / "notebooks/kaggle").glob("*.ipynb")):
            for layout in ("folder", "zip"):
                with self.subTest(book=book.name, layout=layout):
                    calls, working = self.run_all(book, archive, layout)
                    actions = [c[0] for c in calls if isinstance(c, tuple)]
                    if book.name.startswith("01_"):
                        self.assertEqual(actions, ["setup", "failure"])
                        self.assertTrue(
                            list((working / "manual_experiments").glob("failure_analysis_*.zip"))
                        )
                    else:
                        expected = (
                            ["setup"] + (["pilot"] if "_1280_" in book.name else []) + ["train"]
                        )
                        expected += ["evaluate"] * (2 if "_1280_" in book.name else 3) + ["export"]
                        self.assertEqual(actions, expected)
                        modes = [
                            c[1][c[1].index("--mode") + 1]
                            for c in calls
                            if isinstance(c, tuple) and c[0] == "evaluate"
                        ]
                        self.assertEqual(
                            modes,
                            ["full-640", "full-1280"]
                            + ([] if "_1280_" in book.name else ["tiled-ov20"]),
                        )

    def test_safe_pause_does_not_trigger_evaluation_backup_or_export(self):
        archive = self.bundle()
        book = ROOT / "notebooks/kaggle/02_yolo26m_640_seed43.ipynb"
        calls, _ = self.run_all(book, archive, pause=True)
        self.assertEqual([c[0] for c in calls if isinstance(c, tuple)], ["setup", "train"])


if __name__ == "__main__":
    unittest.main()
