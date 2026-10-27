"""
================================================================================
test_persistence.py - 표·학습 객체 직렬화의 값 보존·신뢰 경계·쓰기 실패 계약을 검증합니다.
================================================================================
설명:
    - 표·학습 객체 직렬화의 값 보존·신뢰 경계·쓰기 실패 계약을 검증합니다.
분석 가정:
    - 테스트의 고정 입력·합성 자료와 명시한 허용오차를 사용합니다. 시험 데이터의 성능 보장이 아닙니다.
작성 정보:
    - 작성자: 김재현(Finn) (penguin.klg@gmail.com)
    - 작성일: 2026-10-27
    - 라이선스: Proprietary
변경 이력 (Revision History):
    DATE        AUTHOR          VERSION   DESCRIPTION
    2026-10-27  김재현(Finn)     v1.0.0    모듈 명세 헤더 표준화
================================================================================
"""

from io import BytesIO, StringIO
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression
from src.datasets import save, load, save_artifact, load_artifact, read_table
from src.datasets.persistence import PersistenceProfile
from src.Recipes import load_profile


class PersistenceTest(unittest.TestCase):
    def test_pickle_table_and_dto(self):
        frame = pd.DataFrame(
            {"x": [1, 2], "name": ["가", "나"]}, index=pd.Index([4, 5], name="row")
        )
        buffer = BytesIO()
        saved = save(frame, buffer, overrides={"format": "pickle"})
        self.assertFalse(buffer.closed)
        self.assertEqual(saved.n_bytes, len(buffer.getvalue()))
        self.assertNotIn("value", saved.to_dict())
        assert_frame_equal(
            load(buffer.getvalue(), overrides={"format": "pickle", "trusted": True}),
            frame,
        )
        restored = load_artifact(
            buffer.getvalue(), overrides={"format": "pickle", "trusted": True}
        )
        self.assertNotIn("value", json.loads(restored.to_json()))
        assert_frame_equal(restored.value, frame)
        assert_frame_equal(
            read_table(
                buffer.getvalue(), overrides={"format": "pickle", "trusted": True}
            ),
            frame,
        )

    def test_pickle_requires_trust_before_read_or_execution(self):
        with patch(
            "src.datasets.persistence.pickle.loads",
            side_effect=AssertionError("must not run"),
        ):
            with self.assertRaises(ValueError):
                load(b"anything", overrides={"format": "pickle"})
            with self.assertRaises(ValueError):
                read_table(b"anything", overrides={"format": "pickle"})
        with patch.object(
            Path, "read_bytes", side_effect=AssertionError("must not read")
        ):
            with self.assertRaises(ValueError):
                load("missing.pkl")

    def test_model_pickle_roundtrip_predictions(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = 2 * X[:, 0] - X[:, 1] + 3
        model = LinearRegression().fit(X, y)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pickle"
            save(model, path)
            restored = load(path, overrides={"trusted": True})
            np.testing.assert_allclose(restored.predict(X), model.predict(X))
            with self.assertRaises(TypeError):
                read_table(path, overrides={"trusted": True})

    def test_path_formats_overwrite_and_missing_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "value.PKL"
            save({"x": 1}, path)
            with self.assertRaises(FileExistsError):
                save({"x": 2}, path)
            self.assertEqual(load(path, overrides={"trusted": True}), {"x": 1})
            save({"x": 2}, path, overrides={"overwrite": True})
            self.assertEqual(load(path, overrides={"trusted": True}), {"x": 2})
            with self.assertRaises(OSError):
                save({}, Path(folder) / "missing" / "a.pkl")
            unknown = Path(folder) / "data.bin"
            with self.assertRaises(ValueError):
                save({}, unknown)
            save({}, unknown, overrides={"format": "pickle"})
            self.assertEqual(
                load(unknown, overrides={"format": "pickle", "trusted": True}), {}
            )
            with self.assertRaises(ValueError):
                save({}, "https://example.com/a.pkl")

    def test_failed_serialization_and_atomic_write_preserve_old(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.pkl"
            save({"x": 1}, path)
            before = path.read_bytes()
            with self.assertRaises((AttributeError, pickle.PicklingError)):
                save(lambda: 0, path, overrides={"overwrite": True})
            self.assertEqual(path.read_bytes(), before)
            with patch(
                "src.datasets.persistence.os.replace",
                side_effect=OSError("write failed"),
            ):
                with self.assertRaises(OSError):
                    save({"x": 2}, path, overrides={"overwrite": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["a.pkl"])

    def test_stream_position_restored_and_nonseek_warning(self):
        raw = pickle.dumps({"x": 1}, protocol=4)
        stream = BytesIO(b"XX" + raw)
        stream.seek(2)
        result = load_artifact(stream, overrides={"format": "pickle", "trusted": True})
        self.assertEqual(result.value, {"x": 1})
        self.assertEqual(stream.tell(), 2)
        self.assertFalse(stream.closed)

        class Reader:
            def read(self):
                return raw

        result = load_artifact(
            Reader(), overrides={"format": "pickle", "trusted": True}
        )
        self.assertFalse(result.metadata["stream_position_restored"])
        self.assertTrue(any("seek" in v for v in result.warnings))
        stream = BytesIO(b"bad")
        with self.assertRaises((pickle.UnpicklingError, EOFError)):
            load(stream, overrides={"format": "pickle", "trusted": True})
        self.assertEqual(stream.tell(), 0)

    def test_invalid_streams_and_partial_write(self):
        with self.assertRaises(TypeError):
            save({}, StringIO(), overrides={"format": "pickle"})
        with self.assertRaises(TypeError):
            load(StringIO("x"), overrides={"format": "pickle", "trusted": True})
        with self.assertRaises(TypeError):
            save({}, b"not a destination", overrides={"format": "pickle"})

        class Writer:
            def write(self, data):
                return 1

        with self.assertRaises(OSError):
            save({}, Writer(), overrides={"format": "pickle"})
        with self.assertRaises(ValueError):
            save({}, BytesIO(), overrides={"format": "pickle", "overwrite": True})

    def test_profile_recipe_validation(self):
        self.assertIsInstance(load_profile("artifact-persistence"), PersistenceProfile)
        for options in (
            {"format": "unknown"},
            {"format": "pickle", "trusted": 1},
            {"format": "pickle", "pickle_protocol": 99},
            {"format": "pickle", "joblib_compress": 2},
            {"format": "joblib", "joblib_compress": True},
            {"format": "pickle", "parquet_index": False},
            {"format": "parquet", "parquet_compression": "bad"},
            {"format": "parquet", "pickle_protocol": 3},
        ):
            with self.subTest(options=options), self.assertRaises(
                (ValueError, TypeError)
            ):
                save({}, BytesIO(), overrides=options)
        with self.assertRaises(ValueError):
            load(
                b"", overrides={"format": "pickle", "trusted": True, "overwrite": True}
            )
        with self.assertRaises(ValueError):
            save({}, BytesIO(), PersistenceProfile(format="pickle"), overrides={})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            path.write_text('{"format":"pickle"}')
            stream = BytesIO()
            save({}, stream, recipe_path=path)
            self.assertEqual(
                load(stream.getvalue(), recipe_path=path, overrides={"trusted": True}),
                {},
            )

    @unittest.skipUnless(importlib.util.find_spec("joblib"), "optional joblib")
    def test_joblib_objects_arrays_and_dataframe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.joblib"
            value = {
                "array": np.arange(12).reshape(3, 4),
                "profile": PersistenceProfile(),
            }
            save(value, path, overrides={"joblib_compress": 0})
            with self.assertRaises(ValueError):
                load(path)
            restored = load(path, overrides={"trusted": True})
            np.testing.assert_array_equal(restored["array"], value["array"])
            self.assertEqual(restored["profile"], value["profile"])
            frame = pd.DataFrame({"x": [1, 2]})
            save(frame, path, overrides={"overwrite": True})
            assert_frame_equal(read_table(path, overrides={"trusted": True}), frame)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow")
    def test_parquet_typed_frame_and_index(self):
        frame = pd.DataFrame(
            {
                "x": pd.Series([1, None], dtype="Int64"),
                "label": pd.Categorical(["a", "b"]),
                "day": pd.date_range("2026-01-01", periods=2),
            }
        )
        frame.index = pd.Index(["a", "b"], name="row")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.parquet"
            save(frame, path)
            assert_frame_equal(load(path), frame)
            assert_frame_equal(read_table(path), frame)
            save(
                frame,
                path,
                overrides={
                    "overwrite": True,
                    "parquet_index": False,
                    "parquet_compression": None,
                },
            )
            assert_frame_equal(load(path), frame.reset_index(drop=True))
        with self.assertRaises(TypeError):
            save({"x": [1]}, BytesIO(), overrides={"format": "parquet"})
        with self.assertRaises(ValueError):
            save(
                pd.DataFrame([[1, 2]], columns=["x", "x"]),
                BytesIO(),
                overrides={"format": "parquet"},
            )

    def test_dto_object_roundtrip(self):
        from src.datasets import run_tabular_loader

        original = run_tabular_loader(b'{"x":1}', overrides={"format": "json"})
        buffer = BytesIO()
        save_artifact(original, buffer, overrides={"format": "pickle"})
        restored = load(
            buffer.getvalue(), overrides={"format": "pickle", "trusted": True}
        )
        self.assertEqual(restored.to_json(), original.to_json())


if __name__ == "__main__":
    unittest.main()
