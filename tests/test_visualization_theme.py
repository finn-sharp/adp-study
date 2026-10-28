"""
================================================================================
test_visualization_theme.py - 공통 테마의 설정 복원과 기존 자료·축 소유권 보존을 검증합니다.
================================================================================
설명:
    - 공통 테마의 설정 복원과 기존 자료·축 소유권 보존을 검증합니다.
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

import unittest
import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from src.Visualization.theme import analysis_theme, style_axes


class VisualizationThemeTest(unittest.TestCase):
    def test_context_restores_settings_even_after_failure(self):
        before = mpl.rcParams.copy()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with analysis_theme():
                self.assertNotEqual(
                    mpl.rcParams["axes.prop_cycle"], before["axes.prop_cycle"]
                )
                raise RuntimeError("intentional")
        for key in (
            "axes.prop_cycle",
            "font.family",
            "axes.grid",
            "axes.titlelocation",
        ):
            self.assertEqual(mpl.rcParams[key], before[key])

    def test_styling_preserves_data_scales_and_other_axes(self):
        fig = Figure()
        ax, other = fig.subplots(1, 2)
        data = np.array([[0.0, 0.3], [0.7, 1.0]])
        norm = Normalize(0, 1)
        image = ax.imshow(data, norm=norm, cmap="Blues")
        (line,) = ax.plot([1, 2], [3, 4], color="black", linestyle="--")
        ax.text(0, 0, "0")
        ax.set_yscale("symlog")
        limits = (ax.get_xlim(), ax.get_ylim())
        other_before = (other.get_facecolor(), other.spines["right"].get_visible())
        self.assertIs(style_axes(ax), ax)
        self.assertEqual(len(fig.axes), 2)
        np.testing.assert_array_equal(image.get_array(), data)
        np.testing.assert_array_equal(line.get_ydata(), [3, 4])
        self.assertIs(image.norm, norm)
        self.assertEqual(image.get_clim(), (0, 1))
        self.assertEqual(ax.get_yscale(), "symlog")
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        self.assertEqual(
            (other.get_facecolor(), other.spines["right"].get_visible()), other_before
        )
        with self.assertRaises(TypeError):
            style_axes(fig.add_subplot(projection="polar"))


if __name__ == "__main__":
    unittest.main()
