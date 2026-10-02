"""Fallback-font margins must fit without changing data limits or paired widths."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ed8', ROOT/'figures/extended_data_08/code/build_ed8_standardized_pv.py')
ed8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed8)


class FontMarginTests(unittest.TestCase):
    def tearDown(self):
        ed8.plt.close('all')

    def test_existing_margin_is_unchanged(self):
        fig, ax = ed8.plt.subplots(figsize=(4, 3))
        fig.subplots_adjust(left=.25)
        ax.set_ylabel('Value', fontfamily='DejaVu Sans')
        before = ax.get_position().bounds
        self.assertEqual(ed8.fit_ylabel_margins(fig), [])
        self.assertEqual(ax.get_position().bounds, before)

    def test_fallback_overflow_is_fitted_with_equal_axis_widths(self):
        fig = ed8.plt.figure(figsize=(3, 2), dpi=100)
        axes = [fig.add_axes([left, .25, .3, .6]) for left in (.09, .6)]
        axes[0].set_ylabel('Standardized PV correlation\n(across-gap / within-day)',
                           fontfamily='DejaVu Sans', fontsize=6)
        for ax in axes:
            ax.set(xlim=(-.5, 1.5), ylim=(-.3, .7))
        self.assertTrue(ed8.fit_ylabel_margins(fig))
        self.assertGreaterEqual(axes[0].yaxis.label.get_window_extent().x0, -1.)
        self.assertAlmostEqual(axes[0].get_position().width, axes[1].get_position().width)
        for ax in axes:
            self.assertEqual(ax.get_xlim(), (-.5, 1.5))
            self.assertEqual(ax.get_ylim(), (-.3, .7))


if __name__ == '__main__':
    unittest.main()
