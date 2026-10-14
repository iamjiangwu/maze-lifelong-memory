"""Small installation checks; scientific validation is maintained separately."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduction', ROOT/'reproduce.py')
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class InstallationTests(unittest.TestCase):
    def test_complete_artifact_list(self):
        layout = reproduction.read_layout()
        self.assertEqual(len(layout['artifacts']),18)
        for item in layout['artifacts']:
            folder = ROOT/item['directory']
            self.assertTrue((folder/'build.py').is_file())
            self.assertTrue((folder/'README.md').is_file())

    def test_input_integrity(self):
        self.assertEqual(reproduction.verify_sources(reproduction.read_layout())['status'],'PASS')

    def test_paths_stay_inside_repository(self):
        for relative in ('../outside','.'):
            with self.assertRaises(ValueError):
                reproduction.inside(ROOT,relative)


if __name__ == '__main__':
    unittest.main()
