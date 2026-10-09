"""Regression tests for the WearGait-PD download script and its manifests.

Run from the repo root:  python3 -m unittest discover -s tests -v

Background: an earlier version of the script had a Synapse token written into
the source, and the manifests listed absolute paths from one PSC account.
"""
import csv
import importlib.util
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEARGAIT_DIR = REPO_ROOT / "data" / "gait" / "weargait"
SCRIPT = WEARGAIT_DIR / "download_weargait_pd_v1.py"
MANIFESTS = [
    WEARGAIT_DIR / "SYNAPSE_METADATA_MANIFEST.tsv",
    WEARGAIT_DIR / "CONTROL PARTICIPANTS" / "SYNAPSE_METADATA_MANIFEST.tsv",
]
JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")


def load_script():
    spec = importlib.util.spec_from_file_location("download_weargait_pd_v1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadScriptTests(unittest.TestCase):
    def test_source_has_no_token_literal(self):
        self.assertIsNone(JWT.search(SCRIPT.read_text(encoding="utf-8")))

    def test_token_must_come_from_environment(self):
        script = load_script()
        with self.assertRaises(SystemExit):
            script.get_token({})
        with self.assertRaises(SystemExit):
            script.get_token({"SYNAPSE_AUTH_TOKEN": "   "})

    def test_token_is_read_and_stripped(self):
        script = load_script()
        self.assertEqual(script.get_token({"SYNAPSE_AUTH_TOKEN": " abc \n"}), "abc")

    def test_data_root_must_be_set(self):
        script = load_script()
        with self.assertRaises(SystemExit):
            script.get_download_dir({})

    def test_download_dir_is_under_data_root(self):
        script = load_script()
        self.assertEqual(
            script.get_download_dir({"PD_DATA_ROOT": "/x/data"}),
            Path("/x/data/gait/weargait"),
        )


class ManifestTests(unittest.TestCase):
    def test_manifest_paths_are_relative_and_account_free(self):
        for manifest in MANIFESTS:
            with self.subTest(manifest=manifest.relative_to(REPO_ROOT)):
                text = manifest.read_text(encoding="utf-8")
                self.assertNotIn("/ocean/", text)
                self.assertNotIn("/Users/", text)
                with manifest.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
                self.assertGreater(len(rows), 0)
                for row in rows:
                    self.assertTrue(row["path"])
                    self.assertFalse(row["path"].startswith("/"), row["path"])


if __name__ == "__main__":
    unittest.main()
