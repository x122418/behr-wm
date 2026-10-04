import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import download_data


class DownloadDataTests(unittest.TestCase):
    def test_textworld_plan_is_pinned_and_contains_all_evaluation_assets(self):
        self.assertEqual(
            download_data.REVISION,
            "ff6ae2b924d1a49e4b89825913887f2ea96cb282",
        )
        self.assertEqual(
            download_data._plan(["textworld"], webshop_backend=False),
            [
                "llama_factory/textworld_test_173.json",
                "eval/textworld_test.json",
                "init_contexts/textworld/agent_instruct_test.json",
                "init_contexts/textworld/wm_instruct_test.json",
                "textworld.zip",
            ],
        )

    def test_extract_unpacks_textworld_archive_under_data_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            archive = data_dir / "textworld.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("games/game_1.ulx", b"game")

            with patch.object(download_data, "ROOT", Path(tmpdir)), patch.object(
                download_data, "DATA_DIR", data_dir
            ), patch.dict(
                download_data.ARCHIVES,
                {
                    "textworld.zip": (
                        data_dir / "textworld",
                        data_dir / "textworld" / "games",
                    )
                },
                clear=True,
            ):
                download_data._extract("textworld.zip", force=False)

            self.assertEqual(
                (data_dir / "textworld" / "games" / "game_1.ulx").read_bytes(),
                b"game",
            )


if __name__ == "__main__":
    unittest.main()
