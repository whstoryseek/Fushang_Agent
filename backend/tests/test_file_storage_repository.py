import unittest
from unittest.mock import patch

from app.db.file_storage_repository import FileStorageRepository


class FileStorageRepositoryTests(unittest.TestCase):
    def test_get_bytes_converts_memoryview_to_bytes(self):
        repo = FileStorageRepository()

        with patch.object(
            repo,
            "_execute_select",
            return_value=[{"content": memoryview(b"hello world")}],
        ):
            result = repo.get_bytes("kb/test.txt")

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, b"hello world")


if __name__ == "__main__":
    unittest.main()
