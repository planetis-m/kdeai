import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import portalocker

from kdeai import hash as kdehash
from kdeai import locks
from kdeai import snapshot


class TestLocks(unittest.TestCase):
    def test_run_lock_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with locks.acquire_run_lock(root):
                with self.assertRaises(portalocker.exceptions.LockException):
                    with locks.acquire_run_lock(root):
                        pass

    def test_file_lock_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lock_id = locks.lock_id("project", "path/to/file.po")
            lock_path = locks.per_file_lock_path(root, lock_id)
            with locks.acquire_file_lock(lock_path):
                with self.assertRaises(portalocker.exceptions.LockException):
                    with locks.acquire_file_lock(lock_path):
                        pass


class TestHashes(unittest.TestCase):
    def test_hashes_deterministic(self):
        msgctxt = "ctx"
        msgid = "Hello"
        msgid_plural = "Hellos"
        expected_source_key = hashlib.sha256(
            (msgctxt + "\u0004" + msgid + "\u0000" + msgid_plural).encode("utf-8")
        ).hexdigest()

        self.assertEqual(kdehash.source_key(msgctxt, msgid, msgid_plural), expected_source_key)

        msgstr_plural = {"0": "eins", "1": "zwei"}
        msgstr_plural_json = json.dumps(msgstr_plural, sort_keys=True, separators=(",", ":"))
        translation_body = (
            "v1\n"
            f"source_key={expected_source_key}\n"
            "lang=de\n"
            "msgstr=hallo\n"
            f"msgstr_plural={msgstr_plural_json}\n"
        )
        expected_translation_hash = hashlib.sha256(
            translation_body.encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            kdehash.translation_hash(expected_source_key, "de", "hallo", msgstr_plural),
            expected_translation_hash,
        )

        marker_flags = ["fuzzy", "kdeai-ai"]
        marker_flags_str = ",".join(sorted(marker_flags))
        tool_comment_lines = ["KDEAI-AI: model=test-model"]
        tool_lines = "\n".join(line.rstrip("\n") for line in tool_comment_lines)
        state_body = (
            "v2\n"
            f"source_key={expected_source_key}\n"
            "lang=de\n"
            "msgstr=hallo\n"
            f"msgstr_plural={msgstr_plural_json}\n"
            f"marker_flags={marker_flags_str}\n"
            f"tool_comment_lines={tool_lines}\n"
        )
        expected_state_hash = hashlib.sha256(state_body.encode("utf-8")).hexdigest()
        self.assertEqual(
            kdehash.state_hash(
                expected_source_key,
                "de",
                "hallo",
                msgstr_plural,
                marker_flags,
                tool_comment_lines,
            ),
            expected_state_hash,
        )

        term_body = "v1\n" + "\u001f".join(["foo", "bar"])
        expected_term_key = hashlib.sha256(term_body.encode("utf-8")).hexdigest()
        self.assertEqual(kdehash.term_key(["foo", "bar"]), expected_term_key)


class TestSnapshot(unittest.TestCase):
    def test_locked_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "file.po"
            target.write_bytes(b"abc")
            lock_id = locks.lock_id("project", "file.po")
            lock_path = locks.per_file_lock_path(root, lock_id)

            result = snapshot.locked_read_file(target, lock_path, relpath="file.po")
            stat = target.stat()
            self.assertEqual(result.file_path, "file.po")
            self.assertEqual(result.bytes, b"abc")
            self.assertEqual(result.sha256, hashlib.sha256(b"abc").hexdigest())
            self.assertEqual(result.size, stat.st_size)
            self.assertEqual(result.mtime_ns, stat.st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
