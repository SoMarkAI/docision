"""Packaging boundaries: release archives must not inherit working-directory data."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('build_overlay',ROOT/'release/build_overlay.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class ReleaseTests(unittest.TestCase):
    def test_exact_allowlist_excludes_runtime_data_and_sets_modes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'release').mkdir();(root/'.runtime').mkdir()
            (root/'release/manifest.json').write_text(json.dumps({'files':['start.sh','LICENSE']}))
            (root/'start.sh').write_text('#!/bin/sh\n');(root/'LICENSE').write_text('license')
            (root/'.runtime/api-token').write_text('private-runtime-value')
            (root/'personal.pdf').write_text('private-document')
            archive=root/'out.tar.gz'
            with patch.object(builder,'ROOT',root):builder.build(archive)
            with tarfile.open(archive) as tar:
                self.assertEqual(set(tar.getnames()),{'deployment/start.sh','deployment/LICENSE'})
                self.assertEqual(tar.getmember('deployment/start.sh').mode,0o755)
                self.assertEqual(tar.getmember('deployment/LICENSE').mode,0o644)
            with patch.object(builder,'ROOT',root),self.assertRaises(FileExistsError):builder.build(archive)

    def test_unsafe_path_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'release').mkdir();(root/'outside').write_text('secret')
            for name in ('../outside','/absolute/path'):
                (root/'release/manifest.json').write_text(json.dumps({'files':[name]}))
                with patch.object(builder,'ROOT',root),self.assertRaises(ValueError):builder.build(root/'out.tar.gz')
            (root/'link').symlink_to(root/'outside')
            (root/'release/manifest.json').write_text(json.dumps({'files':['link']}))
            with patch.object(builder,'ROOT',root),self.assertRaises(ValueError):builder.build(root/'out.tar.gz')

    def test_manifest_is_source_only_and_files_exist(self):
        files=json.loads((ROOT/'release/manifest.json').read_text())['files']
        forbidden={'.runtime','.venv','.mcp-venv','.cache','inputs','exports','validation','profile_output','my-codex'}
        self.assertEqual(len(files),len(set(files)))
        for name in files:
            self.assertTrue((ROOT/name).is_file(),name)
            self.assertFalse(set(Path(name).parts)&forbidden,name)
        self.assertIn('LICENSE',files)
        self.assertIn('THIRD_PARTY_NOTICES.md',files)

if __name__=='__main__':unittest.main()
