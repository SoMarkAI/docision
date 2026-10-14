"""Python 3.10 regressions for content-addressed cross-request KV keys."""
import hashlib
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from docision._runtime.hf_request_cache import RequestPrefixCache


class RequestCacheCompatibilityTests(unittest.TestCase):
    def engine(self):
        engine=SimpleNamespace(lock=threading.RLock())
        engine.prepare_context=Mock(side_effect=lambda *a,**k:SimpleNamespace(prefix_length=1,decoded={}))
        engine.prepare=Mock(return_value=(None,{},None))
        return engine

    @staticmethod
    def score(engine,prepared,**kwargs):
        return [{'decision':{'index':0}} for _ in prepared],None

    def test_sha256_vectors_without_file_digest_and_bounded_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input.png'
            for raw in (b'',b'abc',bytes(range(256))*12289):
                path.write_bytes(raw);cache=RequestPrefixCache(self.engine());reads=[]
                original_open=Path.open
                class Reader:
                    def __enter__(self):
                        self.stream=original_open(path,'rb');return self
                    def __exit__(self,*args):self.stream.close()
                    def read(self,n):
                        reads.append(n)
                        assert 0<n<=1024*1024
                        return self.stream.read(n)
                with patch.object(hashlib,'file_digest',create=True,side_effect=AssertionError('Python 3.11-only API')) as forbidden, \
                     patch.object(Path,'open',return_value=Reader()), \
                     patch('docision._runtime.hf_request_cache.score_batch',side_effect=self.score):
                    cache.score_requests([{'image':str(path),'question':'q','candidates':['yes','no']}])
                    forbidden.assert_not_called()
                self.assertEqual(next(iter(cache.entries))[0],hashlib.sha256(raw).digest())
                self.assertTrue(reads)

    def test_alias_hit_content_and_context_change_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.png';b=Path(directory)/'b.png'
            a.write_bytes(b'image-one');b.write_bytes(a.read_bytes())
            engine=self.engine();cache=RequestPrefixCache(engine)
            def request(path,context='same'):
                return cache.score_requests([{'image':str(path),'question':'q',
                    'candidates':['yes','no'],'context':context}])[0]['prefix_cache_hit']
            with patch('docision._runtime.hf_request_cache.score_batch',side_effect=self.score):
                self.assertFalse(request(a));self.assertTrue(request(b))
                b.write_bytes(b'image-two');self.assertFalse(request(b))
                self.assertFalse(request(b,'different context'))
                self.assertTrue(request(b,'different context'))
            self.assertEqual(engine.prepare_context.call_count,3)


if __name__=='__main__':unittest.main()
