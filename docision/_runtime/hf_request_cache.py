"""Worker-local, bounded image/context KV reuse for independent request batches.

Model, processor settings and numerical mode must stay fixed for this object's
lifetime. Different workers never share KV. No model.generate() is used.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path

from .inputs import normalize_questions
from .hf_batch import score_batch


class RequestPrefixCache:
    def __init__(self,engine,*,max_entries=8,max_tokens=8192):
        if max_entries<1 or max_tokens<1:raise ValueError('Cache limits must be positive')
        self.engine=engine;self.max_entries=max_entries;self.max_tokens=max_tokens
        self.entries=OrderedDict();self.tokens=0

    def clear(self):
        self.entries.clear();self.tokens=0

    def _retain(self,key,context):
        if context.prefix_length>self.max_tokens:return
        while self.entries and (len(self.entries)>=self.max_entries or
                                self.tokens+context.prefix_length>self.max_tokens):
            _,old=self.entries.popitem(last=False);self.tokens-=old.prefix_length
        self.entries[key]=context;self.tokens+=context.prefix_length

    def score_requests(self,requests,*,mode='causal',projection='full',batch_size=1):
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:
            raise ValueError('batch_size must be positive')
        if not isinstance(requests,list) or not requests:raise ValueError('Empty requests')
        normalized=normalize_questions(requests);results=[]
        # Lock covers model access, cache metadata and prefixes as one operation.
        with self.engine.lock:
            for start in range(0,len(normalized),batch_size):
                rows=normalized[start:start+batch_size]
                prepared=[];caches=[];hits=[];local={}
                for row in rows:
                    if row.get('modality','image')!='image':raise ValueError('Only images supported')
                    image=row.get('image')
                    if not isinstance(image,(str,Path)):raise ValueError('Image path required')
                    path=Path(image)
                    # Service inputs are immutable request snapshots. Content identity
                    # permits reuse even when separate HTTP requests have new filenames.
                    with path.open('rb') as source:
                        hasher=hashlib.sha256()
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            hasher.update(chunk)
                        digest=hasher.digest()
                    context=row.get('context','Inspect the supplied media carefully.')
                    key=(digest,context)
                    cached=local.get(key) or self.entries.get(key)
                    hits.append(cached is not None)
                    if cached is None:
                        cached=self.engine.prepare_context(str(path),context=context,modality='image')
                        self._retain(key,cached)
                    elif key in self.entries:self.entries.move_to_end(key)
                    local[key]=cached
                    batch,spec,_=self.engine.prepare(str(path),row['question'],row['candidates'],
                        context=context,modality='image',decoded=cached.decoded)
                    prepared.append((batch,spec));caches.append(cached)
                values,_=score_batch(self.engine,prepared,mode=mode,projection=projection,cache=caches)
                results.extend({'id':r.get('id'),**v,'prefix_cache_hit':hit}
                               for r,v,hit in zip(rows,values,hits))
        return results
