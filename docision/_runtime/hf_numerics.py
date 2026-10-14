"""Shape-stable HF scoring using native encoders and FP32 text computation.

Weights retain their loaded dtype and values. Native media encoders run once per
unique item within a forward; repeated items reuse those features. This module
uses PyTorch only: no vLLM, custom kernel, compilation, generation, or training.
"""
from contextlib import contextmanager,ExitStack
from contextvars import ContextVar
from threading import RLock
from unittest.mock import patch
import torch
import torch.nn.functional as F
from .families import text_model, multimodal_model
from torch.nn.attention import sdpa_kernel,SDPBackend

_LOCK=RLock()
_ACTIVE=ContextVar("docision_hf_stable",default=False)
_TEXT=ContextVar("docision_hf_text",default=False)

def combined(outputs):
    values={}
    for k in outputs[0].keys():
        xs=[x[k] for x in outputs]
        if isinstance(xs[0],torch.Tensor):values[k]=torch.cat(xs,dim=0)
        elif isinstance(xs[0],(tuple,list)):
            values[k]=tuple(v for x in xs for v in x) if k=='pooler_output' else tuple(torch.cat([x[i] for x in xs],dim=0) for i in range(len(xs[0])))
        else:raise ValueError(('Unexpected encoder output',k))
    return type(outputs[0])(**values)

@contextmanager
def _stable(model):
    original=F.linear
    original_sdpa=F.scaled_dot_product_attention
    def linear(x,w,b=None):
        if _ACTIVE.get() and x.dtype==torch.float32:
            # Fixed row count removes shape-dependent GEMM reduction choices,
            # including expert batches and the last-token vocabulary readout.
            flat=x.reshape(-1,x.shape[-1]);weight=w.float();bias=None if b is None else b.float()
            chunks=[]
            for start in range(0,flat.shape[0],64):
                value=flat[start:start+64];n=value.shape[0]
                value=F.pad(value,(0,0,0,64-n))
                chunks.append(original(value,weight,bias)[:n])
            if not chunks:return x.new_empty((*x.shape[:-1],w.shape[0]))
            return torch.cat(chunks).reshape(*x.shape[:-1],w.shape[0])
        return original(x,w,b)
    def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None,enable_gqa=False):
        if not _TEXT.get():
            return original_sdpa(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,
                is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        if dropout_p:raise ValueError('Stable scoring requires dropout=0')
        from .stable_attention import aligned_attention
        nq,nk=q.shape[-2],k.shape[-2];prefix=nk-nq
        if attn_mask is None:
            visible=torch.ones((q.shape[0],nq,nk),dtype=torch.bool,device=q.device)
            if is_causal:visible &= torch.arange(nk,device=q.device)[None,None,:]<=torch.arange(prefix,nk,device=q.device)[None,:,None]
        else:
            mask=attn_mask if attn_mask.dtype==torch.bool else attn_mask==0
            visible=mask.expand(q.shape[0],1,nq,nk)[:,0]
        return torch.cat([aligned_attention(q[i:i+1],k[i:i+1],v[i:i+1],visible[i],
            prefix,scale if scale is not None else q.shape[-1]**-.5,sdpa_fn=original_sdpa)
            for i in range(q.shape[0])],dim=0)
    def text_pre(mod,args,kwargs):
        _TEXT.set(True)
        kwargs['inputs_embeds']=kwargs['inputs_embeds'].float()
        return args,kwargs
    def visual_wrap(original):
        def visual(pixels,grid,**kwargs):
            if grid.shape[0]==1:return original(pixels,grid,**kwargs)
            outs=[];seen=[];offset=0
            for g in grid:
                n=int(g.prod());x=pixels[offset:offset+n];offset+=n
                found=next((o for xx,gg,o in seen if torch.equal(x,xx) and torch.equal(g,gg)),None)
                if found is None:found=original(x,g[None],**kwargs);seen.append((x,g,found))
                outs.append(found)
            if offset!=pixels.shape[0]:raise ValueError('Visual grid mismatch')
            return combined(outs)
        return visual
    cfg=model.config.text_config;old=getattr(cfg,'_experts_implementation',None)
    cfg._experts_implementation='eager'
    oldtf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    h=text_model(model).register_forward_pre_hook(text_pre,with_kwargs=True,prepend=True)
    def text_post(mod,args,output):_TEXT.set(False)
    post=text_model(model).register_forward_hook(text_post)
    text_token=_TEXT.set(False)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(F,'linear',linear))
            stack.enter_context(patch.object(F,'scaled_dot_product_attention',attention))
            owner=multimodal_model(model)
            for name in ['get_image_features']:
                stack.enter_context(patch.object(owner,name,visual_wrap(getattr(owner,name))))
            stack.enter_context(sdpa_kernel(SDPBackend.MATH))
            yield
    finally:
        h.remove();post.remove();_TEXT.reset(text_token)
        cfg._experts_implementation=old;torch.backends.cuda.matmul.allow_tf32=oldtf


@contextmanager
def numerical_context(model,mode):
    if mode not in ('native','stable'):raise ValueError('Unknown HF numerical mode')
    # Serialize our engines while process-wide PyTorch settings are scoped.
    # The functional wrapper is also context-local for unrelated Python threads.
    with _LOCK:
        if mode=='native' or _ACTIVE.get():
            yield
            return
        token=_ACTIVE.set(True)
        try:
            with _stable(model):yield
        finally:_ACTIVE.reset(token)
