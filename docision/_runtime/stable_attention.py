"""SDPA blocks aligned to absolute token positions, independent of KV reuse."""
import torch
import torch.nn.functional as F


def aligned_attention(q,k,v,mask,query_start,scale,block_size=64,sdpa_fn=None):
    """Use identical Q/K shapes for a token in full and cached forwards.

    This is causal/isolated prefill only. Mask retains the caller's exact
    visibility; padded rows/keys cannot affect real outputs. No custom kernels.
    """
    nq=q.shape[-2];nk=k.shape[-2]
    sdpa_fn=sdpa_fn or F.scaled_dot_product_attention
    if query_start+nq!=nk or mask.shape!=(nq,nk):
        raise ValueError('Expected contiguous causal suffix with explicit visibility')
    end=((nk+block_size-1)//block_size)*block_size
    kp=F.pad(k,(0,0,0,end-nk));vp=F.pad(v,(0,0,0,end-nk))
    result=torch.empty_like(q)
    for start in range((query_start//block_size)*block_size,end,block_size):
        left=max(start,query_start);right=min(start+block_size,nk)
        a,b=left-query_start,right-query_start
        qp=q.new_zeros((*q.shape[:-2],block_size,q.shape[-1]))
        qp[...,left-start:right-start,:]=q[...,a:b,:]
        visible=torch.zeros((block_size,start+block_size),dtype=torch.bool,device=q.device)
        visible[left-start:right-start,:min(nk,start+block_size)]=mask[a:b,:start+block_size]
        value=sdpa_fn(qp,kp[...,:start+block_size,:],
            vp[...,:start+block_size,:],attn_mask=visible,dropout_p=0.,
            is_causal=False,scale=scale,enable_gqa=True)
        result[...,a:b,:]=value[...,left-start:right-start,:]
    return result
