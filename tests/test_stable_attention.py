import pytest,torch
import torch.nn.functional as F
from docision._runtime.stable_attention import aligned_attention
from docision._runtime.hf import isolated_mask

@pytest.mark.parametrize('mode',['causal','isolated'])
def test_absolute_blocks_preserve_mask_and_cached_suffix(mode):
    torch.manual_seed(2);n=151;p=77
    q=torch.randn(1,4,n,16);k=torch.randn(1,2,n,16);v=torch.randn_like(k)
    spans=[[90,110],[110,135]]
    mask=(isolated_mask(n,spans,q.device,q.dtype)[0,0]==0 if mode=='isolated'
          else torch.ones((n,n),dtype=torch.bool).tril())
    ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,enable_gqa=True,scale=.25)
    actual=aligned_attention(q,k,v,mask,0,.25)
    torch.testing.assert_close(actual,ref,atol=1e-6,rtol=1e-5)
    suffix=aligned_attention(q[:,:,p:],k,v,mask[p:],p,.25)
    torch.testing.assert_close(suffix,actual[:,:,p:],atol=0,rtol=0)
    prefix=aligned_attention(q[:,:,:p],k[:,:,:p],v[:,:,:p],mask[:p,:p],0,.25)
    torch.testing.assert_close(prefix,actual[:,:,:p],atol=0,rtol=0)
