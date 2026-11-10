"""Public PG095 QR on existing conceptmod down Parameters at host a=1 RMS."""
from dataclasses import dataclass
from importlib import metadata
import hashlib,math
import torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical,source_record

INIT_SHA='3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA='Ming-ordered-native-explicit-a1-downQR-zeroUP-criticKEEP-initialization-v1'
MARKER='_ming_native_explicit_fresh95_receipt'

@dataclass(frozen=True)
class Construction:
    mode:str
    cohort:str
    options_sha256:str

class DeclaredOwned(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for index,row in enumerate(rows):
            key='p'+str(index);self.register_parameter(key,row['parameter']);self.specs[key]=row['distribution']

pg_init.register(DeclaredOwned,lambda module:module.specs)

def tensor_sha(value):
    flat=torch.empty(value.numel(),dtype=value.dtype,device='cpu')
    flat.copy_(value.detach().reshape(-1).cpu())
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()

def seed_for(seed,name):
    return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8],'big')%2**63

def declarations(slider,critic):
    from particle_sliders.native_lora import LoRANetwork,LoRAModule
    from ming_native.ordered_frames import OrderedFrameCritic
    if type(slider)is not LoRANetwork or type(critic)is not OrderedFrameCritic or not slider.unet_loras:
        raise ValueError('exact original conceptmod LoRA and ordered HNDL critic required')
    rows=[];seen=set();storages=set()
    for layer in slider.unet_loras:
        if type(layer)is not LoRAModule:raise ValueError('original native LoRAModule required')
        for role,leaf in(('down',layer.lora_down),('up',layer.lora_up)):
            if (type(leaf)not in(nn.Linear,nn.Conv1d,nn.Conv2d)or leaf.bias is not None
                    or any(leaf.children())or list(leaf.buffers())or nn.utils.parametrize.is_parametrized(leaf)):
                raise ValueError('original bias-free unparametrized native LoRA down/up required')
            p=leaf.weight;fan_in=math.prod(p.shape[1:])
            if role=='up'and torch.count_nonzero(p).item():raise ValueError('genuine fresh zero-UP required')
            # Original native_lora constructor is kaiming_uniform_(a=1): gain1,
            # U(+-sqrt(3/fan_in)), mean-square1/fan_in. QR need not stay within
            # the old Uniform entrywise support; it preserves its second moment.
            bound=math.sqrt(3/fan_in)
            rows.append(dict(name='adapter.'+layer.lora_name+'.lora_'+role+'.weight',parameter=p,
                distribution=pg_init.Uniform(-bound,bound)if role=='down'else pg_init.KEEP,
                fan_in=fan_in,role='host-a1-down-QR'if role=='down'else'protected-zero-UP-KEEP'))
    rows.extend(dict(name='critic.'+name,parameter=p,distribution=pg_init.KEEP,fan_in=None,
        role='original-ordered-HNDL-critic-constructor-KEEP')for name,p in critic.named_parameters())
    for row in rows:
        p=row['parameter'];storage=(str(p.device),p.untyped_storage().data_ptr())
        if (type(p)is not nn.Parameter or not p.requires_grad or p.grad is not None or p.dtype!=torch.float32
                or not p.is_contiguous()or not torch.isfinite(p).all()or id(p)in seen
                or p.numel()and storage in storages or hasattr(p,MARKER)):
            raise ValueError('unique finite fresh FP32 owned Parameters required')
        seen.add(id(p));storages.add(storage)
    adapter={id(p)for p in slider.parameters()};d={id(p)for p in critic.parameters()}
    if seen!=adapter|d or adapter&d:raise ValueError('exact adapter/critic declaration coverage required; no bank/router')
    return rows

def initialize(slider,critic,construction,options,sources):
    if (type(construction)is not Construction or construction.mode!='fresh'or not construction.cohort
            or construction.options_sha256!=hashlib.sha256(canonical(options).encode()).hexdigest()):
        raise ValueError('explicit new fresh intent before optimizers required')
    if metadata.version('particlegan')!='0.9.5'or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise ValueError('exact public PG095 initializer Source required')
    rows=declarations(slider,critic);before=[tensor_sha(x['parameter'])for x in rows]
    ids=[id(x['parameter'])for x in rows];holder=DeclaredOwned(rows)
    seeds={'p'+str(i):seed_for(options['config']['seed'],row['name'])for i,row in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder,strict=True,parameter_seeds=seeds)
    if ids!=[id(x['parameter'])for x in rows]:raise RuntimeError('original Parameter replaced')
    parameters=[]
    for index,row in enumerate(rows):
        after=tensor_sha(row['parameter'])
        if row['distribution']is pg_init.KEEP and before[index]!=after:raise RuntimeError('zero-UP/critic KEEP changed')
        parameters.append(dict(name=row['name'],role=row['role'],seed=seeds['p'+str(index)],
            shape=list(row['parameter'].shape),dtype=str(row['parameter'].dtype),fan_in=row['fan_in'],
            distribution=repr(row['distribution']),before_sha256=before[index],after_sha256=after))
    receipt=dict(schema=SCHEMA,profile=PROFILE,cohort=construction.cohort,
        options_sha256=construction.options_sha256,completed_steps=0,restored_from=None,
        source_identity=sources,seed=options['config']['seed'],initializer=source_record(pg_init.__file__),
        parameters=parameters,parameter_identity_preserved=True,particle_bank=None,router=None,
        critic_policy='original-ordered-HNDL-constructor-KEEP',host_down_kaiming_a=1,
        host_down_target_mean_square='1/fan_in',selection='raw',ema_decay=None)
    marker=hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:setattr(row['parameter'],MARKER,marker)
    return receipt

def validate_receipt(receipt,construction,options,sources):
    if (not isinstance(receipt,dict)or receipt.get('schema')!=SCHEMA or receipt.get('profile')!=PROFILE
            or receipt.get('cohort')!=construction.cohort or receipt.get('options_sha256')!=construction.options_sha256
            or receipt.get('source_identity')!=sources or receipt.get('seed')!=options['config']['seed']
            or receipt.get('completed_steps')!=0 or receipt.get('restored_from')is not None
            or receipt.get('particle_bank')is not None or receipt.get('router')is not None
            or receipt.get('ema_decay')is not None or receipt.get('selection')!='raw'
            or receipt.get('initializer',{}).get('sha256')!=INIT_SHA):
        raise ValueError('exact new-profile original initialization receipt required')
