"""Four NEW authored/UNRUN native ordered-Ming software nodes; ONE tiny update.

Original native/core/PG/HNDL guards remain enabled. Tiny explicit CPU fixture
uses original conceptmod Modules and the original ordered HNDL graph. No old
tests, real teachers, preparation, pretrained weights or RGBA renderer run.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import pytest
import torch
from torch import nn
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow,FrozenResponseAdapter
from ming_native import native as original_native
from ming_native.constants import BASE_MODEL_ID
from ming_native_fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from ming_native_fresh95 import api,cache,policy,ownership
from ming_native_fresh95.util import equal

class Attention(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Linear(2,2)
    def forward(self,value):return self.proj(value)

class TinyMing(nn.Module):
    multi_frame_output=True
    def __init__(self):super().__init__();self.attn=Attention()
    def forward(self,hidden_states,timestep,encoder_hidden_states,*,return_dict,
                encoder_hidden_states_2=None,ref_hidden_states=None):
        output=[]
        for index,value in enumerate(hidden_states):
            result=self.attn(value.permute(1,2,3,0)).permute(3,0,1,2)
            result=result+encoder_hidden_states[index].mean()+timestep[index]
            if encoder_hidden_states_2 is not None:result=result+encoder_hidden_states_2[index].mean()
            if ref_hidden_states is not None:
                result=result+ref_hidden_states[index].mean()
                result=torch.cat((result,ref_hidden_states[index]),dim=1)
            output.append(result)
        return(output,)

def component_factory(request):
    assert request['operation']in('fresh','resume','inference')and request['particle_bank']is None
    assert request['new_teacher_or_trajectory_calls']==0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        return TinyMing().eval().requires_grad_(False)

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no preparation/teacher/pretrained renderer in NEW software cases')
    monkeypatch.setattr(FrozenResponseAdapter,'prepare',forbidden)
    monkeypatch.setattr(original_native.MingResponseAdapter,'prepare',forbidden)
    config=dict(pg_version='0.9.5',rank=2,alpha=2.,seed=7,response_shape=[1,2,2,4,4],
        batch_size=2,steps=3,lr=.001,d_lr_mult=18.,critic_width=4)
    base=component_factory(dict(operation='fresh',particle_bank=None,new_teacher_or_trajectory_calls=0))
    base_state=deepcopy(base.state_dict());base_path=tmp_path/'synthetic-original-base.pt'
    torch.save(base_state,base_path);bp=file_pin(base_path)
    identity=dict(base_model_id=BASE_MODEL_ID,revision='1'*40,upstream_commit=original_native.UPSTREAM_COMMIT,
        scope='synthetic-software-only',whole_files=[dict(path=bp.path,bytes=bp.bytes,sha256=bp.sha256)],constructor_seed=123)
    prepared=[]
    for index in range(2):
        z=torch.linspace(-.2,.2,64).reshape(1,2,2,4,4)+index*.01
        references=torch.full((1,2,1,4,4),.05)
        neutral=dict(embeds=torch.ones(1,2,3)*.1,embeds_2=None,reference_latents=references)
        positive=dict(embeds=torch.ones(1,2,3)*.3,embeds_2=None,reference_latents=references.clone())
        row=ResponseRow(neutral=neutral,positive=positive,negative=None,latent=z,
            timestep=torch.tensor(1000.-index*500.),guidance=1.,scale=1.,strength=1.,schedule=None)
        prepared.append(PreparedResponse(row,torch.full_like(z,.2+index*.01),torch.zeros_like(z)))
    original=tmp_path/'synthetic-completed-two-fields.pt';torch.save([cache.dump(x)for x in prepared],original)
    whole=file_pin(original);source=file_pin(__file__)
    origins=[dict(scope='synthetic-software-only',row_id='NEW-row-'+str(index),complete_prepared_output=True,
        whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],
        source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],real_teacher_calls=0)
        for index in range(2)]
    frozen=save_prepared_cache(tmp_path/'completed-cache.pt',config=config,base_identity=identity,prepared=prepared,
        conditions=torch.ones(2,1),teacher_origins=origins,cache_id='NEW-Ming-ordered-synthetic-two-rows',
        fixture=True,owner_admission=lambda request:True)
    return dict(config=config,cache=frozen,base_identity=identity,device='cpu',fixture=True,
        cohort_id='NEW-Ming-cached-fresh95-software-v1',component_factory=component_factory,
        owner_admission=lambda request:True),tmp_path,base_state

def opened(staged,name='fresh'):
    options,root,_=staged
    return open_training(directory=root/name,mode='fresh',**options)

def test_new_original_Parameters_policy_receipt_before_optimizers_zeroUP_critic_KEEP(staged,monkeypatch):
    assert ownership.installed_owner()['seven_module_RECORD_ownership_verified']is True
    initialize=policy.initialize;game=api.ConditionalPG095;seen=[]
    def initialized(slider,critic,*args):
        identities=[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        original=deepcopy(critic.state_dict());receipt=initialize(slider,critic,*args)
        assert identities==[id(p)for p in slider.parameters()]+[id(p)for p in critic.parameters()]
        assert equal(original,critic.state_dict())
        assert all(torch.count_nonzero(layer.lora_up.weight)==0 for layer in slider.unet_loras)
        seen.append(receipt);return receipt
    def constructed(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is None and kwargs['response_scale']==1.
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialized);monkeypatch.setattr(api,'ConditionalPG095',constructed)
    with opened(staged)as owner:
        assert owner.game.completed_steps==0 and owner.game.prior is None
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[0]
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
        assert {id(p)for p in owner.native.network.parameters()}=={id(p)for p in owner.slider.parameters()}
        for (module,_,_,owned),layer in zip(owner.attachment.pending_forwards,owner.slider.unet_loras):
            assert module.forward is owned and owned.__self__ is layer
        assert owner.native.metadata()['ordered_frame_critic']['mapping']=='packed_index=channel*target_frames+frame'

def test_new_a1_QR_moment_deterministic_KEEP_base_and_caller_RNG(staged):
    before=torch.get_rng_state().clone();_,_,base=staged
    with opened(staged,'a')as owner:
        state=deepcopy(owner.native.network.state_dict());receipt=deepcopy(owner.initialization)
        model=owner.model;flags=owner.attachment.flags
        for layer in owner.slider.unet_loras:
            down=layer.lora_down.weight.detach().flatten(1);rows,cols=down.shape;variance=1/cols
            gram=down@down.T if rows<=cols else down.T@down
            assert torch.allclose(gram,torch.eye(min(rows,cols))*variance*max(rows,cols),rtol=1e-5,atol=1e-6)
            assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
            assert torch.count_nonzero(layer.lora_up.weight)==0
        assert equal(base,model.state_dict())
    assert all(p.requires_grad==flag for p,flag in flags)
    assert all('_ming_explicit_fresh95_owner'not in vars(m)and'_ming_response_binding'not in vars(m)for m in model.modules())
    assert all('forward'not in vars(m)for _,m in owner.attachment.targets)
    assert torch.equal(before,torch.get_rng_state())
    with opened(staged,'b')as again:
        assert equal(state,again.native.network.state_dict())and receipt==again.initialization
        assert all(p['before_sha256']==p['after_sha256']for p in receipt['parameters']if p['role']!='host-a1-down-QR')
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_whole_FULL_optimizers_RNG_and_RAW_native_no_init(staged,monkeypatch):
    options,root,base=staged
    with opened(staged)as trained:
        trained.update()  # Exactly ONE total tiny update across the four nodes.
        full=trained.save(root/'saved-FULL1.pt');raw=trained.export(root/'saved-RAW1.pt')
        state=deepcopy(trained.game.state_dict());network=deepcopy(trained.native.network.state_dict())
        row=trained.game.rows[0].row;model=trained.model
    assert equal(base,model.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('restore/inference must skip PG initializer/cache/game/teacher')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,component_factory=component_factory,owner_admission=lambda request:True)as inferred:
        assert not hasattr(inferred,'game')and not hasattr(inferred,'critic')
        assert equal(network,inferred.native.network.state_dict())
        before=torch.get_rng_state().clone();output=inferred.response(row,strength=1.,seed=17)
        assert output.shape==(1,2,2,4,4)and torch.isfinite(output).all()and torch.equal(before,torch.get_rng_state())
        assert equal(base,inferred.model.state_dict())

def test_new_early_intent_cache_EMA_refusal_and_retryable_foreign_safe_cleanup(staged,monkeypatch):
    options,root,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('bad intent reached native component factory')
    bad=dict(options,component_factory=forbidden)
    for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):open_training(directory=root/'bad',mode=mode,resume=resume,ema_decay=ema,**bad)
    legacy=root/'old-FULL.pt';torch.save(dict(schema='old-Ming-full2'),legacy)
    with pytest.raises(ValueError):open_training(directory=root/'legacy',mode='resume',resume=file_pin(legacy),**bad)
    with pytest.raises(ValueError):
        open_training(directory=root/'denied',mode='fresh',**dict(options,component_factory=forbidden,owner_admission=lambda request:False))
    assert calls==[]
    # Wrong native frames cannot be silently flattened, averaged or rebuilt.
    cache_pin=options['cache'];changed=replace(cache_pin,sha256='0'*64)
    with pytest.raises(ValueError):open_training(directory=root/'wrong-cache',mode='fresh',**dict(bad,cache=changed))
    assert calls==[]
    owner=opened(staged,'cleanup');original=owner.attachment._restore_forward;attempts=[]
    def fail_once(item):
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('deliberate fail-once owned native forward cleanup')
        return original(item)
    monkeypatch.setattr(owner.attachment,'_restore_forward',fail_once)
    with pytest.raises(RuntimeError,match='fail-once'):owner.close()
    assert owner.closing and not owner.closed and len(owner.attachment.pending_forwards)==1
    with pytest.raises(RuntimeError,match='cleanup'):owner.update()
    with pytest.raises(RuntimeError,match='cleanup'):owner.response(None)
    owner.close();assert owner.closed and len(attempts)==2
    another=opened(staged,'foreign');module,_,_,owned=another.attachment.pending_forwards[0]
    foreign=lambda *args,**kwargs:None;module.forward=foreign
    with pytest.raises(RuntimeError,match='foreign Ming forward'):another.close()
    assert module.forward is foreign and another.closing and not another.closed
    module.forward=owned;another.close();assert another.closed

    # A failure AFTER callback removal must retain every exclusive owner marker.
    late=opened(staged,'late-mode');restore_mode=late.attachment._restore_mode;tries=[]
    def fail_mode_once(item):
        tries.append(item)
        if len(tries)==1:raise RuntimeError('deliberate fail-once owned mode cleanup')
        return restore_mode(item)
    monkeypatch.setattr(late.attachment,'_restore_mode',fail_mode_once)
    with pytest.raises(RuntimeError,match='fail-once owned mode'):late.close()
    assert late.closing and not late.closed and len(late.attachment.pending_modes)==1
    assert late.attachment.pending_owner and all(vars(m).get('_ming_explicit_fresh95_owner')is late.attachment for m,_,_ in late.attachment.pending_owner)
    with pytest.raises(RuntimeError,match='cleanup'):late.response(None)
    late.close();assert late.closed and not late.attachment.pending_owner
    # A later foreign checkpoint callback is never overwritten by owned cleanup.
    callback=opened(staged,'foreign-callback');module=callback.model
    original_callback=lambda *args,**kwargs:None;foreign_callback=lambda *args,**kwargs:None
    module._gradient_checkpointing_func=foreign_callback
    callback.attachment.pending_native.append((module,'_gradient_checkpointing_func',False,None,True,original_callback))
    with pytest.raises(RuntimeError,match='foreign Ming checkpoint'):callback.close()
    assert module._gradient_checkpointing_func is foreign_callback and callback.attachment.pending_owner
    module._gradient_checkpointing_func=original_callback;callback.close();assert callback.closed
    # Finite-scale semantics match the original native velocity API: no invented cap.
    # The invalid seed is refused before observing even a fabricated response row.
    finite=opened(staged,'finite-scale')
    with pytest.raises(ValueError,match='scale/seed'):finite.response(None,strength=2.,seed=-1)
    finite.close()

    marked=opened(staged,'foreign-marker');module=marked.model
    forward=marked.attachment.pending_forwards[0][3];foreign_marker=object()
    module._ming_explicit_fresh95_owner=foreign_marker
    with pytest.raises(RuntimeError,match='foreign Ming ownership marker'):marked.close()
    assert module._ming_explicit_fresh95_owner is foreign_marker
    assert marked.attachment.pending_forwards[0][0].forward is forward
    module._ming_explicit_fresh95_owner=marked.attachment;marked.close();assert marked.closed
    flagged=opened(staged,'foreign-flag');parameter=flagged.attachment.flags[0][0]
    parameter.requires_grad_(True)
    with pytest.raises(RuntimeError,match='foreign Ming requires-grad'):flagged.close()
    assert parameter.requires_grad and flagged.attachment.pending_owner
    parameter.requires_grad_(False);flagged.close();assert flagged.closed
