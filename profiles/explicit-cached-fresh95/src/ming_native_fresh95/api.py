"""Opt-in cached native ordered-frame fresh/FULL/RAW owners; no renderer."""
from copy import deepcopy
import hashlib,math
from pathlib import Path
import torch
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.pg095 import ConditionalPG095
from ming_native.native import learning_rate_overrides
from ming_native.ordered_frames import OrderedFrameCritic,FEATURE_PROFILE
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device,validate_base
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json

FULL_SCHEMA='Ming-native-ordered-explicit-cached-fresh95-v1-FULL'
RAW_SCHEMA='Ming-native-ordered-explicit-cached-fresh95-v1-RAW'
ORIGINAL_PROFILE='native-pg095-3dd672a7'

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit original ordered-native configuration required')
    required={'pg_version','rank','alpha','seed','response_shape','batch_size','steps','lr','d_lr_mult','critic_width'}
    if set(value)!=required or value['pg_version']!='0.9.5':raise ValueError('exact native PG095 adapter-only recipe required')
    for key,low,high in(('rank',1,128),('seed',0,2**63-101),('batch_size',2,512),('steps',1,10000),('critic_width',1,4096)):
        if type(value[key])is not int or not low<=value[key]<=high:raise ValueError('invalid recipe: '+key)
    if type(value['alpha'])not in(int,float)or not math.isfinite(value['alpha'])or value['alpha']<=0:
        raise ValueError('positive finite original LoRA alpha required')
    shape=value['response_shape']
    if (not isinstance(shape,list)or len(shape)!=5 or shape[0]!=1
            or any(type(n)is not int or n<1 for n in shape)or not 2<=shape[2]<=5
            or shape[-1]%2 or shape[-2]%2):
        raise ValueError('full native [1,C,F,H,W] with2..5 ordered targets and even spatial patches required')
    learning_rate_overrides(lr=value['lr'],d_lr_mult=value['d_lr_mult'],prior_lr_mult=None)
    canonical(value);return deepcopy(value)

def intent(mode,resume,cohort_id,fixture,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort_id,str)or not cohort_id.strip():
        raise ValueError('explicit fresh|resume and new cohort ID required')
    if (mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):
        raise ValueError('fresh refuses state; resume requires exact typed new FULL')
    if type(fixture)is not bool or ema_decay is not None:raise ValueError('explicit fixture/no weight EMA required')

def devices_for(device,fixture):
    device=torch.device(device)
    if device.type not in('cpu','cuda')or device.type=='cuda'and device.index is None:
        raise ValueError('explicit cpu or indexed CUDA device required; library grants no resources')
    if fixture and device.type!='cpu':raise ValueError('original Ming fixture contract is CPU-only')
    return device,[device.index]if device.type=='cuda'else[]

class ReceiptBackend:
    def __init__(self,native,receipt,sources,cache_identity):
        self.backend=native;self.network=native.network
        self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache_identity)
    def __getattr__(self,key):return getattr(self.backend,key)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=self.receipt,sources=self.sources),
            completed_native_cache=self.cache,particle_bank=None,router=None,teacher_collector_called=False,
            finished_RGBA_renderer_provided=False)

class NativeOwner(ownership.Owner):
    def response(self,row,*,strength=1.,seed=7):
        """Original ordered native velocity only; not a finished RGBA generator."""
        from particle_sliders.lifecycle import _network_observation
        from particle_sliders.pg09_backends import ResponseRow
        self.require_open()
        if type(seed)is not int or not 0<=seed<2**63-100 or type(strength)not in(int,float)or not math.isfinite(strength):
            raise ValueError('explicit finite response scale/seed required')
        if (type(row)is not ResponseRow or row.schedule is not None or row.guidance!=1.
                or not isinstance(row.latent,torch.Tensor)or row.latent.dtype!=torch.float32
                or list(row.latent.shape)!=self.config['response_shape']or row.latent.requires_grad
                or row.latent.device!=self.device
                or not torch.isfinite(row.latent).all()):
            raise ValueError('original full native ordered response geometry/conditioning required')
        with _network_observation(self.native.network,seed,devices=self.devices),torch.no_grad():
            value=self.native.velocity(row.neutral,row.latent,row.timestep,scale=strength)
            if list(value.shape)!=self.config['response_shape']or value.dtype!=torch.float32 or not torch.isfinite(value).all():
                raise RuntimeError('complete finite FP32 native ordered response required')
            return value

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,base_identity,device,
                 component_factory,owner_admission,resume=None,fixture=False,ema_decay=None):
        super().__init__()
        # Resolve operation/state/Source/data identities before model components,
        # teachers, adapter/critic/game construction or any fresh initializer.
        intent(mode,resume,cohort_id,fixture,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('explicit owner factory/admission required')
        config=configuration(config);validate_base(base_identity,fixture)
        self.device,self.devices=devices_for(device,fixture);self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        self.runtime_identity=ownership.guard();self.sources=ownership.sources()
        self.config,self.cohort_id=config,cohort_id
        self.options=dict(config=config,base_identity=deepcopy(base_identity),device=str(self.device),fixture=fixture,
            original_profile=ORIGINAL_PROFILE,feature_profile=FEATURE_PROFILE,targets=['Attention'],train_method='full')
        self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA
                    or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id or packet['options']!=self.options
                    or packet['sources']!=self.sources or packet['ema_decay']is not None):
                raise ValueError('exact typed new-profile FULL required; no old artifact migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=config,base_identity=base_identity,fixture=fixture)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('whole original completed cache differs')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),
            completed_cache=deepcopy(self.cache_identity),new_teacher_or_trajectory_calls=0,
            fresh_PG_initialization_calls=int(mode=='fresh'),particle_bank=None,router=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('external owner did not authenticate/admit this operation')
        try:
            with torch.random.fork_rng(devices=self.devices):
                self.model=component_factory(deepcopy(request))
                # Original adapter validates exact official classes/Source and
                # dtype/device; fixture is its original explicit CPU-only route.
                seed_initialization_rng(config['seed'],self.devices)
                self.attachment=ownership.Attachment.__new__(ownership.Attachment)
                ownership.Attachment.__init__(self.attachment,self.model)
                self.slider=self.attachment.attach(config['rank'],config['alpha'],self.device)
                self.native=self.attachment.bind(base_identity,fixture,ORIGINAL_PROFILE)
                if next(self.model.parameters()).device!=self.device:raise ValueError('owner staged the native model on a different device')
                with torch.random.fork_rng(devices=self.devices):
                    # Same original ordered critic constructor seed/law; KEEP.
                    seed_initialization_rng(config['seed'],self.devices)
                    self.critic=OrderedFrameCritic(config['response_shape'],conditions.shape[1],
                        feature_profile=FEATURE_PROFILE,width=config['critic_width']).to(self.device)
                self.initialization=(policy.initialize(self.slider,self.critic,self.construction,self.options,self.sources)
                    if mode=='fresh'else deepcopy(packet['initialization']))
            prepared=to_device(prepared,self.device);conditions=conditions.to(self.device)
            self.backend=ReceiptBackend(self.native,self.initialization,self.sources,self.cache_identity)
            # The real original shared engine owns optimizers/math/RNG. Receipt
            # already participates in its immutable signature before optimizers.
            self.game=ConditionalPG095(self.backend,prepared,self.critic,conditions=conditions,particles=None,
                total_steps=config['steps'],seed=config['seed'],batch_size=config['batch_size'],
                response_domain='continuous_field',response_scale=1.,sampling='balanced',preset='bispf_optimized',
                recipe_overrides=learning_rate_overrides(lr=config['lr'],d_lr_mult=config['d_lr_mult'],prior_lr_mult=None))
            if self.game.prior is not None:raise RuntimeError('original native lane must not gain a bank')
            if packet is not None:
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('whole typed game/optimizers/cursors/RNG restore differs')
            self.inherited_steps=self.game.completed_steps
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt')
            self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Ming entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):self.require_open();return self.game.step()
    def save(self,path):
        self.require_open()
        return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),model_identity=deepcopy(self.native.model_identity),
            network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__();packet=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','model_identity','network','completed_steps','selection','ema_decay'}
        if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA
                or packet['profile']!=PROFILE or packet['selection']!='raw'or packet['ema_decay']is not None
                or type(packet['completed_steps'])is not int or packet['completed_steps']<0
                or not callable(component_factory)or not callable(owner_admission)):
            raise ValueError('exact typed new-profile RAW and explicit owner required')
        ownership.guard();sources=ownership.sources()
        if packet['sources']!=sources:raise ValueError('RAW Source identity differs')
        options=packet['options'];config=configuration(options['config']);fixture=options['fixture']
        intent('fresh',None,packet['cohort_id'],fixture,None);validate_base(options['base_identity'],fixture)
        expected_options=dict(config=config,base_identity=options['base_identity'],device=options['device'],fixture=fixture,
            original_profile=ORIGINAL_PROFILE,feature_profile=FEATURE_PROFILE,targets=['Attention'],train_method='full')
        if options!=expected_options:raise ValueError('RAW original target/feature profile differs')
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(options).encode()).hexdigest())
        policy.validate_receipt(packet['initialization'],construction,options,sources)
        expected_identity=dict(options['base_identity'],qualification='CPU fixture only')if fixture else options['base_identity']
        if (packet['model_identity']!=expected_identity or packet['cache_identity']['base_identity']!=options['base_identity']
                or packet['completed_steps']>config['steps']):raise ValueError('RAW native/cache identity or update budget differs')
        self.device,self.devices=devices_for(options['device'],fixture);self.config=config
        self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(options),
            cache_metadata_only=deepcopy(packet['cache_identity']),fresh_PG_initialization_calls=0,
            new_teacher_or_trajectory_calls=0,particle_bank=None,router=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not admit native RAW response')
        try:
            with torch.random.fork_rng(devices=self.devices):
                self.model=component_factory(deepcopy(request));seed_initialization_rng(config['seed'],self.devices)
                self.attachment=ownership.Attachment.__new__(ownership.Attachment)
                ownership.Attachment.__init__(self.attachment,self.model)
                self.slider=self.attachment.attach(config['rank'],config['alpha'],self.device)
                self.native=self.attachment.bind(options['base_identity'],fixture,ORIGINAL_PROFILE)
                if next(self.model.parameters()).device!=self.device:raise ValueError('native owner device differs')
                self.native.network.load_state_dict(packet['network'],strict=True)
                if not equal(self.native.network.state_dict(),packet['network']):raise RuntimeError('strict native RAW tensors differ')
            self.completed_steps=packet['completed_steps']
            # No completed cache payload, critic, game, optimizer or initializer.
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Ming RAW entry and cleanup failed',[primary,cleanup])
            raise
