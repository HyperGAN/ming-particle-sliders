"""Original strict native admission and retryable instance-owned attachments."""
from pathlib import Path
from .util import source_record

def installed_owner():
    """Require a genuine separate installed helper, never a Source overlay."""
    import base64,hashlib
    from importlib import metadata
    distribution=metadata.distribution('ming-particle-sliders-explicit-fresh95')
    if (distribution.metadata['Name']!='ming-particle-sliders-explicit-fresh95'
            or distribution.version!='0.1.0'):
        raise RuntimeError('exact installed Ming fresh95 helper distribution required')
    root=Path(__file__).absolute().parent
    names=('__init__.py','api.py','baseline.py','cache.py','ownership.py','policy.py','util.py')
    recorded={str(row):row for row in distribution.files or ()}
    for name in names:
        relative='ming_native_fresh95/'+name;row=recorded.get(relative)
        target=root/name
        if (row is None or Path(distribution.locate_file(relative)).absolute()!=target
                or row.hash is None or row.hash.mode!='sha256'):
            raise RuntimeError('genuine installed helper RECORD ownership required: '+relative)
        data=target.read_bytes()
        digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        if row.size!=len(data) or row.hash.value!=digest:
            raise RuntimeError('installed helper Source differs from genuine RECORD: '+relative)
    return dict(name=distribution.metadata['Name'],version=distribution.version,
        seven_module_RECORD_ownership_verified=True,Source_overlay_allowed=False)

def guard():
    installed_owner()
    from ming_native import cohort,native,ordered_frames
    from particle_sliders import pg09
    from particlegan import init as pg_init
    from importlib import metadata
    from .baseline import ORIGINAL_SOURCES,CORE_SOURCES
    from .policy import INIT_SHA
    root=Path(cohort.__file__).absolute().parent
    for name,pin in ORIGINAL_SOURCES.items():
        if source_record(root/name)!=pin:raise RuntimeError('original Ming native Source differs: '+name)
    core=Path(pg09.__file__).absolute().parent
    for name,pin in CORE_SOURCES.items():
        if source_record(core/name)!=pin:raise RuntimeError('exact original core013 Source differs: '+name)
    require=cohort.require_cohort
    if (getattr(require,'__module__',None)!='ming_native.cohort'
            or getattr(require,'__globals__',None)is not vars(cohort)
            or native.require_cohort is not require or ordered_frames.require_cohort is not require):
        raise RuntimeError('original Ming admission functions required; historical current-core bridge refused')
    # All original installed product, core origin/recursive closure, PG, HNDL,
    # HyperGAN and dependency checks execute. No admission monkeypatch/overlay.
    identity=require('native-pg095-3dd672a7')
    if metadata.version('particlegan')!='0.9.5' or source_record(pg_init.__file__)['sha256']!=INIT_SHA:
        raise RuntimeError('exact public PG095 initializer Source required')
    return identity

def sources():
    from ming_native import cohort
    from particle_sliders import pg09
    from .baseline import ORIGINAL_SOURCES,CORE_SOURCES
    root=Path(cohort.__file__).absolute().parent;core=Path(pg09.__file__).absolute().parent
    return dict(own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original_native={n:source_record(root/n)for n in ORIGINAL_SOURCES},
        original_core={n:source_record(core/n)for n in CORE_SOURCES})

class Attachment:
    """Preserve native Modules/Parameters, prior forwards and attribute presence.

    The caller owns base model lifetime/device. Failed cleanup remains pending;
    a later foreign callback/marker is refused and retained, never overwritten.
    """
    def __init__(self,model):
        from particle_sliders.native_lora import lora_target_plan
        self.model=model;self.closed=self.closing=False;self.slider=None;self.native=None
        self.pending_native=[];self.pending_forwards=[];self.pending_flags=[];self.pending_modes=[]
        self.pending_owner=[];self.targets=[];self.named_ids=[]
        self.flags=[(p,p.requires_grad)for p in model.parameters()]
        self.modes=[(m,m.training)for m in model.modules()]
        self.pending_flags=[(p,old,p.requires_grad)for p,old in self.flags];self.pending_modes=[(m,old,m.training)for m,old in self.modes]
        for module in model.modules():
            if vars(module).get('_ming_explicit_fresh95_owner')is not None:
                raise ValueError('foreign explicit Ming owner already attached')
            previous=getattr(module,'_ming_response_binding',None)
            if previous is not None and (not previous._closed or previous._cleanup_failed):
                raise ValueError('foreign original Ming response owner already attached')
        self.targets=lora_target_plan(model,['Attention'],train_method='full')
        if not self.targets:raise ValueError('original native Attention LoRA target set is empty')
        # Refuse an instance override before taking ownership. We never remove
        # another wrapper/compiled callback or infer that it is safe to replace.
        if any('forward'in vars(m)for _,m in self.targets):
            raise ValueError('foreign target forward override retained')
        self.original_forwards=[(m,'forward'in vars(m),vars(m).get('forward'),m.forward)for _,m in self.targets]
        self.pending_owner=[(m,'_ming_explicit_fresh95_owner'in vars(m),vars(m).get('_ming_explicit_fresh95_owner'))for m in model.modules()]
        for m,_,_ in self.pending_owner:m._ming_explicit_fresh95_owner=self

    def attach(self,rank,alpha,device):
        from particle_sliders.native_lora import LoRANetwork
        self.model.requires_grad_(False)
        # Keep a partial original constructor reachable if it fails mid-attach.
        self.slider=LoRANetwork.__new__(LoRANetwork)
        try:
            LoRANetwork.__init__(self.slider,self.model,rank=rank,alpha=alpha,target_replace=['Attention'],train_method='full')
            self.slider.to(device=device)
        finally:
            self.pending_flags=[(p,old,p.requires_grad)for p,old in self.flags]
            self.pending_forwards=[]
            for module,present,old,original in self.original_forwards:
                current=vars(module).get('forward')
                layer=getattr(current,'__self__',None)
                if layer in getattr(self.slider,'unet_loras',()) and getattr(layer,'org_forward',None)==original:
                    self.pending_forwards.append((module,present,old,current))
        self.named_ids=[(n,id(p))for n,p in self.slider.named_parameters()]
        return self.slider

    def bind(self,identity,fixture,profile):
        from ming_native.ordered_frames import OrderedFrameResponseAdapter,FEATURE_PROFILE
        self.native=OrderedFrameResponseAdapter.__new__(OrderedFrameResponseAdapter)
        try:
            OrderedFrameResponseAdapter.__init__(self.native,self.model,self.slider,
                model_identity=identity,fixture=fixture,cohort=profile,feature_profile=FEATURE_PROFILE)
        finally:
            # Own the exact original binding's borrowed state, including absent
            # checkpoint flags. Cleanup below retries only unfinished actions.
            self.pending_native=[]
            for module,training,state in getattr(self.native,'_borrowed',()):
                for key,(present,old)in state.items():
                    now_present=key in vars(module);now=vars(module).get(key)
                    if (now_present,now)==(present,old):continue
                    self.pending_native.append((module,key,present,old,now_present,now))
            self.pending_modes=[(m,old,m.training)for m,old in self.modes]
        return self.native

    def require_active(self):
        if self.closed or self.closing:raise RuntimeError('Ming owner closed/cleanup pending')
        if self.slider is None or [(n,id(p))for n,p in self.slider.named_parameters()]!=self.named_ids:
            raise RuntimeError('original LoRA Parameter identity differs')
        if any(vars(m).get('_ming_explicit_fresh95_owner')is not self for m,_,_ in self.pending_owner):
            raise RuntimeError('Ming owner marker changed')
        if any(vars(m).get('forward')is not owned for m,_,_,owned in self.pending_forwards):
            raise RuntimeError('Ming native forward ownership changed')
        if self.native is not None:self.native._require_open()

    @staticmethod
    def _run(pending,fn,errors):
        failed=[]
        for item in pending:
            try:fn(item)
            except BaseException as error:failed.append(item);errors.append(error)
        return failed

    def _restore_native(self,item):
        module,key,present,old,owned_present,owned=item
        if (key in vars(module))!=owned_present or owned_present and vars(module)[key]is not owned:
            raise RuntimeError('foreign Ming checkpoint/binding attribute retained: '+key)
        if present:setattr(module,key,old)
        elif key in vars(module):delattr(module,key)

    def _restore_forward(self,item):
        module,present,old,owned=item
        if vars(module).get('forward')is not owned:raise RuntimeError('foreign Ming forward retained')
        if present:setattr(module,'forward',old)
        else:delattr(module,'forward')

    def _restore_flag(self,item):
        parameter,original,owned=item
        if parameter.requires_grad is not owned:raise RuntimeError('foreign Ming requires-grad state retained')
        parameter.requires_grad_(original)

    def _restore_mode(self,item):
        module,original,owned=item
        if module.training is not owned:raise RuntimeError('foreign Ming module training mode retained')
        module.training=original

    def close(self):
        if self.closed:return
        self.closing=True;errors=[]
        # Refuse a foreign marker BEFORE changing any borrowed model state.
        if any(vars(m).get('_ming_explicit_fresh95_owner')is not self for m,_,_ in self.pending_owner):
            raise RuntimeError('foreign Ming ownership marker retained before cleanup')
        if self.native is not None:self.native._closed=True;self.native._cleanup_failed=True
        self.pending_native=self._run(self.pending_native,self._restore_native,errors)
        if not self.pending_native:self.pending_forwards=self._run(self.pending_forwards,self._restore_forward,errors)
        # Keep the exclusive model marker until ALL borrowed state is restored.
        # Failed actions stay pending; successful actions are never repeated.
        if not self.pending_native and not self.pending_forwards:
            self.pending_flags=self._run(self.pending_flags,self._restore_flag,errors)
            self.pending_modes=self._run(self.pending_modes,self._restore_mode,errors)
        if not any((self.pending_native,self.pending_forwards,self.pending_flags,self.pending_modes)):
            def marker(item):
                m,present,old=item
                if vars(m).get('_ming_explicit_fresh95_owner')is not self:raise RuntimeError('foreign Ming ownership marker retained')
                if present:setattr(m,'_ming_explicit_fresh95_owner',old)
                else:delattr(m,'_ming_explicit_fresh95_owner')
            self.pending_owner=self._run(self.pending_owner,marker,errors)
        if errors:
            if len(errors)==1:raise errors[0]
            raise BaseExceptionGroup('Ming cleanup pending',errors)
        if any((self.pending_native,self.pending_forwards,self.pending_owner,self.pending_flags,self.pending_modes)):
            raise RuntimeError('Ming cleanup remains pending')
        if self.native is not None:self.native._cleanup_failed=False
        self.closed,self.closing=True,False

class Owner:
    def __init__(self):self.closed=self.closing=False;self.attachment=None
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('Ming owner closed/cleanup pending')
        if self.attachment is not None:self.attachment.require_active()
    def close(self):
        if self.closed:return
        self.closing=True
        if self.attachment is not None:self.attachment.close()
        self.closed,self.closing=True,False
    def __enter__(self):self.require_open();return self
    def __exit__(self,kind,primary,traceback):
        try:self.close()
        except BaseException as cleanup:
            if primary is not None:raise BaseExceptionGroup('Ming work and cleanup failed',[primary,cleanup])
            raise
