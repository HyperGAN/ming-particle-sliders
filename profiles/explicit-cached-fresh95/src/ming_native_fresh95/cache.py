"""Whole-file binding of already completed ordered native Ming responses."""
from dataclasses import fields
import math
from pathlib import Path
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from ming_native.constants import BASE_MODEL_ID
from ming_native.native import UPSTREAM_COMMIT,_check_condition
from ming_native.ordered_frames import FEATURE_PROFILE
from . import PROFILE,ownership
from .util import canonical,read_packet,save_packet,source_record,tree

SCHEMA='Ming-completed-ordered-native-prepared-cache-fresh95-v1'

def file_record(row):
    if (not isinstance(row,dict)or set(row)!={'path','bytes','sha256'}
            or not isinstance(row['path'],str)or not row['path']or not Path(row['path']).is_absolute()
            or '..'in Path(row['path']).parts or type(row['bytes'])is not int or row['bytes']<1
            or not isinstance(row['sha256'],str)or len(row['sha256'])!=64
            or any(c not in'0123456789abcdef'for c in row['sha256'])):
        raise ValueError('explicit whole-file byte/SHA provenance required')

def validate_base(identity,fixture):
    if (not isinstance(identity,dict)or identity.get('base_model_id')!=BASE_MODEL_ID
            or not isinstance(identity.get('revision'),str)or len(identity['revision'])!=40
            or any(c not in'0123456789abcdef'for c in identity['revision'])
            or identity.get('upstream_commit')!=UPSTREAM_COMMIT
            or identity.get('scope')!=('synthetic-software-only'if fixture else'pretrained-native')
            or not isinstance(identity.get('whole_files'),list)or not identity['whole_files']):
        raise ValueError('original native model/revision/Source and honest scope required')
    for row in identity['whole_files']:file_record(row)
    canonical(identity)

def teacher_configuration(config):
    return dict(response_shape=config['response_shape'],ordered_frame_profile=FEATURE_PROFILE,
        scale=1.,strength=1.,guidance=1.,condition_profile='original-constant-one-v1')

def finite(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.layout!=torch.strided or value.is_complex()or not torch.isfinite(value).all():
            raise ValueError('frozen dense finite completed tensors required')
    elif isinstance(value,dict):
        if any(not isinstance(k,str)for k in value):raise ValueError('string frozen cache keys required')
        for item in value.values():finite(item)
    elif isinstance(value,(tuple,list)):
        for item in value:finite(item)
    elif value is not None and type(value)not in(str,int,float,bool):raise ValueError('unsupported cached object')
    elif type(value)is float and not math.isfinite(value):raise ValueError('finite cache scalars required')

def validate_prepared(prepared,conditions,config):
    if not isinstance(prepared,(list,tuple))or len(prepared)<2:raise ValueError('at least two complete original native rows required')
    shape=tuple(config['response_shape'])
    for item in prepared:
        if type(item)is not PreparedResponse or type(item.row)is not ResponseRow:
            raise ValueError('original PreparedResponse/ResponseRow required; no packed record reinterpretation')
        row=item.row
        if (row.schedule is not None or row.scale!=1. or row.strength!=1. or row.guidance!=1.
                or not isinstance(row.timestep,torch.Tensor)or row.timestep.numel()!=1
                or not row.timestep.is_floating_point()or not 0<=float(row.timestep)<=1000):
            raise ValueError('original unipolar raw single-timestep native contract required')
        for value in(row.latent,item.neutral,item.target):
            if not isinstance(value,torch.Tensor)or tuple(value.shape)!=shape or value.dtype!=torch.float32:
                raise ValueError('all ordered native target frames and full FP32 velocity fields required')
        for condition in(row.neutral,row.positive):_check_condition(condition,row.latent)
        if row.negative is not None:_check_condition(row.negative,row.latent)
        for value in(row.latent,row.timestep,row.neutral,row.positive,row.negative,item.target,item.neutral):finite(value)
    # The actual original Ming route supplied one constant feature per row.
    # It is bound verbatim rather than deriving a new semantic feature vector.
    if (not isinstance(conditions,torch.Tensor)or conditions.shape!=(len(prepared),1)
            or conditions.dtype!=torch.float32 or conditions.requires_grad
            or not torch.equal(conditions,torch.ones_like(conditions))):
        raise ValueError('exact original FP32 constant-one conditions required')
    return shape

def dump(item):
    return dict(row={f.name:tree(getattr(item.row,f.name),'cpu')for f in fields(ResponseRow)},
        target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))

def restore(value,device):
    if (not isinstance(value,dict)or set(value)!={'row','target','neutral'}
            or set(value['row'])!={f.name for f in fields(ResponseRow)}):
        raise ValueError('complete original serialized row required')
    return PreparedResponse(ResponseRow(**tree(value['row'],device)),tree(value['target'],device),tree(value['neutral'],device))

def validate_origins(origins,prepared,fixture):
    if not isinstance(origins,list)or len(origins)!=len(prepared):raise ValueError('each completed original row requires provenance')
    ids=[]
    for origin in origins:
        if (not isinstance(origin,dict)or origin.get('scope')!=('synthetic-software-only'if fixture else'completed-native')
                or origin.get('complete_prepared_output')is not True
                or not isinstance(origin.get('row_id'),str)or not origin['row_id']
                or not isinstance(origin.get('whole_output_files'),list)or not origin['whole_output_files']
                or not isinstance(origin.get('source_files'),list)or not origin['source_files']):
            raise ValueError('complete row identity, whole-file and Source origins required')
        ids.append(origin['row_id'])
        for row in origin['whole_output_files']+origin['source_files']:file_record(row)
    if len(set(ids))!=len(ids):raise ValueError('distinct original cache row IDs required')
    canonical(origins)

def save_prepared_cache(path,*,config,base_identity,prepared,conditions,teacher_origins,
                        cache_id,owner_admission,fixture=False):
    """Serialize caller-owned completed fields only; no prepare/teacher API.

    The external owner authenticates original output/base whole-file records.
    This library binds those attestations; it does not prove loaded base weights.
    """
    from .api import configuration
    config=configuration(config);ownership.guard()
    if type(fixture)is not bool:raise ValueError('explicit fixture boolean required')
    validate_base(base_identity,fixture);validate_prepared(prepared,conditions,config)
    validate_origins(teacher_origins,prepared,fixture)
    if not isinstance(cache_id,str)or not cache_id.strip()or not callable(owner_admission):raise ValueError('cache ID/external owner required')
    request=dict(operation='serialize-completed-cache',profile=PROFILE,cache_id=cache_id,
        base_identity=base_identity,teacher_origins=teacher_origins,model_or_teacher_calls=0)
    if owner_admission(request)is not True:raise ValueError('owner did not authenticate completed origins')
    return save_packet(path,dict(schema=SCHEMA,profile=PROFILE,cache_id=cache_id,
        teacher_configuration=teacher_configuration(config),base_identity=base_identity,fixture=fixture,
        original_sources={k:v for k,v in ownership.sources().items()if k!='own'},
        serializer_source=source_record(__file__),teacher_origins=teacher_origins,
        rows=[dump(x)for x in prepared],conditions=tree(conditions,'cpu'),complete=True,model_or_teacher_calls=0))

def load_cache(pin,*,config,base_identity,fixture):
    packet=read_packet(pin)
    keys={'schema','profile','cache_id','teacher_configuration','base_identity','fixture','original_sources',
        'serializer_source','teacher_origins','rows','conditions','complete','model_or_teacher_calls'}
    if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA or packet['profile']!=PROFILE
            or packet['fixture']is not fixture or packet['complete']is not True or packet['model_or_teacher_calls']!=0
            or packet['teacher_configuration']!=teacher_configuration(config)or packet['base_identity']!=base_identity
            or packet['original_sources']!={k:v for k,v in ownership.sources().items()if k!='own'}
            or packet['serializer_source']!=source_record(__file__)):
        raise ValueError('completed original cache Source/configuration/identity differs')
    prepared=[restore(x,'cpu')for x in packet['rows']]
    shape=validate_prepared(prepared,packet['conditions'],config);validate_origins(packet['teacher_origins'],prepared,fixture)
    identity={k:packet[k]for k in('schema','profile','cache_id','teacher_configuration','base_identity','fixture',
        'original_sources','serializer_source','teacher_origins')}
    identity.update(whole_file=dict(path=pin.path,bytes=pin.bytes,sha256=pin.sha256),
        row_count=len(prepared),response_shape=list(shape),condition_shape=list(packet['conditions'].shape))
    return prepared,packet['conditions'],identity

def to_device(prepared,device):return[restore(dump(x),device)for x in prepared]
