"""Player-only CNN/LSTM policy with variable, exactly masked candidate menus.

This module does not decide which game actions are legal. Its input is the
published menu from the native adapter, including explicit unsupported cases.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.distributions import Categorical

ACTOR_BATCH_KEYS = frozenset({'map','map_mask','map_categorical','entities','entity_mask','candidates','action_mask','task_unit_types',
                             'entity_categorical','action_categorical','global','task','global_categorical',
                             'policy_catalogs','policy_action_indices', 'dialogue_context',
                             'dialogue_menu_text', 'dialogue_menu_mask', 'dialogue_menu_flags',
                             'dialogue_present', 'dialogue_candidate_rows', 'candidate_source_rows',
                             'movement_points','movement_point_mask','movement_candidate_rows'})


@dataclass(frozen=True)
class PolicyConfig:
    map_channels: int
    global_features: int
    entity_features: int
    action_features: int
    vocabulary_size: int = 1
    hidden_size: int = 128
    spatial_size: int = 32
    category_size: int = 16
    global_categories: int = 2
    task_features: int = 12
    text_embedding_size: int = 16
    text_channels: int = 32
    map_categories: int = 2
    # Contract identity only; it is serialized but never changes layer widths.
    actor_schema: str | None = None
    # Explicit base measure of the root categorical policy. None preserves the
    # historical flat-candidate semantics and serialized checkpoint shape.
    candidate_family_prior: str | None = None


FAMILY_PRIOR = 'inverse_legal_candidate_count_v1'
FAMILY_PRIOR_CONTRACT = {
    'id': FAMILY_PRIOR,
    'family': 'published_action_kind_one_hot',
    'count': 'action_mask_true_only',
    'logit_adjustment': 'subtract_log_family_count',
    'support': 'unchanged',
    'entropy': 'shannon_of_corrected_joint_policy',
}


def family_prior_logits(logits, kind_features, action_mask):
    """Correct candidate multiplicity using only the published legal menu.

    Equal logits yield equal mass per present kind, then uniform choices within
    that kind. Learned logits may override this prior. This is neither a new
    legality test nor a guarantee of useful exploration or improved play.
    """
    if (action_mask.dtype != torch.bool or logits.shape != action_mask.shape
            or kind_features.ndim != 3 or kind_features.shape[:2] != logits.shape
            or kind_features.shape[-1] == 0):
        raise ValueError('family prior requires matching root menu tensors')
    legal = kind_features[action_mask]
    if (not action_mask.any(dim=-1).all() or not torch.isfinite(legal).all()
            or not ((legal == 0) | (legal == 1)).all()
            or not (legal.sum(dim=-1) == 1).all()):
        raise ValueError('each legal candidate requires exactly one public action kind')
    # Masked candidates and padding contribute nothing, including their kind.
    # Counts are public constants, not a differentiable policy parameter.
    kinds = torch.where(action_mask.unsqueeze(-1), kind_features,
                        torch.zeros_like(kind_features)).detach()
    family_counts = kinds.sum(dim=1, keepdim=True)
    candidate_counts = (kinds * family_counts).sum(dim=-1)
    return logits - candidate_counts.clamp_min(1).log()


def collate(frames: list[dict], device="cpu", *, value_only=False) -> dict:
    """Pad only; never drop an entity, map cell or legal action."""
    if not frames:
        raise ValueError("empty batch")
    if type(value_only) is not bool:
        raise ValueError('value_only must be boolean')
    if len({f.get("vocabulary_fingerprint") for f in frames}) != 1:
        raise ValueError("cannot mix content vocabularies in one policy batch")
    from .encoding import ACTOR_V6, ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15, actor_feature_names
    schemas = {f.get("actor_schema", ACTOR_V6) for f in frames}
    if len(schemas) != 1: raise ValueError("cannot mix actor feature versions")
    actor_schema = next(iter(schemas))
    entity_names, action_names = actor_feature_names(actor_schema)
    from .encoding import action_categorical_fields, entity_categorical_fields
    categorical_action_names = action_categorical_fields(actor_schema)
    for frame in frames:
        if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and (frame.get("entity_features") != list(entity_names)
                or frame.get("action_features") != list(action_names)):
            raise ValueError("versioned frame feature layout metadata differs")
        if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and (frame.get('action_categorical_fields') != list(categorical_action_names)
                or any(type(row) is not list or len(row) != len(categorical_action_names)
                       for row in frame.get('action_categorical', []))
                or len(frame.get('action_categorical', [])) != len(frame['candidates'])):
            raise ValueError('v10 action categorical layout metadata differs')
        if actor_schema == ACTOR_V12:
            from .world_input import CONTRACT as WORLD_INPUT_CONTRACT
            if frame.get('world_input_contract') != WORLD_INPUT_CONTRACT:
                raise ValueError('v12 frame lacks public WorldInput contract metadata')
        if actor_schema in (ACTOR_V13, ACTOR_V14, ACTOR_V15):
            from .world_input import CONTRACT as WORLD_INPUT_CONTRACT
            from .war_confirmation13 import CONTRACT as WAR_CONFIRMATION_CONTRACT
            if (frame.get('world_input_contract') != WORLD_INPUT_CONTRACT
                    or frame.get('war_confirmation_contract') != WAR_CONFIRMATION_CONTRACT):
                raise ValueError('v13/v14 frame lacks the public WorldInput/war-confirmation contract')
            if actor_schema in (ACTOR_V14, ACTOR_V15):
                from .resource_quantity14 import CONTRACT as RESOURCE_QUANTITY_CONTRACT
                if frame.get('resource_quantity_contract') != RESOURCE_QUANTITY_CONTRACT:
                    raise ValueError('v14 frame lacks the resource-quantity contract')
                from .encoding import MAP_CHANNELS
                for channel_name in ('resource_count', 'resource_count_known'):
                    plane = frame['map'][MAP_CHANNELS.index(channel_name)]
                    if any(value != 0 for row in plane for value in row):
                        raise ValueError('v14 frames must keep reserved resource-quantity channels zero')
            if actor_schema == ACTOR_V15:
                from .public_purchases import CONTRACT as PURCHASE_CONTRACT
                entity_categories = entity_categorical_fields(actor_schema)
                if (frame.get('public_purchases_contract') != PURCHASE_CONTRACT
                        or frame.get('entity_categorical_fields') != list(entity_categories)
                        or len(frame.get('entity_categorical', [])) != len(frame['entities'])
                        or any(type(row) is not list or len(row) != len(entity_categories)
                               for row in frame.get('entity_categorical', []))):
                    raise ValueError('v15 public purchase observation contract differs')
        elif 'dialogue_candidate_mask' in frame:
            raise ValueError('only v13/v14 frames may carry a modal dialogue candidate mask')
        if (any(len(row) != len(entity_names) for row in frame["entities"])
                or any(len(row) != len(action_names) for row in frame["candidates"])):
            raise ValueError("frame feature dimensions differ from declared actor version")
    first = frames[0]
    channels = len(first["map"])
    height = max(len(f["map"][0]) for f in frames)
    width = max(len(f["map"][0][0]) for f in frames)
    ne = max(1, max(len(f["entities"]) for f in frames))
    na = max(len(f["candidates"]) for f in frames)
    if not value_only and (na == 0 or any(not any(f["action_mask"]) for f in frames)):
        raise ValueError("no legal action: terminal/wait/unsupported frames need explicit handling")
    # Dimensions are declared even when an entire batch has no entity.
    from .encoding import ENTITY_FEATURES, ACTION_FEATURES, GLOBAL_CATEGORICAL_FIELDS, MAP_CATEGORICAL_FIELDS, MAP_CHANNELS
    if channels != len(MAP_CHANNELS):
        raise ValueError("map feature layout differs; implicit migration is unsupported")
    ed = len(entity_names)
    ad = len(action_names)
    ec = max(1, max((len(row) for f in frames for row in f.get("entity_categorical", [])), default=0))
    ac = max(1, max((len(row) for f in frames for row in f.get("action_categorical", [])), default=0))
    b = len(frames)
    out = {"map": torch.zeros(b, channels, height, width),
           "map_mask": torch.zeros(b, 1, height, width, dtype=torch.bool),
           "map_categorical": torch.zeros(b, len(MAP_CATEGORICAL_FIELDS), height, width, dtype=torch.long),
           "entities": torch.zeros(b, ne, ed), "entity_mask": torch.zeros(b, ne, dtype=torch.bool),
           "candidates": torch.zeros(b, na, ad), "action_mask": torch.zeros(b, na, dtype=torch.bool),
           "entity_categorical": torch.zeros(b, ne, ec, dtype=torch.long),
           "action_categorical": torch.zeros(b, na, ac, dtype=torch.long)}
    out["candidate_source_rows"] = torch.full((b, na), -1, dtype=torch.long)
    movement_routes=[]
    for frame in frames:
        routes=frame.get('movement_point_features',[[] for _ in frame['candidates']])
        if type(routes) is not list or len(routes)!=len(frame['candidates']):
            raise ValueError('movement point routes differ from candidate rows')
        if any(type(points) is not list or len(points)>16384 for points in routes):
            raise ValueError('movement path exceeds the complete native plot bound')
        if actor_schema in (ACTOR_V11,ACTOR_V12,ACTOR_V13, ACTOR_V14, ACTOR_V15):
            from .movement_information import CONTRACT as MOVEMENT_INFORMATION_CONTRACT
            if frame.get('movement_information_contract') != MOVEMENT_INFORMATION_CONTRACT:
                raise ValueError('v11 frame lacks public movement-information contract metadata')
        elif any(points for points in routes):
            raise ValueError('legacy actors cannot consume movement preview points')
        movement_routes.append(routes)
    max_movement_points=max(1,max(sum(len(points) for points in routes) for routes in movement_routes))
    if max_movement_points>524288:
        raise ValueError('movement point batch exceeds its public preview bound')
    out['movement_points']=torch.zeros(b,max_movement_points,4,dtype=torch.float32)
    out['movement_point_mask']=torch.zeros(b,max_movement_points,dtype=torch.bool)
    out['movement_candidate_rows']=torch.full((b,max_movement_points),-1,dtype=torch.long)
    globals_ = []
    for i, f in enumerate(frames):
        grid = torch.tensor(f["map"], dtype=torch.float32)
        c, h, w = grid.shape
        if c != channels or len(f["candidates"]) != len(f["action_mask"]):
            raise ValueError("inconsistent feature shape")
        if any(type(x) is not bool for x in f["action_mask"] + f["entity_mask"]):
            raise ValueError("masks must contain Booleans")
        source_rows = f.get("candidate_source_rows")
        if actor_schema in (ACTOR_V10,ACTOR_V11,ACTOR_V12,ACTOR_V13, ACTOR_V14, ACTOR_V15):
            if (type(source_rows) is not list or len(source_rows) != len(f['candidates'])
                    or any(type(row) is not int or row < -1 or row >= len(f['entities']) for row in source_rows)):
                raise ValueError('v10 candidate source rows differ from candidate layout')
            source_index = action_names.index('source_entity_bound')
            for action_row, entity_row in zip(f['candidates'], source_rows):
                if action_row[source_index] not in (0, 1, 0.0, 1.0) or action_row[source_index] != int(entity_row >= 0):
                    raise ValueError('candidate source binding flag differs from source row')
                if entity_row >= 0 and not f['entity_mask'][entity_row]:
                    raise ValueError('candidate source row is not a present entity')
            if source_rows:
                out['candidate_source_rows'][i, :len(source_rows)] = torch.tensor(source_rows, dtype=torch.long)
        elif source_rows is not None and (type(source_rows) is not list or any(row != -1 for row in source_rows)):
            raise ValueError('legacy frames cannot contain candidate source bindings')
        if actor_schema in (ACTOR_V11,ACTOR_V12,ACTOR_V13, ACTOR_V14, ACTOR_V15):
            from .encoding import ACTION_FEATURES_V10
            known_index=action_names.index('movement_preview_known')
            length_index=action_names.index('movement_path_length')
            arrival_index=action_names.index('movement_arrival_turn')
            cursor=0
            for candidate_index,(candidate,points) in enumerate(zip(f['candidates'],movement_routes[i])):
                is_move=candidate[ACTION_FEATURES_V10.index('kind:move')]==1.0
                if (type(points) is not list or (is_move != bool(points))
                        or candidate[known_index] != float(is_move)
                        or candidate[length_index] != len(points)
                        or candidate[arrival_index] != (points[-1][2] if points else 0.0)):
                    raise ValueError('v11 movement candidate scalar/route binding differs')
                for point_index,point in enumerate(points):
                    if (type(point) is not list or len(point)!=4
                            or any(type(x) not in (int,float) for x in point)
                            or any(not torch.isfinite(torch.tensor(float(x))) for x in point[:2])
                            or not 0.0<=point[0]<=1.0 or not 0.0<=point[1]<=1.0
                            or type(point[2]) is not int or not 1<=point[2]<=0xFFFFFFFF
                            or type(point[3]) is not int or point[3]!=point_index):
                        raise ValueError('v11 movement point is malformed')
                    out['movement_points'][i,cursor]=torch.tensor(point,dtype=torch.float32)
                    out['movement_point_mask'][i,cursor]=True
                    out['movement_candidate_rows'][i,cursor]=candidate_index
                    cursor+=1
        out["map"][i, :, :h, :w] = grid
        out["map_mask"][i, :, :h, :w] = True
        fields = f.get("map_categorical_fields", list(MAP_CATEGORICAL_FIELDS))
        if type(fields) is not list or fields != list(MAP_CATEGORICAL_FIELDS):
            raise ValueError("map categorical fields differ")
        categories = f.get("map_categorical")
        if categories is None:
            # Historical synthetic fixtures have no map content labels.
            categories = [[[0 for _ in range(w)] for _ in range(h)] for _ in MAP_CATEGORICAL_FIELDS]
        if (type(categories) is not list or len(categories) != len(MAP_CATEGORICAL_FIELDS)
                or any(type(channel) is not list or len(channel) != h
                       or any(type(row) is not list or len(row) != w
                              or any(type(token) is not int or token < 0 for token in row)
                              for row in channel) for channel in categories)):
            raise ValueError("map categorical shape or token type differs")
        tokens = torch.tensor(categories, dtype=torch.long)
        bound = f.get("vocabulary_size", 1)
        if type(bound) is not int or bound < 1 or (tokens >= bound).any():
            raise ValueError("map categorical token outside frozen vocabulary")
        for field, label in enumerate(("improvement_displayed", "district_displayed")):
            presence = grid[MAP_CHANNELS.index(label)]
            if not torch.equal(tokens[field].ne(0), presence.eq(1)):
                raise ValueError("map category and public display presence differ")
            if (tokens[field].ne(0) & grid[MAP_CHANNELS.index("plot_display_known")].ne(1)).any():
                raise ValueError("map category lacks a known public display")
        out["map_categorical"][i, :, :h, :w] = tokens
        for name, mask in (("entities", "entity_mask"), ("candidates", "action_mask")):
            n = len(f[name])
            if n:
                out[name][i, :n] = torch.tensor(f[name], dtype=torch.float32)
                out[mask][i, :n] = torch.tensor(f[mask], dtype=torch.bool)
        for name in ("entity_categorical", "action_categorical"):
            if f.get(name):
                a = torch.tensor(f[name], dtype=torch.long)
                out[name][i, :a.shape[0], :a.shape[1]] = a
        g = f["global"]
        globals_.append(g["values"] + g["mask"])
    out["global"] = torch.tensor(globals_, dtype=torch.float32)
    from .task_context import TASK_FEATURES
    task_rows = []; task_types = []
    for frame in frames:
        task = frame.get('task', {'values': [0.0]*len(TASK_FEATURES), 'mask': [False]*len(TASK_FEATURES)})
        if (set(task) != {'values','mask'} or len(task['values']) != len(TASK_FEATURES)
                or len(task['mask']) != len(TASK_FEATURES) or any(type(x) is not bool for x in task['mask'])):
            raise ValueError('task feature layout differs')
        if any(type(x) not in (float,int) or (not known and x != 0)
               for x,known in zip(task['values'],task['mask'])):
            raise ValueError('task numeric features require canonical unknown zeros')
        tokens = frame.get('task_unit_types',[])
        if (type(tokens) is not list or len(tokens)>100000
                or any(type(x) is not int or not 0<x<frame.get('vocabulary_size',1) for x in tokens)
                or len(set(tokens))!=len(tokens)):
            raise ValueError('task unit types must be unique public vocabulary tokens')
        cardinality = TASK_FEATURES.index('target_unit_type_count')
        if ((tokens and not task['mask'][cardinality]) or
                (task['mask'][cardinality] and task['values'][cardinality]!=len(tokens))):
            raise ValueError('task unit type set cardinality differs')
        task_types.append(tokens)
        task_rows.append(task['values'] + task['mask'])
    out['task'] = torch.tensor(task_rows, dtype=torch.float32)
    out['task_unit_types'] = torch.zeros(b,max(1,max(map(len,task_types))),dtype=torch.long)
    for i,tokens in enumerate(task_types):
        if tokens: out['task_unit_types'][i,:len(tokens)] = torch.tensor(tokens,dtype=torch.long)
    gc = len(GLOBAL_CATEGORICAL_FIELDS)
    if any(len(f.get("global_categorical", [])) not in (0, gc) for f in frames):
        raise ValueError("global categorical layout differs")
    out["global_categorical"] = torch.zeros(b, gc, dtype=torch.long)
    for i, f in enumerate(frames):
        if f.get("global_categorical"):
            out["global_categorical"][i, :len(f["global_categorical"])] = torch.tensor(f["global_categorical"], dtype=torch.long)
    from .public_text987 import dialogue_bytes
    texts = [dialogue_bytes(f.get('dialogue'), len(f['candidates']),
                            candidate_mask=f.get('dialogue_candidate_mask')) for f in frames]
    nc = max(1, max(len(row) for text in texts for row in text['context']))
    nm = max(1, max(len(text['menu']) for text in texts))
    nt = max(1, max((len(field) for text in texts for row in text['menu'] for field in row), default=0))
    out['dialogue_context'] = torch.zeros(b, 3, nc, dtype=torch.long)
    out['dialogue_menu_text'] = torch.zeros(b, nm, 2, nt, dtype=torch.long)
    out['dialogue_menu_mask'] = torch.zeros(b, nm, dtype=torch.bool)
    out['dialogue_menu_flags'] = torch.zeros(b, nm, 3)
    out['dialogue_present'] = torch.tensor([[text['present']] for text in texts], dtype=torch.float32)
    out['dialogue_candidate_rows'] = torch.full((b, na), -1, dtype=torch.long)
    for i, text in enumerate(texts):
        for j, row in enumerate(text['context']):
            if row: out['dialogue_context'][i, j, :len(row)] = torch.tensor(row, dtype=torch.long)
        for j, row in enumerate(text['menu']):
            out['dialogue_menu_mask'][i, j] = True
            out['dialogue_menu_flags'][i, j] = torch.tensor(text['flags'][j])
            for k, field in enumerate(row):
                if field: out['dialogue_menu_text'][i, j, k, :len(field)] = torch.tensor(field, dtype=torch.long)
        out['dialogue_candidate_rows'][i, :len(text['candidate_rows'])] = torch.tensor(text['candidate_rows'], dtype=torch.long)
    if any(not torch.isfinite(v).all() for v in out.values() if v.is_floating_point()):
        raise ValueError("nonfinite observation")
    out = {k: v.to(device) for k, v in out.items()}
    # Opaque native identifiers are used only for membership/receipts. The
    # neural inputs below are the public vocabulary tokens prepared by encoding.
    out["policy_catalogs"] = [f.get("policy_catalog", {"slots": [], "prior_tokens": [], "option_tokens": []}) for f in frames]
    out["policy_action_indices"] = [f.get("policy_action_index") for f in frames]
    for i, (catalog, index) in enumerate(zip(out["policy_catalogs"], out["policy_action_indices"])):
        slots = catalog["slots"]
        if len(slots) > 64 or len(catalog["prior_tokens"]) != len(slots) or len(catalog["option_tokens"]) != len(slots):
            raise ValueError("policy token layout differs")
        if any(len(tokens) != len(slot["options"]) for slot, tokens in zip(slots, catalog["option_tokens"])):
            raise ValueError("policy option token layout differs")
        if index is not None and (type(index) is not int or not 0 <= index < len(frames[i]["candidates"]) or not slots):
            raise ValueError("policy root index differs")
    return out


def masked_distribution(logits, mask):
    if mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError("legal mask shape/dtype mismatch")
    if not mask.any(dim=-1).all() or not torch.isfinite(logits).all():
        raise ValueError("empty legal menu or nonfinite logits")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class PublicByteEncoder(nn.Module):
    """Order-sensitive local text features, masked after each convolution.

    This is a small trainable baseline, not a pretrained language model. Full
    strings are preserved at the input; finite learned features are a
    compression and do not establish semantic understanding.
    """
    def __init__(self, embedding_size, channels):
        super().__init__()
        self.embedding = nn.Embedding(257, embedding_size, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_size, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, tokens):
        shape = tokens.shape[:-1]
        flat = tokens.reshape(-1, tokens.shape[-1])
        # The qualified Mac MPS backend aborts during two-layer Conv1d
        # backward at length 1. Extra zero tokens are masked at every layer,
        # so this backend accommodation cannot add public information.
        if flat.shape[-1] < 3:
            flat = torch.nn.functional.pad(flat, (0, 3 - flat.shape[-1]))
        mask = flat.ne(0).unsqueeze(1)
        value = self.embedding(flat).transpose(1, 2) * mask
        value = torch.relu(self.conv1(value)) * mask
        value = torch.relu(self.conv2(value)) * mask
        mean = value.sum(-1) / mask.sum(-1).clamp_min(1)
        maximum = value.max(-1).values
        return torch.cat((mean, maximum), -1).reshape(*shape, -1)


class RecurrentPolicy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        if config.candidate_family_prior not in (None, FAMILY_PRIOR):
            raise ValueError('unknown candidate family prior')
        if config.candidate_family_prior is not None:
            from .encoding import ACTOR_V14, ACTOR_V15, actor_feature_names
            if actor_schema_for_config(config) not in (ACTOR_V14, ACTOR_V15):
                raise ValueError('family prior requires the explicit V14/V15 actor')
            names = actor_feature_names(actor_schema_for_config(config))[1]
            kinds = tuple(name for name in names if name.startswith('kind:'))
            if names[:len(kinds)] != kinds:
                raise ValueError('public action kind feature layout differs')
        self.config = config
        h, s, cat = config.hidden_size, config.spatial_size, config.category_size
        self.map1 = nn.Conv2d(config.map_channels + config.map_categories * cat, s, 3, padding=1)
        self.map2 = nn.Conv2d(s, s, 3, padding=1)
        self.categories = nn.Embedding(config.vocabulary_size, cat, padding_idx=0)
        self.global_categories = nn.Linear(config.global_categories * cat, cat)
        self.entity = nn.Sequential(nn.Linear(config.entity_features + cat, h), nn.Tanh())
        self.global_encoder = nn.Sequential(nn.Linear(2 * config.global_features + cat, h), nn.Tanh())
        self.policy_ordinals = nn.Embedding(64, cat)
        self.policy_slot = nn.Sequential(nn.Linear(3 * cat + 2, h), nn.Tanh())
        self.task_encoder = nn.Sequential(nn.Linear(2 * config.task_features + cat, h), nn.Tanh())
        self.public_text = PublicByteEncoder(config.text_embedding_size, config.text_channels)
        self.dialogue_row = nn.Sequential(nn.Linear(4 * config.text_channels + 3, h), nn.Tanh())
        self.dialogue_context = nn.Sequential(nn.Linear(6 * config.text_channels + h + 1, h), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(s + 5 * h, h), nn.Tanh())
        self.memory = nn.LSTMCell(h, h)
        self.candidate = nn.Sequential(nn.Linear(config.action_features + cat, h), nn.Tanh())
        self.entity_attention = nn.MultiheadAttention(h, num_heads=4, batch_first=True)
        self.query = nn.Linear(h, h, bias=False)
        self.score = nn.Linear(h, 1)
        self.value = nn.Linear(h, 1)
        self.policy_option = nn.Sequential(nn.Linear(cat, h), nn.Tanh())
        self.policy_query = nn.Linear(2 * h + 2 * cat, h)
        self.policy_score = nn.Linear(h, 1)
        self.policy_prefix = nn.GRUCell(h, h)

    def initial_state(self, batch_size, *, device=None):
        p = next(self.parameters())
        return tuple(torch.zeros(batch_size, self.config.hidden_size, device=device or p.device) for _ in range(2))

    def _categories(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        return (self.categories(tokens) * mask).sum(-2) / mask.sum(-2).clamp_min(1)

    @staticmethod
    def _scale(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _policy_tokens(self, catalog, slot_index):
        device = self.categories.weight.device
        prior = self.categories(torch.tensor(catalog["prior_tokens"][slot_index], device=device))
        ordinal = self.policy_ordinals(torch.tensor(slot_index, device=device))
        tokens = catalog["option_tokens"][slot_index]
        options = (self._categories(torch.tensor(tokens, dtype=torch.long, device=device)) if tokens else
                   self.categories.weight.new_zeros((0, self.config.category_size)))
        return prior, ordinal, options

    def _policy_context(self, batch):
        contexts = []
        for catalog in batch["policy_catalogs"]:
            rows = []
            for i, slot in enumerate(catalog["slots"]):
                prior, ordinal, options = self._policy_tokens(catalog, i)
                pooled = options.mean(0) if len(options) else torch.zeros_like(prior)
                flags = prior.new_tensor([float(slot["prior_index"] >= 0), float(len(options))])
                rows.append(self.policy_slot(torch.cat((prior, ordinal, pooled, self._scale(flags)))))
            contexts.append(torch.stack(rows).mean(0) if rows else
                            self.categories.weight.new_zeros(self.config.hidden_size))
        return torch.stack(contexts)

    def forward(self, batch, state=None, reset=None, *, value_only=False):
        if type(value_only) is not bool:
            raise ValueError('value_only must be boolean')
        if set(batch) != ACTOR_BATCH_KEYS:
            raise ValueError('actor batch fields differ; privileged inputs are forbidden')
        if not value_only and not batch['action_mask'].any(dim=-1).all():
            raise ValueError('no legal action for policy forward')
        if (batch['entities'].shape[-1] != self.config.entity_features
                or batch['candidates'].shape[-1] != self.config.action_features):
            raise ValueError('batch dimensions differ from actor; explicit version migration required')
        from .encoding import ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15, actor_feature_names
        schema = actor_schema_for_config(self.config)
        source_rows = batch['candidate_source_rows']
        if (source_rows.dtype != torch.long or source_rows.shape != batch['candidates'].shape[:2]
                or (source_rows < -1).any()):
            raise ValueError('candidate source row tensor differs from actor contract')
        if schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
            entity_names, action_names = actor_feature_names(schema)
            max_entities = batch['entities'].shape[1]
            if (source_rows >= max_entities).any():
                raise ValueError('candidate source row outside entity tensor')
            bound = batch['candidates'][..., action_names.index('source_entity_bound')]
            if not torch.equal(bound, source_rows.ge(0).to(dtype=bound.dtype)):
                raise ValueError('candidate source binding flag differs from source row')
            present = batch['entity_mask'].gather(1, source_rows.clamp_min(0))
            if (source_rows.ge(0) & ~present).any():
                raise ValueError('candidate source row is not a present entity')
        elif source_rows.ne(-1).any():
            raise ValueError('legacy actor cannot consume candidate source bindings')
        # Mask every convolution, so a padded larger map cannot alter a small
        # map's boundary through biases in the artificial cells.
        valid = batch["map_mask"]
        tokens = batch["map_categorical"]
        if (tokens.dtype != torch.long or tokens.ndim != 4
                or tokens.shape != (batch["map"].shape[0], self.config.map_categories,
                                    batch["map"].shape[-2], batch["map"].shape[-1])
                or (tokens < 0).any() or (tokens >= self.config.vocabulary_size).any()):
            raise ValueError("map categorical tensor differs from actor contract")
        embedded = self.categories(tokens) * tokens.ne(0).unsqueeze(-1)
        embedded = embedded.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()
        spatial_input = torch.cat((self._scale(batch["map"]), embedded), 1) * valid
        spatial = torch.relu(self.map1(spatial_input)) * valid
        spatial = torch.relu(self.map2(spatial)) * valid
        spatial = spatial.sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)
        ent = self.entity(torch.cat((self._scale(batch["entities"]), self._categories(batch["entity_categorical"])), -1))
        em = batch["entity_mask"].unsqueeze(-1)
        entity_tokens = ent
        ent = (ent * em).sum(1) / em.sum(1).clamp_min(1)
        # Research and civics use the same content-index namespace. Keep their
        # slots ordered so exchanging their indices cannot look identical.
        global_cat = self.global_categories(self.categories(batch["global_categorical"]).flatten(1))
        glob = self.global_encoder(torch.cat((self._scale(batch["global"]), global_cat), -1))
        task_types = batch['task_unit_types']
        if (task_types.dtype != torch.long or task_types.ndim != 2
                or task_types.shape[0] != batch['task'].shape[0]
                or (task_types < 0).any() or (task_types >= self.config.vocabulary_size).any()):
            raise ValueError('task unit type tensor differs from actor contract')
        task = self.task_encoder(torch.cat((self._scale(batch['task']),self._categories(task_types)),-1))
        text_context = self.public_text(batch['dialogue_context']).flatten(1)
        text_menu = self.public_text(batch['dialogue_menu_text']).flatten(2)
        menu_mask = batch['dialogue_menu_mask'].unsqueeze(-1)
        menu = self.dialogue_row(torch.cat((text_menu, batch['dialogue_menu_flags']), -1)) * menu_mask
        menu_pool = menu.sum(1) / menu_mask.sum(1).clamp_min(1)
        dialogue = self.dialogue_context(torch.cat((text_context, menu_pool, batch['dialogue_present']), -1)) * batch['dialogue_present']
        fused = self.fusion(torch.cat((spatial, ent, glob, self._policy_context(batch), task, dialogue), -1))
        if state is None:
            state = self.initial_state(fused.shape[0])
        if reset is not None:
            if reset.dtype != torch.bool or reset.shape != (fused.shape[0],):
                raise ValueError("reset must be a Boolean per batch member")
            state = tuple(x * (~reset).unsqueeze(-1) for x in state)
        hidden, cell = self.memory(fused, state)
        if value_only:
            # A native collection cap can close its action menu while still
            # providing a genuine public successor. Estimate its value without
            # fabricating an available action or constructing an empty policy.
            return None, self.value(hidden).squeeze(-1), (hidden, cell)
        action = self.candidate(torch.cat((self._scale(batch["candidates"]), self._categories(batch["action_categorical"])), -1))
        if schema in (ACTOR_V11,ACTOR_V12,ACTOR_V13, ACTOR_V14, ACTOR_V15):
            points=batch['movement_points'];point_mask=batch['movement_point_mask'];point_rows=batch['movement_candidate_rows']
            if (points.dtype!=torch.float32 or points.ndim!=3 or points.shape[0]!=action.shape[0]
                    or points.shape[2]!=4 or point_mask.dtype!=torch.bool
                    or point_mask.shape!=points.shape[:2] or point_rows.dtype!=torch.long
                    or point_rows.shape!=points.shape[:2] or not torch.isfinite(points).all()):
                raise ValueError('v11 movement point tensor layout differs')
            candidate_count=action.shape[1]
            if ((point_rows[point_mask]<0).any() or (point_rows[point_mask]>=candidate_count).any()
                    or point_rows[~point_mask].ne(-1).any() or points[~point_mask].ne(0).any()):
                raise ValueError('v11 movement point mask or candidate binding differs')
            from .movement_information import POINT_ENTITY_FEATURES
            point_columns=[entity_names.index(name) for name in POINT_ENTITY_FEATURES]
            entity_linear=self.entity[0]
            point_hidden=torch.tanh(torch.nn.functional.linear(self._scale(points),
                entity_linear.weight[:,point_columns],entity_linear.bias))
            route_context=action.new_zeros(action.shape)
            route_counts=action.new_zeros(action.shape[:2])
            if point_mask.any():
                batch_rows=torch.arange(points.shape[0],device=points.device)[:,None].expand_as(point_rows)[point_mask]
                candidate_rows=point_rows[point_mask]
                flattened=batch_rows*candidate_count+candidate_rows
                pooled=route_context.reshape(-1,route_context.shape[-1]).index_add(
                    0,flattened,point_hidden[point_mask])
                counts=route_counts.reshape(-1).index_add(
                    0,flattened,torch.ones_like(flattened,dtype=route_counts.dtype))
                route_context=pooled.view_as(route_context)
                route_counts=counts.view_as(route_counts)
            action=action+route_context/route_counts.clamp_min(1).unsqueeze(-1)
        else:
            if (batch['movement_point_mask'].any() or batch['movement_candidate_rows'].ne(-1).any()
                    or batch['movement_points'].ne(0).any()):
                raise ValueError('legacy actor cannot consume movement preview points')
        if schema in (ACTOR_V10,ACTOR_V11,ACTOR_V12,ACTOR_V13, ACTOR_V14, ACTOR_V15):
            bound_rows = source_rows.clamp_min(0)
            source_token = entity_tokens.gather(1, bound_rows.unsqueeze(-1).expand(-1, -1, entity_tokens.shape[-1]))
            action = action + source_token * source_rows.ge(0).unsqueeze(-1)
        row_indices = batch['dialogue_candidate_rows']
        selected_menu = menu.gather(1, row_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, menu.shape[-1]))
        action = action + selected_menu * row_indices.ge(0).unsqueeze(-1)
        # A permanent null token avoids all-masked attention for empty rosters.
        # Candidate queries can inspect city/unit details instead of receiving
        # only the pooled empire summary.
        tokens = torch.cat((entity_tokens, torch.zeros_like(entity_tokens[:, :1])), 1)
        padding = torch.cat((~batch["entity_mask"], torch.zeros_like(batch["entity_mask"][:, :1])), 1)
        context, _ = self.entity_attention(action, tokens, tokens, key_padding_mask=padding, need_weights=False)
        logits = self.score(torch.tanh(action + context + self.query(hidden).unsqueeze(1))).squeeze(-1)
        if self.config.candidate_family_prior is not None:
            kind_count = sum(name.startswith('kind:') for name in action_names)
            logits = family_prior_logits(logits,
                batch['candidates'][..., :kind_count], batch['action_mask'])
        return masked_distribution(logits, batch["action_mask"]), self.value(hidden).squeeze(-1), (hidden, cell)

    def _decode_policies(self, hidden, catalog, assignments=None):
        """One native action, factored into exact conditional distributions.

        The entropy is a fresh-sample estimate over prefixes. The score-function
        term accounts for how changing earlier choices changes later entropy;
        its zero-valued correction changes only the gradient, not the estimate.
        """
        from .policy_assignments import valid_next_hashes, validate_joint_assignment
        slots = catalog["slots"]
        if assignments is not None:
            validate_joint_assignment(slots, assignments)
        prefix, recurrent = [], hidden
        log_prob = hidden.new_zeros(())
        entropy = hidden.new_zeros(())
        for i, slot in enumerate(slots):
            allowed = set(valid_next_hashes(slots, prefix))
            prior, ordinal, option_tokens = self._policy_tokens(catalog, i)
            options = self.policy_option(option_tokens)
            query = self.policy_query(torch.cat((hidden, recurrent, prior, ordinal)))
            logits = self.policy_score(torch.tanh(options + query)).squeeze(-1)
            mask = torch.tensor([o["hash"] in allowed for o in slot["options"]], dtype=torch.bool, device=hidden.device)
            dist = masked_distribution(logits, mask)
            if assignments is None:
                chosen = dist.sample()
            else:
                chosen = torch.tensor(next(j for j, o in enumerate(slot["options"]) if o["hash"] == assignments[i]), device=hidden.device)
            conditional_entropy = dist.entropy()
            correction = conditional_entropy.detach() * log_prob
            entropy = entropy + conditional_entropy + (correction - correction.detach())
            log_prob = log_prob + dist.log_prob(chosen)
            prefix.append(slot["options"][int(chosen.item())]["hash"])
            recurrent = self.policy_prefix(options[chosen].unsqueeze(0), recurrent.unsqueeze(0)).squeeze(0)
        validate_joint_assignment(slots, prefix)
        return prefix, log_prob, entropy

    def _joint_entropy(self, root, hidden, batch):
        estimates, samples = [], []
        for i, index in enumerate(batch["policy_action_indices"]):
            if index is None or not bool(batch["action_mask"][i, index]):
                estimates.append(hidden.new_zeros(())); samples.append(None)
                continue
            sample = self._decode_policies(hidden[i], batch["policy_catalogs"][i])
            samples.append(sample)
            estimates.append(root.probs[i, index] * sample[2])
        return root.entropy() + torch.stack(estimates), samples

    def act(self, batch, state=None, reset=None):
        root, value, newstate = self(batch, state, reset)
        actions = root.sample()
        entropy, policy_samples = self._joint_entropy(root, newstate[0], batch)
        logps, assignments = [], []
        root_logps = root.log_prob(actions)
        for i, action in enumerate(actions.tolist()):
            sample = policy_samples[i] if action == batch["policy_action_indices"][i] else None
            assignments.append(None if sample is None else sample[0])
            logps.append(root_logps[i] if sample is None else root_logps[i] + sample[1])
        return actions, torch.stack(logps), value, newstate, entropy, assignments

    def evaluate_actions(self, batch, actions, assignments=None, state=None, reset=None):
        """Teacher-force the complete recorded action; advance game LSTM once."""
        size = batch["global"].shape[0]
        if actions.shape != (size,) or actions.dtype != torch.long:
            raise ValueError("recorded root action layout differs")
        if (actions < 0).any() or (actions >= batch["action_mask"].shape[1]).any() or not batch["action_mask"].gather(1, actions[:, None]).all():
            raise ValueError("recorded root action is outside the legal menu")
        assignments = [None] * size if assignments is None else assignments
        if len(assignments) != size:
            raise ValueError("recorded policy assignments layout differs")
        # Validate the whole batch before creating scores (or sampling entropy).
        from .policy_assignments import validate_joint_assignment
        for i, action in enumerate(actions.tolist()):
            if action == batch["policy_action_indices"][i]:
                if assignments[i] is None:
                    raise ValueError("policy action requires its joint assignments")
                validate_joint_assignment(batch["policy_catalogs"][i]["slots"], assignments[i])
            elif assignments[i] is not None:
                raise ValueError("ordinary action cannot have policy assignments")
        root, value, newstate = self(batch, state, reset)
        root_logps = root.log_prob(actions)
        logps = []
        for i, assignment in enumerate(assignments):
            extra = (self._decode_policies(newstate[0][i], batch["policy_catalogs"][i], assignment)[1]
                     if assignment is not None else root_logps.new_zeros(()))
            logps.append(root_logps[i] + extra)
        entropy, _ = self._joint_entropy(root, newstate[0], batch)
        return torch.stack(logps), value, newstate, entropy


class PlayerMemory:
    """One state per episode/player, with duplicate/out-of-order protection."""
    def __init__(self):
        self._states = {}

    def take(self, episode, player, sequence):
        value = self._states.get((episode, player))
        if value is None:
            return None
        previous, state = value
        if sequence <= previous:
            raise ValueError("replayed or stale decision sequence")
        return state

    def put(self, episode, player, sequence, state):
        self.take(episode, player, sequence)
        self._states[(episode, player)] = (sequence, tuple(x.detach() for x in state))

    def reset_episode(self, episode):
        self._states = {key: value for key, value in self._states.items() if key[0] != episode}


def actor_schema_for_config(config):
    from .encoding import ACTOR_V6, ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15, actor_feature_names
    explicit = getattr(config, 'actor_schema', None)
    if explicit is not None:
        entities, actions = actor_feature_names(explicit)
        if (config.entity_features, config.action_features) != (len(entities), len(actions)):
            raise ValueError('explicit actor schema differs from configured feature dimensions')
        return explicit
    # V13/V14 intentionally share V12 widths; old untagged configs retain
    # their historical V12 interpretation.
    for schema in (ACTOR_V6, ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12):
        entities, actions = actor_feature_names(schema)
        if (config.entity_features, config.action_features) == (len(entities), len(actions)):
            return schema
    raise ValueError('unsupported versioned actor dimensions')


def checkpoint_payload(model, *, vocabulary=None):
    from .encoding import MAP_CHANNELS, GLOBAL_FEATURES, ENTITY_FEATURES, ACTION_FEATURES
    from .encoding import GLOBAL_CATEGORICAL_FIELDS, MAP_CATEGORICAL_FIELDS, action_categorical_fields, entity_categorical_fields
    from .task_context import TASK_FEATURES, TASK_SCHEMA
    from .skill_context993 import TASK_TYPE_ENCODER
    from .public_text987 import TEXT_LAYOUT
    from .encoding import actor_feature_names
    schema = actor_schema_for_config(model.config)
    entity_names, action_names = actor_feature_names(schema)
    serialized_config = asdict(model.config)
    # Preserve the byte-level shape of historical checkpoints. The optional
    # field is emitted only for actors whose schema identity cannot be inferred
    # from dimensions (currently V13, which shares V12 widths).
    if serialized_config.get('actor_schema') is None:
        serialized_config.pop('actor_schema', None)
    if serialized_config.get('candidate_family_prior') is None:
        serialized_config.pop('candidate_family_prior', None)
    result = {"schema": schema, "config": serialized_config, "state_dict": model.state_dict(),
            "feature_layout": {"map": list(MAP_CHANNELS), "global": list(GLOBAL_FEATURES),
                               "map_categories": list(MAP_CATEGORICAL_FIELDS),
                               "map_category_encoder": "shared_content_embedding_concat_v1",
                               "entities": list(entity_names), "actions": list(action_names),
                               "global_categories": list(GLOBAL_CATEGORICAL_FIELDS),
                               "entity_categories": list(entity_categorical_fields(schema)),
                               "action_categories": list(action_categorical_fields(schema)),
                               "policy_categories": ["policy.index", "policy.hash", "policy.type"],
                               "policy_decoder": "autoregressive_exact_completion_mask_v1",
                               "task": list(TASK_FEATURES), "task_schema": TASK_SCHEMA,
                               "task_unit_type_encoder": TASK_TYPE_ENCODER,
                               "public_text": TEXT_LAYOUT},
            "vocabulary": vocabulary.serialize() if vocabulary is not None else None,
            "vocabulary_sha256": vocabulary.fingerprint if vocabulary is not None else None,
            "training_eligible": False}
    if model.config.candidate_family_prior is not None:
        result['feature_layout']['candidate_family_prior'] = dict(FAMILY_PRIOR_CONTRACT)
        # Canonicalize the repeated string alias for torch/pickle: loading a
        # checkpoint creates a new config string, so the constant alone would
        # serialize differently despite identical tensor/contract values.
        result['feature_layout']['candidate_family_prior']['id'] = serialized_config['candidate_family_prior']
    if schema in ("civ6rl.recurrent_policy.v7", "civ6rl.recurrent_policy.v8", "civ6rl.recurrent_policy.v9", "civ6rl.recurrent_policy.v10", "civ6rl.recurrent_policy.v11", "civ6rl.recurrent_policy.v12", "civ6rl.recurrent_policy.v13", "civ6rl.recurrent_policy.v14", "civ6rl.recurrent_policy.v15"):
        result['feature_layout']['city_focus'] = 'single_ui_cycle_native_public_flags1000_v1'
    if schema in ('civ6rl.recurrent_policy.v8', 'civ6rl.recurrent_policy.v9', 'civ6rl.recurrent_policy.v10', 'civ6rl.recurrent_policy.v11', 'civ6rl.recurrent_policy.v12', 'civ6rl.recurrent_policy.v13', 'civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15'):
        from .city_management import FEATURE_CONTRACT
        result['feature_layout']['city_management'] = FEATURE_CONTRACT
    if schema in ('civ6rl.recurrent_policy.v9', 'civ6rl.recurrent_policy.v10', 'civ6rl.recurrent_policy.v11', 'civ6rl.recurrent_policy.v12', 'civ6rl.recurrent_policy.v13', 'civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15'):
        from .production_queue import FEATURE_CONTRACT, SOURCE_SHA256, CATALOG_SCHEMA
        result['feature_layout']['production_queue'] = dict(contract=FEATURE_CONTRACT,
            source_sha256=SOURCE_SHA256, catalog_schema=CATALOG_SCHEMA)
    if schema in ('civ6rl.recurrent_policy.v10','civ6rl.recurrent_policy.v11','civ6rl.recurrent_policy.v12','civ6rl.recurrent_policy.v13','civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15'):
        from .action_bindings import SCHEMA as ACTION_SOURCE_BINDINGS_SCHEMA
        from .action_content import SCHEMA as ACTION_CONTENT_SCHEMA
        result['feature_layout']['action_source_bindings'] = ACTION_SOURCE_BINDINGS_SCHEMA
        result['feature_layout']['action_content'] = ACTION_CONTENT_SCHEMA
    if schema in ('civ6rl.recurrent_policy.v11','civ6rl.recurrent_policy.v12','civ6rl.recurrent_policy.v13','civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15'):
        from .movement_information import CONTRACT as MOVEMENT_INFORMATION_CONTRACT
        result['feature_layout']['movement_information'] = MOVEMENT_INFORMATION_CONTRACT
    if schema in ('civ6rl.recurrent_policy.v12','civ6rl.recurrent_policy.v13','civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15'):
        from .world_input import CONTRACT as WORLD_INPUT_CONTRACT
        result['feature_layout']['world_input'] = WORLD_INPUT_CONTRACT
    if schema == 'civ6rl.recurrent_policy.v13':
        from .war_confirmation13 import CONTRACT as WAR_CONFIRMATION_CONTRACT
        result['feature_layout']['war_confirmation'] = WAR_CONFIRMATION_CONTRACT
    if schema in ('civ6rl.recurrent_policy.v14', 'civ6rl.recurrent_policy.v15'):
        from .war_confirmation13 import CONTRACT as WAR_CONFIRMATION_CONTRACT
        from .resource_quantity14 import CONTRACT as RESOURCE_QUANTITY_CONTRACT
        result['feature_layout']['war_confirmation'] = WAR_CONFIRMATION_CONTRACT
        result['feature_layout']['resource_quantity'] = RESOURCE_QUANTITY_CONTRACT
    if schema == 'civ6rl.recurrent_policy.v15':
        from .public_purchases import CONTRACT as PURCHASE_CONTRACT
        result['feature_layout']['public_purchases'] = PURCHASE_CONTRACT
    return result
