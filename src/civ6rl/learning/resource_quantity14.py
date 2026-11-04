"""V14 contract for unqualified per-plot resource quantities."""

CONTRACT = {
    'schema': 'civ6rl.resource_quantity_actor.v1',
    'resource_count_channel': 'always_zero_reserved',
    'resource_count_known_channel': 'always_zero_reserved',
    'native_source_quantity_is_actor_input': False,
    'resource_type_semantics_unchanged': True,
}
