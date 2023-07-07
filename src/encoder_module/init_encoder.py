from encoder_module.multi_modal_encoder import CoAttentionEncoder,CrossModalityEncoder,GuidedAttentionEncoder

def build_encoder(config):
    if config['encoder']['type']=='cross':
        return CrossModalityEncoder(config)
    if config['encoder']['type']=='co':
        return CoAttentionEncoder(config)
    if config['encoder']['type']=='guide':
        return GuidedAttentionEncoder(config)