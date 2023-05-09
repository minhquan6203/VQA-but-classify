from model_apply_multi_head_att import MultimodalVQAModel

def countTrainableParameters(model: MultimodalVQAModel) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
