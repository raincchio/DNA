import torch.optim as optim
from adams import Adams_ZeRO

# from xAdam import xAdam

def get_optimizer(cfg, q_network):
    assert sum([cfg.enable_muon, cfg.enable_adams, cfg.enable_adam]) == 1
    if cfg.enable_muon:
        if cfg.wob:
            from muon import SingleDeviceMuon
            optimizer = SingleDeviceMuon(q_network.parameters(), lr=cfg.learning_rate)
        else:
            from muon import SingleDeviceMuonWithAuxAdam
            paramters = q_network.named_modules()
            muon_name =[]
            muon_weight = []
            adam_param = []
            for name, param in paramters:
                if not name:continue
                adam_param.append(param.bias)
                if name in muon_name:
                    muon_weight.append(param.weight)
                else:
                    adam_param.append(param.weight)
            param_groups = [
                dict(params=muon_weight, use_muon=True,
                     lr=cfg.muon_lr, weight_decay=0),
                dict(params=adam_param, use_muon=False,
                     lr=cfg.learning_rate, betas=(0.9, 0.999),eps=cfg.adam_eps, weight_decay=0),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    if cfg.enable_adams:
        optimizer = Adams_ZeRO(q_network.parameters(), lr=cfg.learning_rate,  weight_decay=cfg.adams_weight_decay, scalar_vector_weight_decay=0.1, betas=(0.9, 0.999))
    if cfg.enable_adam:
        optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate,eps=cfg.adam_eps)
    return optimizer