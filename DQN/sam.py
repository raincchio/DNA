import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer

    def step(self, closure):
        assert closure is not None, "SAM optimizer requires a closure that reevaluates the model and returns the loss"

        loss = closure()
        loss.backward()
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]['rho'] / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.data.add_(e_w)

        loss = closure()
        loss.backward()

        self.base_optimizer.step()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.data.sub_(e_w)

        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.detach().norm(p=2).to(device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ])
        )
        return norm