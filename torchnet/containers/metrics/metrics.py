import torch


class Metrics:
    def setup(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        args = preds.argmax(1, keepdim=True)
        probs = preds.gather(1, args)
        out = torch.concat([args, probs], axis=1)
        self.preds, self.target = out, target

    def accuracy(self, preds: torch.Tensor, target: torch.Tensor, th: float=0.51):
        self.setup(preds, target)
        self._compute(th)

    def _compute(self, th: float):
        preds = self.preds
        dim = preds.ndim
        self.matrix = torch.zeros(dim, dim)
        preds[:, 1][preds[:, 1] <= th] = 0.0
        print(preds)
