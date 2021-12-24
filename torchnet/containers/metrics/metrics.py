import torch


class Metrics:
    def __init__(self) -> None:
        self.eps = 1e-4

    def setup(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        args = preds.argmax(1, keepdim=True)
        probs = preds.gather(1, args)
        out = torch.concat([args, probs], axis=1)
        self.preds, self.target = out, target

    def accuracy(self, preds: torch.Tensor, target: torch.Tensor, th: float=0.0) -> torch.Tensor:
        self.setup(preds, target)
        self.matrix = self._compute_matrix(th)
        acc = self.matrix.diag().sum().item() / len(self.target)
        return acc

    def recall(self, preds: torch.Tensor, target: torch.Tensor, th: float=0.0) -> torch.Tensor:
        self.setup(preds, target)
        self.matrix = self._compute_matrix(th)
        recall = 0
        for i in range(len(self.matrix)):
            recall += self.matrix[i, i] / (self.matrix[i, :].sum() + self.eps)
        return recall / len(self.matrix)

    def precision(self, preds: torch.Tensor, target: torch.Tensor, th: float=0.0) -> torch.Tensor:
        self.setup(preds, target)
        self.matrix = self._compute_matrix(th)
        precision = 0
        for i in range(len(self.matrix)):
            precision += self.matrix[i, i] / (self.matrix[:, i].sum() + self.eps)
        return precision / len(self.matrix)

    def f1_score(self, preds: torch.Tensor, target: torch.Tensor, th: float=0.0) -> torch.Tensor:
        self.setup(preds, target)
        recall = self.recall(preds, target)
        precision = self.precision(preds, target)
        return 2 * precision * recall / (precision + recall)

    def _compute_matrix(self, th: float) -> torch.Tensor:
        preds = self.preds
        dim = preds.ndim
        matrix = torch.zeros(dim, dim)
        mask = preds[:, 1] >= th

        rows = self.target[mask].argmax(dim=1)
        cols = preds[mask][:, 0].type(torch.IntTensor)

        for row, col in zip(rows, cols):
            matrix[row, col] += 1

        return matrix
