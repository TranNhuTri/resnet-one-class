from torch import randn, norm, div, matmul, transpose, FloatTensor, unsqueeze
from torch.autograd import Variable
from torch.nn import Module, Parameter


class AMSoftmax(Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = Parameter(randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = norm(feat, p=2, dim=-1, keepdim=True)
        num_feat = div(feat, norms)

        norms_c = norm(self.centers, p=2, dim=-1, keepdim=True)
        num_centers = div(self.centers, norms_c)
        logits = matmul(num_feat, transpose(num_centers, 0, 1))

        y_onehot = FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits