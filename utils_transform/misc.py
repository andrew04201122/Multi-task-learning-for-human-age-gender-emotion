import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_oracle(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        pred = output.argmax(-1)
        batch_size = target.size(0)

        correct = pred.eq(target.view(-1, 1).expand_as(pred)).sum(-1)

        correct = correct.view(-1).clamp(0, 1).float().sum(0, keepdim=True)
        correct = correct.mul_(100.0 / batch_size)
        return correct

def get_activation(name, dictionary):
    def hook(model, input, output):
        dictionary[name] = output.detach()
    return hook

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

# https://github.com/richardaecn/class-balanced-loss/blob/master/src/cifar_main.py#L425-L430
def class_balanced_weights(beta, num_per_cls, num_cls=8):
    effective_num = 1.0 - beta**num_per_cls
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * num_cls
    return weights