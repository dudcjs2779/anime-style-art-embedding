from typing import Optional
import torch
import torch.nn as nn

from torch.nn import functional as F

def growth_function(x, midpoint=0.5, steepness=10, max_y=0.5):
    """
    Logistic growth-like function.

    Parameters:
        x (float or np.ndarray): 입력 값 (0 ~ 1 권장)
        midpoint (float): y=0.5가 되는 x값 (기본 0.5)
        steepness (float): 곡선의 가파른 정도 (값이 클수록 급격히 변함)
        max_y (float): x=1.0 일때의 y값
    Returns:
        float or np.ndarray: y값 (0 ~ 1)
    """
    return  2 * max_y / (1 + torch.exp(-steepness * (x - midpoint))) 
    # return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv

def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left

class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)

def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)

class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)
    
def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)

class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}


    def get_ground_truth(self, device, dtype, num_logits, classes, same_class_weight=1.0, 
                         times=None, time_range=10, time_weight=0.3, 
                         negative_only=False) -> torch.Tensor:
        
        if same_class_weight != 1.0:
            labels = (classes[:, None] == classes[None, :]).float()
            if times is not None:
                time_diff = torch.cdist(times[:, None], times[:, None], p=1.0)
                time_weights = growth_function(torch.clamp(time_diff, max=time_range) / time_range,
                                            midpoint=1.0, steepness=6, max_y=time_weight)
                
                labels = labels * (same_class_weight - time_weights)
                labels.fill_diagonal_(1.0)
            else:
                eye = torch.eye(num_logits, dtype=torch.bool, device=classes.device)
                labels[labels.bool() & ~eye] = same_class_weight

        else:
            labels = (classes[:, None] == classes[None, :]).float()

        labels = labels.to(device=device, dtype=dtype)

        # labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        # if not negative_only:
        #     labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features1, image_features2, logit_scale, logit_bias=None):
        logits = logit_scale * image_features1 @ image_features2.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features1, image_features2, classes, logit_scale, logit_bias=None, same_class_weight=1.0, 
              times=None, time_range=10, time_weight=0.3, negative_only=False):
        logits = self.get_logits(image_features1, image_features2, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features1.device,
            image_features1.dtype,
            image_features1.shape[0],
            classes,
            same_class_weight,
            times=times, time_range=time_range, time_weight=time_weight,
            negative_only=negative_only,
        )
        # loss = -F.logsigmoid(labels * logits).sum() / image_features1.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        return loss

    def forward(self, image_features1, image_features2, logit_scale, logit_bias, classes, same_class_weight=1.0, 
                times=None, time_range=10, time_weight=0.3,
                output_dict=False):
        loss = self._loss(image_features1, image_features2, classes, logit_scale, logit_bias, same_class_weight, times, time_range, time_weight)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                image_features2_to_right = image_features2_to_left = image_features2
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    image_features2_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        image_features2_to_left,
                        image_features2_to_right,
                    )
                    for f in image_features2_recv:
                        loss += self._loss(
                            image_features1,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    image_features2_to_left, image_features2_to_right = image_features2_recv

                if remainder:
                    image_features2_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        image_features2_to_right
                    )
                    loss += self._loss(
                        image_features1,
                        image_features2_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                image_features2_to_right = image_features2
                for i in range(self.world_size - 1):
                    image_features2_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        image_features2_to_right,
                    )
                    loss += self._loss(
                        image_features1,
                        image_features2_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    image_features2_to_right = image_features2_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    image2_from_other = torch.distributed.nn.all_reduce(
                        image_features2 * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features1,
                        image2_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_images2 = torch.distributed.nn.all_gather(image_features2)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features1,
                        all_images2[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
    

class CosineSimilarityLoss(nn.Module):
    '''
    vector-wise하게 pred와 label간의  Cosine Similarity를 계산해 
    두 vector의 Cosine Similarity가 최대화 되도록 loss를 계산
    loss가 0에 가까울 수록 pred와 label간의 Cosine Similarity가 높음
    범위: 0~2
    '''
    def __init__(self, dim=-1):
        super().__init__()
        self.loss_fn = nn.CosineSimilarity(dim=dim)

    def forward(self, pred, label, mask_weight=None):
        loss = 1 - self.loss_fn(pred, label)
        if mask_weight is not None:
            loss = loss * mask_weight
        return loss.mean()
    

def mae_loss(pred, label, mask, norm_pix_loss=False):
    if norm_pix_loss:
        mean = label.mean(dim=-1, keepdim=True)
        var = label.var(dim=-1, keepdim=True)
        label = (label - mean) / (var + 1.e-6)**.5

    loss = ((pred - label) ** 2).mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    return loss



