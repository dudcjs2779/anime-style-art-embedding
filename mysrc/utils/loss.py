from typing import Optional
import torch
import torch.nn as nn

from torch.nn import functional as F
from mysrc.utils import function

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

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


class SimilarityStatisticsEMA:
    def __init__(self, momentum, num_classes, device):
        self.momentum = momentum
        self.num_classes = num_classes
        self.mean = torch.zeros(
            num_classes,
            device=device,
        )

    @torch.no_grad()
    def update(self, similaritys, classes):
        sums = torch.zeros(self.num_classes, device=similaritys.device, dtype=similaritys.dtype)
        # sum_sq = torch.zeros_like(sums)
        counts = torch.zeros_like(sums)

        sums.index_add_(0, classes, similaritys)
        # sum_sq.index_add_(0, classes, similaritys * similaritys)
        counts.index_add_(0, classes, torch.ones_like(classes, dtype=similaritys.dtype))

        mask = counts > 0
        new_mean = sums[mask] / counts[mask]
        # new_secnond = sum_sq[mask] / counts[mask]

        # momentum update
        self.mean[mask] = (
            self.momentum * self.mean[mask] + (1 - self.momentum) * new_mean
        )

class RelationSigLipLoss(nn.Module):
    def __init__(
            self, rel_matrix=None, rel_real=None,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
            device=None,
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
        self.rel_matrix = rel_matrix.to(device=device, dtype=torch.float32) if rel_matrix is not None else None
        self.rel_real = rel_real.to(device=device, dtype=torch.float32) if rel_real is not None else None
        self.gt_mask = self.rel_real != 0 if rel_real is not None else None

    def get_ground_truth(
        self, device, dtype, num_logits, classes,
    ) -> torch.Tensor:
        if self.rel_matrix is not None and classes is not None:
            labels = self.rel_matrix[classes][:, classes]

        elif classes is not None:
            labels = classes[:, None] == classes[None, :]

        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)
        
        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_logits(self, image_features1, image_features2, classes=None):
        logits = image_features1 @ image_features2.T

        return logits
    
    def not_fars_masking(self, labels, logits, classes, not_fars, low, high):
        weight_mask = torch.ones_like(logits, dtype=logits.dtype, device=logits.device)

        if not_fars is not None and not_fars.any():
            same_class = classes[:, None] == classes[None, :]

            pair_not_far = not_fars[:, None] | not_fars[None, :]    # 본인 클래스 외 not_fars와 연관되어 있는 데이터 학습 X
            both_not_far = not_fars[:, None] & not_fars[None, :]    # 둘다 not_fars 데이터인 경우 학습 X

            # 정확히 하나만 not_far + 같은 클래스
            valid = same_class & pair_not_far & ~both_not_far


            # not_far 관련 학습 제외 조건
            invalid = (
                both_not_far
                | (pair_not_far & ~same_class)
                | (valid & (logits >= low) & (logits <= high))
            )

            weight_mask[invalid] = 0

            # label 변경
            labels[valid & (logits < low)] = low
            labels[valid & (logits > high)] = high

        return labels, weight_mask

    def _loss(
            self, image_features1, image_features2, 
            logit_scale, logit_bias=None, 
            classes=None,
            not_fars=None,
        ):
        logits = self.get_logits(image_features1, image_features2, classes)
        labels = self.get_ground_truth(
            image_features1.device,
            image_features1.dtype,
            image_features1.shape[0],
            classes,
        )

        # if not_fars is not None and not_fars.sum() > 1:
        #     breakpoint()

        labels, not_fars_weight_mask = self.not_fars_masking(labels, logits, classes, not_fars, 0.5, 0.7)

        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias

        # loss = -F.logsigmoid(labels * logits).sum() / image_features1.shape[0]
        if classes is not None:
            loss  = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

            # 자기자신과의 semantic한 의미 학습 방지하기
            mask = 1 - torch.eye(loss.size(0), dtype=logits.dtype, device=loss.device)
            mask = mask * not_fars_weight_mask

            loss = (loss * mask).sum() / mask.sum()
            # loss = loss[mask].mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        return loss

    def forward(
            self, image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3,
            not_fars=None,
            output_dict=False
        ):
        loss = self._loss(
            image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes, 
            not_fars,
        )

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
    
    
class RelationPrototypeContrastiveLoss(nn.Module):
    def __init__(
            self, num_classes, momentum, rel_matrix=None, rel_real=None, topk_rate=0.1,
            ema_threshold_on=False,
            rank = 0, 
            world_size = 1, 
            dist_impl=None,
            device=None, 
        ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        self.num_classes = num_classes
        self.all_classes = torch.arange(num_classes, device=device, dtype=torch.float32)
        self.momentum = momentum
        self.topk_rate = topk_rate
        self.rel_matrix = rel_matrix.to(device=device, dtype=torch.float32) if rel_matrix is not None else None
        self.rel_real = rel_real.to(device=device, dtype=torch.float32) if rel_real is not None else None
        self.gt_mask = self.rel_real != 0 if rel_real is not None else torch.eye(num_classes, device=device, dtype=torch.bool)
        self.ema_threshold_on = ema_threshold_on
        self.statics = SimilarityStatisticsEMA(0.99, num_classes, device)

    @torch.no_grad()
    def update_prototypes(self, features, prototypes, classes, logits):
        # proto_sims = (features * prototypes[classes]).sum(dim=-1)
        proto_sims = logits.gather(1, classes.unsqueeze(1)).squeeze(1)
        sim_means = self.statics.mean[classes]
        threshold = sim_means * 0.9 if self.ema_threshold_on else -1
        selected = proto_sims >= threshold

        if not selected.any():
            return

        selected_classes = classes[selected]
        selected_features = features[selected]

        sums = torch.zeros_like(prototypes)
        counts = torch.zeros(self.num_classes, device=features.device, dtype=features.dtype)

        sums.index_add_(0, selected_classes, selected_features)
        counts.index_add_(
            0,
            selected_classes,
            torch.ones_like(selected_classes, dtype=features.dtype),
        )

        mask = counts > 0
        new_protos = sums[mask] / counts[mask, None]

        # momentum update
        prototypes[mask] = (
            self.momentum * prototypes[mask] + (1 - self.momentum) * new_protos
        )
        prototypes[mask] = F.normalize(prototypes[mask], dim=-1)


    def get_ground_truth(
        self, device, dtype, num_logits, classes
    ) -> torch.Tensor:
        if self.rel_matrix is not None and classes is not None:
            labels = self.rel_matrix[classes]

        elif classes is not None:
            labels = (classes[:, None] == self.all_classes[None, :]).float()

        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)

        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_topk_mask(self, logits, prototypes, classes=None):
        # protos = F.normalize(prototypes, dim=1)
        # logits = image_features @ prototypes.T

        k = int(prototypes.size(0) * self.topk_rate)    # 상위 topk_rate sim
        topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        
        return mask
    
    def not_far_sample(self, not_fars, logits, labels, classes):
        not_far_mask = torch.ones_like(logits)

        if not_fars is not None and not_fars.any():
            device = logits.device
            batch_size = logits.size(0)

            batch_idx = torch.arange(batch_size, device=device)

            own_logits = logits[batch_idx, classes] # 본인 클래스 로짓값
            not_far_mask[not_fars] = 0

            # 본인 클래스 weight 활성화
            not_far_indices = torch.where(not_fars)[0]
            not_far_mask[not_far_indices, classes[not_far_indices]] = 1

            # logit < 0.5
            low_mask = not_fars & (own_logits < 0.5)
            labels[batch_idx[low_mask], classes[low_mask]] = 0.5


            # logit > 0.7
            high_mask = not_fars & (own_logits > 0.7)
            labels[batch_idx[high_mask], classes[high_mask]] = 0.7

            # 0.5 <= logit <= 0.7
            middle_mask = (not_fars & (own_logits >= 0.5) & (own_logits <= 0.7))
            not_far_mask[batch_idx[middle_mask], classes[middle_mask]] = 0

        return not_far_mask, labels

    def tracking_loss(self, index, loss, labels, classes, sample_manager, not_fars):
        # self_logits = logits[torch.arange(0, logits.size(0)), classes]
        self_loss = loss[torch.arange(0, loss.size(0)), classes]
        self_labels = labels[torch.arange(0, loss.size(0)), classes]
        eps = 1e-8
        entropy = -(
            self_labels * torch.log(self_labels + eps)
            + (1 - self_labels) * torch.log(1 - self_labels + eps)
        )
        self_loss = (self_loss - entropy) * ~not_fars
        sample_manager.update(index, self_loss)


    def _loss(
            self, image_features, logits, ema_prototypes, memory_prototypes,
            logit_scale, logit_bias=None, train=False,
            classes=None, same_class_weight=1.0, 
            hards=None, centers=None, not_fars=None,
            index=None, sample_manager=None, sample_manager_update=True,
        ):
        topk_mask = self.get_topk_mask(logits, ema_prototypes, classes)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            classes,
        )

        not_far_mask, labels = self.not_far_sample(not_fars, logits, labels, classes)

        mask = (self.gt_mask[classes] | topk_mask) * not_far_mask
        # mask = (self.gt_mask[classes] | topk_mask)

        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

        if train and sample_manager_update:
            self.tracking_loss(index, loss, labels, classes, sample_manager, not_fars)

        loss = (loss * mask).sum() / mask.sum() # 마스킹된 부분만 평균내기
        
        return loss

    def forward(
            self, features, prototypes,
            logit_scale, logit_bias,
            classes, same_class_weight=1.0, 
            hards=None, centers=None, not_fars=None,
            index=None, sample_manager=None, sample_manager_update=True,
            output_dict=False,
            train=False, 
        ):
        logits = features @ prototypes.T
        # proto_sims = (features * prototypes).sum(dim=-1)

        if train and sample_manager_update:
            with torch.no_grad():
                # self_sims = (features * prototypes).sum(dim=-1)

                # self.statics.update(self_sims, classes)
                # z_score = self.statics.z_score(self_sims, classes)
                # mask = z_score < -3
                
                self.update_prototypes(features, prototypes, classes, logits)

        loss = self._loss(
            features, logits, prototypes, None,
            logit_scale, logit_bias, train,
            classes, same_class_weight,
            hards, centers, not_fars,
            index, sample_manager, sample_manager_update,
        )

        return {"contrastive_loss": loss} if output_dict else loss

def mse_loss(pred, label, mask):
    loss = F.mse_loss(pred, label, reduction='none')
    loss = (loss * mask).sum() / mask.sum()

    return loss

def similarity_mse_loss(features, classes, prototypes, rel_real):

    proto_targets = prototypes[classes]
    pred = features @ proto_targets.T

    rel = rel_real.to(device=features.device, dtype=features.dtype)
    # label = 2 * rel[classes][:, classes] - 1
    label = rel[classes][:, classes]
    mask = rel[classes][:, classes] != 0

    return mse_loss(pred, label, mask)

def content_style_sim_loss(content_features, style_features):

    cosine_sim = F.cosine_similarity(content_features, style_features, dim=-1)

    # return (cosine_sim**2).mean()
    return ((1 + cosine_sim)**2).mean()

class BCELoss(nn.Module):
    def __init__(self, class_weights=None, pos_weights=None, device='cpu'):
        super().__init__()
        self.class_weight = class_weights.to(device=device) if pos_weights is not None else None
        self.pos_weights = pos_weights.to(device=device) if pos_weights is not None else None

    def forward(self, features, labels):
        loss = F.binary_cross_entropy_with_logits(features, labels, reduction='none', pos_weight=self.pos_weights)
        if self.class_weight is not None:
            loss = loss * self.class_weight.to(dtype=loss.dtype)

        return loss.mean()


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

