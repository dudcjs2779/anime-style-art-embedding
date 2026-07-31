from typing import Optional
import torch
import torch.nn as nn

from torch.nn import functional as F

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

def growth_function(x, midpoint=0.5, steepness=10, max_y=0.5, tolerance=0.1):
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
    y = 2 * max_y / (1 + torch.exp(-steepness * (x - midpoint)))
    y[x < tolerance] = 0
    return  y
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


class SimilarityStatisticsEMA:
    def __init__(self, momentum, num_classes, device):
        self.momentum = momentum
        self.num_classes = num_classes
        self.mean = torch.zeros(
            num_classes,
            device=device,
        )

        # self.second_moment = torch.zeros(
        #     num_classes,
        #     device=device,
        # )

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

        # self.second_moment[mask] = (
        #     self.momentum * self.second_moment[mask] + (1 - self.momentum) * new_secnond
        # )

    # def variance(self, classes):
    #     return torch.clamp(
    #         self.second_moment[classes] - self.mean[classes] ** 2,
    #         min=1e-8,
    #     )

    # def std(self, classes):
    #     return torch.sqrt(self.variance(classes))




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


    def get_ground_truth(
        self, device, dtype, num_logits, classes, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, 
        negative_only=False
    ) -> torch.Tensor:
        if classes is not None:
            labels = (classes[:, None] == classes[None, :]).float()
            if times is not None:
                time_diff = torch.cdist(times[:, None], times[:, None], p=1.0)
                time_weights = growth_function(torch.clamp(time_diff, max=time_range) / time_range,
                                            midpoint=1.0, steepness=6, max_y=time_weight, tolerance=0.1)
                
                labels = labels * (same_class_weight - time_weights)
            else:
                eye = torch.eye(num_logits, dtype=torch.bool, device=classes.device)
                labels[labels.bool() & ~eye] = same_class_weight
        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)
        
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

    def _loss(
            self, image_features1, image_features2, 
            logit_scale, logit_bias=None, 
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3, 
            negative_only=False
            ):
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
        if classes is not None:
            loss_matrix  = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            mask = ~torch.eye(loss_matrix.size(0), dtype=torch.bool, device=loss_matrix.device)
            loss = loss_matrix[mask].mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        return loss

    def forward(
            self, image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3,
            output_dict=False
        ):
        loss = self._loss(
            image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes, same_class_weight, 
            times, time_range, time_weight
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
    
class RelationSigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self, rel_matrix, rel_real=None, loss_weight_on=False,
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
        self.rel_matrix = rel_matrix.to(device=device, dtype=torch.float32)
        self.rel_real = rel_real.to(device=device, dtype=torch.float32) if rel_real is not None else None
        self.gt_mask = self.rel_real != 0 if rel_real is not None else None
        self.loss_weight_on = loss_weight_on

    def get_ground_truth(
        self, device, dtype, num_logits, classes, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, 
        negative_only=False
    ) -> torch.Tensor:
        if classes is not None:
            labels = self.rel_matrix[classes][:, classes]
        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)
        
        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_logits(self, image_features1, image_features2, logit_scale, logit_bias=None, classes=None):
        output = {}
        logits = image_features1 @ image_features2.T

        # if classes is not None and self.loss_weight_on:
        #     loss_weights = torch.ones_like(logits, dtype=logits.dtype, device=logits.device)
        #     neg_mask = ~self.gt_mask[classes][:, classes]
        #     loss_weights[neg_mask & (logits > 0.5)] = 0.5
        #     loss_weights[neg_mask & (logits > 0.7)] = 0.1

        #     output["loss_weights"] = loss_weights

        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias

        output["logits"] = logits
        return output

    def _loss(
            self, image_features1, image_features2, 
            logit_scale, logit_bias=None, 
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3, 
            negative_only=False
        ):
        logits = self.get_logits(image_features1, image_features2, logit_scale, logit_bias, classes)
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
        if classes is not None:
            loss  = F.binary_cross_entropy_with_logits(logits["logits"], labels, reduction='none')

            # if self.loss_weight_on:
            #     loss = loss * logits["loss_weights"]

            # 자기자신과의 semantic한 의미 학습 방지하기
            mask = ~torch.eye(loss.size(0), dtype=torch.bool, device=loss.device)
            loss = loss[mask].mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits["logits"], labels, reduction='mean')

        return loss

    def forward(
            self, image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3,
            output_dict=False
        ):
        loss = self._loss(
            image_features1, image_features2, 
            logit_scale, logit_bias, 
            classes, same_class_weight, 
            times, time_range, time_weight
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
            self, num_classes, feat_dim, momentum, rel_matrix, rel_real=None, topk_rate=0.1,
            loss_weight_on=False, ema_threshold_on=False,
            rank = 0, 
            world_size = 1, 
            dist_impl = None,
            device=None, 
        ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        self.num_classes = num_classes
        self.all_classes = torch.arange(num_classes, device=device, dtype=torch.float32)
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.topk_rate = topk_rate
        self.rel_matrix = rel_matrix.to(device=device, dtype=torch.float32)
        self.rel_real = rel_real.to(device=device, dtype=torch.float32)
        self.gt_mask = self.rel_real != 0
        self.loss_weight_on = loss_weight_on
        self.ema_threshold_on = ema_threshold_on
        self.statics = SimilarityStatisticsEMA(0.99, num_classes, device)


    @torch.no_grad()
    def update_prototypes(self, features, prototypes, classes):
        proto_sims = (features * prototypes[classes]).sum(dim=-1)
        sim_means = self.statics.mean[classes]
        threshold = sim_means * 0.85 if self.ema_threshold_on else -1
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
        prototypes[mask] = torch.nn.functional.normalize(prototypes[mask], dim=-1)


    def get_ground_truth(
        self, device, dtype, num_logits, classes, same_class_weight=1.0, label_adjust=None,
    ) -> torch.Tensor:
        if classes is not None:
            labels = self.rel_matrix[classes]
            # labels = (classes[:, None] == self.all_classes[None, :]).float()
        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)

        if label_adjust is not None:
            labels = torch.clamp(labels + label_adjust, min=0, max=1.0)
        
        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_logits(self, image_features, prototypes, logit_scale, classes=None, logit_bias=None):
        output = {}
        # protos = F.normalize(prototypes, dim=1)
        logits = image_features @ prototypes.T

        self_indices = (torch.arange(logits.size(0), device=logits.device), classes)
        sims_self = logits[self_indices]

        if self.loss_weight_on:
            with torch.no_grad():
                proto_sims = prototypes[classes] @ prototypes.T
                rel_mask = self.gt_mask[classes]

                sub_style_weight = torch.clamp(1 + (sims_self - self.statics.mean[classes]), min=0.8, max=1.0)[:, None] # 선택된 샘플이 본인 클래스의 평균 유사도보다 조금 작은 경우 보정

                label_adjust = (logits - (proto_sims * sub_style_weight))
                label_adjust = torch.clamp(label_adjust, min=-0.1, max=0.1)
                label_adjust = torch.where(rel_mask, label_adjust, torch.zeros_like(label_adjust))
                label_adjust[self_indices] = 0

                output["label_adjust"] = label_adjust

        self.statics.update(sims_self, classes)

        k = int(prototypes.size(0) * self.topk_rate)    # 상위 topk_rate sim
        topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
        
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        mask = mask | self.gt_mask[classes]
        output["mask"] = mask

        # if self.loss_weight_on:
        #     diff = torch.clamp(1 + self.statics.mean[classes] - sims_self, min=1.0).to(dtype=sims_self.dtype)
        #     loss_weights = torch.ones_like(logits)
        #     loss_weights[self_indices] = diff
        #     output["loss_weights"] = loss_weights


        # mean = topk_values.mean(dim=-1, keepdim=True)   # mean이 높다 => 비슷한 그림체가 많다(또는 흔하다) => 컷을 높여 진짜 비슷한 그림만 선별한다
        # std = topk_values.std(dim=-1, keepdim=True) # std가 높다 => 특별히 비슷한 그림체가 있다(또는 개성있다) => 컷을 높여 진짜 비슷한 그림만 선별한다

        # threshold = mean + std
        # mask = (logits > threshold) | (self.gt_mask[classes])
        # output["mask"] = mask

        # # 연관성 라벨이 없음에도 높은 sim 값을 가진 경우 loss의 영향력을 낮춰준다.
        # if self.loss_weight_on:
        #     # sim 값이 0.7 이상인 경우의 loss 감쇄할 class의 갯수를 제한한다. => 무분별하게 전부 sim 값이 높아지도록 학습하는 것 방지
        #     w_k = int(k * 0.2)    # 상위 topk_rate sim
        #     top_wk_indices = topk_indices[:, :w_k]

        #     top_wk_mask = torch.zeros_like(logits, dtype=torch.bool)
        #     top_wk_mask.scatter_(dim=-1, index=top_wk_indices, value=True)

        #     loss_weights = torch.ones_like(logits, dtype=logits.dtype, device=logits.device)
        #     neg_mask = ~self.gt_mask[classes]   # 연관성 라벨이 없는 경우
        #     loss_weights[neg_mask & (logits > 0.5)] = 0.5
        #     loss_weights[top_wk_mask & neg_mask & (logits > 0.7)] = 0.1
        #     output["loss_weights"] = loss_weights

        logits = logit_scale * logits
        if logit_bias is not None:
            logits += logit_bias

        output["logits"] = logits
        return output
    
    def _loss(
            self, image_features, ema_prototypes, memory_prototypes,
            logit_scale, logit_bias=None, 
            classes=None, same_class_weight=1.0, 
            ):
        logits = self.get_logits(image_features, ema_prototypes, logit_scale, classes, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            classes,
            same_class_weight,
            label_adjust=logits.get('label_adjust')
        )
        # loss = -F.logsigmoid(labels * logits).sum() / image_features1.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits['logits'], labels, reduction='none')

        # if self.loss_weight_on:
        #     loss = loss * logits['loss_weights']

        # loss = loss.mean()

        loss = (loss * logits['mask']).sum() / logits['mask'].sum() # 마스킹된 부분만 평균내기
        
        # # negative 데이터 영향력 낮추기
        # weights = torch.where(labels == 0, 0.01, 1.0)
        # loss = (loss * weights).mean()
        return loss

    def forward(
            self, features, prototypes,
            logit_scale, logit_bias, 
            classes, same_class_weight=1.0, 
            output_dict=False,
            train=False,
        ):
        if train:
            with torch.no_grad():
                # self_sims = (features * prototypes).sum(dim=-1)

                # self.statics.update(self_sims, classes)
                # z_score = self.statics.z_score(self_sims, classes)
                # mask = z_score < -3

                
                self.update_prototypes(features, prototypes, classes)

        loss = self._loss(
            features, prototypes, None,
            logit_scale, logit_bias, 
            classes, same_class_weight, 
        )

        return {"contrastive_loss": loss} if output_dict else loss
    

class PrototypeContrastiveLoss(nn.Module):
    def __init__(
            self, num_classes, feat_dim, momentum, 
            rank = 0, 
            world_size = 1, 
            dist_impl = None,
            device=None, 
        ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        self.num_classes = num_classes
        self.all_classes = torch.arange(num_classes, device=device, dtype=torch.float32)
        self.feat_dim = feat_dim
        self.momentum = momentum


    @torch.no_grad()
    def update_prototypes(self, features, prototypes, classes):
        sums = torch.zeros_like(prototypes)
        counts = torch.zeros(self.num_classes, device=features.device)

        sums.index_add_(0, classes, features)
        counts.index_add_(0, classes, torch.ones_like(classes, dtype=features.dtype))

        mask = counts > 0
        new_protos = torch.zeros_like(prototypes)
        new_protos[mask] = sums[mask] / counts[mask, None]

        # momentum update
        prototypes[mask] = (
            self.momentum * prototypes[mask] + (1 - self.momentum) * new_protos[mask]
        )
        prototypes[mask] = F.normalize(prototypes[mask], dim=-1)

        # """ 배치 내 feature로 prototype 갱신 (momentum 평균) """
        # _classes = classes.unique().to(dtype=int)
        # for cls in _classes:
        #     mask = (classes == cls)
        #     new_proto = features[mask].mean(dim=0)
        #     prototypes[cls] = (
        #         self.momentum * prototypes[cls] + (1 - self.momentum) * new_proto
        #     )

        # prototypes[_classes] = F.normalize(prototypes[_classes], dim=-1)



    def get_ground_truth(
        self, device, dtype, num_logits, classes, same_class_weight=1.0, 
    ) -> torch.Tensor:
        if classes is not None:
            labels = (classes[:, None] == self.all_classes[None, :]).float()
        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)
        
        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_logits(self, image_features, prototypes, logit_scale, logit_bias=None):
        # protos = F.normalize(prototypes, dim=1)
        logits = logit_scale * image_features @ prototypes.T
        if logit_bias is not None:
            logits += logit_bias
        return logits
    
    def _loss(
            self, image_features, prototypes,
            logit_scale, logit_bias=None, 
            classes=None, same_class_weight=1.0, 
            ):
        logits = self.get_logits(image_features, prototypes, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            classes,
            same_class_weight,
        )
        # loss = -F.logsigmoid(labels * logits).sum() / image_features1.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        return loss

    def forward(
            self, features, prototypes,
            logit_scale, logit_bias, 
            classes, same_class_weight=1.0, 
            output_dict=False,
            train=False,
        ):
        if train:
            with torch.no_grad():
                self.update_prototypes(features, prototypes, classes)

        loss = self._loss(
            features, prototypes,
            logit_scale, logit_bias, 
            classes, same_class_weight, 
        )

        return {"contrastive_loss": loss} if output_dict else loss


class SigLipLoss2(nn.Module):
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


    def get_ground_truth(
        self, device, dtype, num_logits, 
        s2c_tag_ratio=None, c2s_tag_ratio=None, batch_size=0,
        full_tags=None, sel_tags=None,
        classes=None, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, 
        negative_only=False
    ) -> torch.Tensor:
        if full_tags is not None:
            # labels = torch.zeros(num_logits, num_logits, device=device, dtype=dtype)
            overlap = full_tags @ sel_tags.T  # (num_full, num_sel)
            sel_counts = sel_tags.sum(dim=1)
            labels = overlap / sel_counts[None, :] / 2

            # for i, full_tag in enumerate(full_tags):
            #     for j, sel_tag in enumerate(sel_tags):
            #         full_tag, sel_tag = set(full_tag), set(sel_tag)
            #         overlap_count = len(full_tag.intersection(sel_tag))

            #         labels[i, j] = overlap_count / len(sel_tag) / 2

            class_mask = classes[:, None] == classes[None, :]
            labels[class_mask] += 0.5
        
        # elif s2c_tag_ratio is not None:
        #     num_styled = batch_size // 2
        #     accum = num_logits // batch_size

        #     s2c_tag_ratio_chunked = torch.chunk(s2c_tag_ratio, accum, dim=0)
        #     c2s_tag_ratio_chunked = torch.chunk(c2s_tag_ratio, accum, dim=0)

        #     # 기본 단위행렬
        #     labels = torch.eye(num_logits, device=device, dtype=torch.float32)
        #     idx = torch.arange(num_styled)
            
        #     for i in range(accum):
        #         start_point = batch_size * i

        #         # 위쪽 블록 수정
        #         labels[start_point + idx, start_point + num_styled + idx] = s2c_tag_ratio_chunked[i]
        #         # 아래쪽 블록 수정
        #         labels[start_point+ num_styled + idx, start_point + idx] = c2s_tag_ratio_chunked[i]

        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)


        labels = labels.to(device=device, dtype=dtype)

        # labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        # if not negative_only:
        #     labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, features1, features2, logit_scale, logit_bias=None):
        logits = logit_scale * features1 @ features2.T
        if logit_bias is not None:
            logits += logit_bias

        return logits

    def _loss(
        self, features1, features2, logit_scale, logit_bias=None, 
        s2c_tag_ratio=None, c2s_tag_ratio=None, batch_size=0,
        full_tags=None, sel_tags=None,
        classes=None, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, negative_only=False
    ):
        logits = self.get_logits(features1, features2, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            features1.device, features1.dtype, features1.shape[0],
            s2c_tag_ratio=s2c_tag_ratio, c2s_tag_ratio=c2s_tag_ratio, batch_size=batch_size,
            full_tags=full_tags, sel_tags=sel_tags,
            classes=classes, same_class_weight=same_class_weight,
            times=times, time_range=time_range, time_weight=time_weight,
            negative_only=negative_only,
        )

        if features1 is features2:
            loss_matrix  = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            mask = ~torch.eye(loss_matrix.size(0), dtype=torch.bool, device=loss_matrix.device)
            loss = loss_matrix[mask].mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        # loss = -F.logsigmoid(labels * logits).sum() / image_features1.shape[0]
        return loss
    
    def forward(
            self, features1, features2, logit_scale, logit_bias=None, 
            s2c_tag_ratio=None, c2s_tag_ratio=None, batch_size=0,
            full_tags=None, sel_tags=None,
            classes=None, same_class_weight=1.0, 
            times=None, time_range=10, time_weight=0.3,
            output_dict=False
        ):
        loss = self._loss(
            features1, features2, logit_scale, logit_bias, 
            s2c_tag_ratio, c2s_tag_ratio, batch_size,
            full_tags, sel_tags,
            classes, same_class_weight,
            times, time_range, time_weight
        )

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                image_features2_to_right = image_features2_to_left = features2
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
                            features1,
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
                        features1,
                        image_features2_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                image_features2_to_right = features2
                for i in range(self.world_size - 1):
                    image_features2_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        image_features2_to_right,
                    )
                    loss += self._loss(
                        features1,
                        image_features2_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    image_features2_to_right = image_features2_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    image2_from_other = torch.distributed.nn.all_reduce(
                        features2 * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        features1,
                        image2_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_images2 = torch.distributed.nn.all_gather(features2)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        features1,
                        all_images2[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
    
class StyleSigLip(SigLipLoss2):
    def __init__(self, cache_labels = False, rank = 0, world_size = 1, dist_impl = None):
        super().__init__(cache_labels, rank, world_size, dist_impl)

    def get_ground_truth(
        self, device, dtype, num_logits, 
        classes=None, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, 
        negative_only=False
    ) -> torch.Tensor:
        if classes is not None:
            labels = (classes[:, None] == classes[None, :]).float()
            if times is not None:
                time_diff = torch.cdist(times[:, None], times[:, None], p=1.0)
                time_weights = growth_function(torch.clamp(time_diff, max=time_range) / time_range, midpoint=1.0, steepness=6, max_y=time_weight)
                
                labels = labels * (same_class_weight - time_weights)
                labels.fill_diagonal_(1.0)

                content_mask = (classes[:, None] == 0) & (classes[None, :] == 0)
                labels[content_mask] = 1.0
            else:
                eye = torch.eye(num_logits, dtype=torch.bool, device=device)
                labels[labels.bool() & ~eye] = same_class_weight

        else:
            labels = torch.eye(num_logits, dtype=dtype, device=device)


        labels = labels.to(device=device, dtype=dtype)

        return labels
    
    def _loss(
        self, features1, features2, logit_scale, logit_bias=None, 
        s2c_tag_ratio=None, c2s_tag_ratio=None, batch_size=0,
        full_tags=None, sel_tags=None,
        classes=None, same_class_weight=1.0, 
        times=None, time_range=10, time_weight=0.3, negative_only=False
    ):
        logits = self.get_logits(features1, features2, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            features1.device, features1.dtype, features1.shape[0],
            classes=classes, same_class_weight=same_class_weight,
            times=times, time_range=time_range, time_weight=time_weight,
            negative_only=negative_only,
        )
        loss_matrix  = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        content_mask = (classes[:, None] == 0) & (classes[None, :] == 0)
        loss_matrix[content_mask] *= 0.1

        if features1 is features2:
            mask = ~torch.eye(loss_matrix.size(0), dtype=torch.bool, device=loss_matrix.device)
            loss = loss_matrix[mask].mean()
        else:
            loss = loss_matrix.mean()

        return loss



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

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, classes, full_tags, sel_tags) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        # if self.prev_num_logits != num_logits or device not in self.labels:
        #     labels = torch.arange(num_logits, device=device, dtype=torch.long)
        #     if self.world_size > 1 and self.local_loss:
        #         labels = labels + num_logits * self.rank
        #     if self.cache_labels:
        #         self.labels[device] = labels
        #         self.prev_num_logits = num_logits
        # else:
        #     labels = self.labels[device]

        if full_tags is not None:
            labels = torch.zeros(num_logits, num_logits, device=device, dtype=dtype)
            for i, full_tag in enumerate(full_tags):
                for j, sel_tag in enumerate(sel_tags):
                    full_tag, sel_tag = set(full_tag), set(sel_tag)
                    overlap_count = len(full_tag.intersection(sel_tag))

                    labels[i, j] = overlap_count / len(sel_tag) / 2

            class_mask = classes[:, None] == classes[None, :]
            labels[class_mask] += 0.5

        labels = labels.to(device=device, dtype=dtype)

        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            features1,
            features2,
            logit_scale,
            logit_bias=None,
            classes=None, full_tags=None, sel_tags=None,
            output_dict=False,
    ):
        device = features1.device
        dtype = features1.dtype

        logits_per_image, logits_per_text = self.get_logits(
            features1,
            features2,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(
            device, 
            dtype, 
            logits_per_image.shape[0], 
            classes, full_tags, sel_tags
        )

        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        # ) / 2

        total_loss = (
            F.kl_div(F.log_softmax(logits_per_image, dim=1), F.softmax(labels, dim=1), reduction="batchmean") +
            F.kl_div(F.log_softmax(logits_per_text, dim=1), F.softmax(labels, dim=1), reduction="batchmean")
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()



