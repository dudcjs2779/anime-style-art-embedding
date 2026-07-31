import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_embed_metrics(image_features, classes):
    metrics = {}

    ## 분류 accuracy
    logits_image = (image_features @ image_features.t()).detach().cpu()

    ground_truth = (classes.unsqueeze(0) == classes.unsqueeze(1)).int()
    N = logits_image.size(0)

    # 자기 자신 제외
    logits_no_diag = logits_image.clone()
    logits_no_diag.fill_diagonal_(-float("inf"))

    ## 클래스 별 총 데이터수 만큼의 top-k 랭크에서 얼마나 해당 class를 상위에 랭크시켰는지 accuracy 계산
    sorted_idx = logits_no_diag.argsort(dim=1, descending=True)  # 각 row에서 similarity 순으로 정렬한 index
    gt_mask = ground_truth.gather(1, sorted_idx)  # ground_truth 요소를 sorted_idx의 순서대로 재배열 즉, 높은 sim에 해당하는 ground_truth값이 앞으로 배치됨

    num_gt_per_row = ground_truth.sum(dim=1) # 각 샘플별 gt의 갯수
    ranks = torch.arange(N, device=logits_image.device).unsqueeze(0)

    rank_range = (ranks < num_gt_per_row.unsqueeze(1)) # (존재하는 클래스 갯수 만큼 앞부분을 True)
    rank_range_x1_5 = (ranks < (num_gt_per_row * 1.5).to(dtype=int).unsqueeze(1))
    rank_range_x3 = (ranks < (num_gt_per_row * 3).to(dtype=int).unsqueeze(1))
    rank_range_x10 = (ranks < (num_gt_per_row * 10).to(dtype=int).unsqueeze(1))

    acc_per_row = rank_range & (gt_mask.bool()) # (측정 랭킹 구간) & (sim이 높은 순으로 정렬한 gt_mask)
    acc_per_row_x1_5 = rank_range_x1_5 & (gt_mask.bool())
    acc_per_row_x3 = rank_range_x3 & (gt_mask.bool())
    acc_per_row_x10 = rank_range_x10 & (gt_mask.bool())

    accuracy = acc_per_row.float().sum(dim=1) / num_gt_per_row
    accuracy_x1_5 = acc_per_row_x1_5.float().sum(dim=1) / num_gt_per_row
    accuracy_x3 = acc_per_row_x3.float().sum(dim=1) / num_gt_per_row
    accuracy_x10 = acc_per_row_x10.float().sum(dim=1) / num_gt_per_row

    accuracy = accuracy.mean().item()
    accuracy_x1_5 = accuracy_x1_5.mean().item()
    accuracy_x3 = accuracy_x3.mean().item()
    accuracy_x10 = accuracy_x10.mean().item()

    metrics[f"accuracy"] = accuracy
    metrics[f"accuracy_x1_5"] = accuracy_x1_5
    metrics[f"accuracy_x3"] = accuracy_x3
    metrics[f"accuracy_x10"] = accuracy_x10

    ## 클래스별 평균 유사도 및 표준편차
    image_features = image_features.numpy()
    classes = classes.numpy()
    unique_classes = np.unique(classes)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = image_features[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    metrics[f"sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"sim_stu_mean_by_class"] = np.mean(std_list)

    ## silhouette_score
    metrics[f"silhouette_score_real_class"] = silhouette_score(image_features, classes, metric='cosine')

    kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
    labels = kmeans.fit_predict(image_features)
    metrics[f"silhouette_score_kmeans"] = silhouette_score(image_features, labels, metric='cosine')

    return metrics

def get_proto_metrics(image_features, prototypes, classes):
    metrics = {}

    ## 분류 accuracy
    logits_image = (prototypes @ image_features.t()).detach().cpu()
    unique_classes = classes.unique()

    ground_truth = (unique_classes.unsqueeze(1) == classes.unsqueeze(0)).int()
    N = image_features.size(0)

    # 자기 자신 제외
    logits_no_diag = logits_image.clone()

    ## 클래스 별 총 데이터수 만큼의 top-k 랭크에서 얼마나 해당 class를 상위에 랭크시켰는지 accuracy 계산
    sorted_idx = logits_no_diag.argsort(dim=1, descending=True)  # 각 row에서 similarity 순으로 정렬한 index
    gt_mask = ground_truth.gather(1, sorted_idx)  # ground_truth 요소를 sorted_idx의 순서대로 재배열 즉, 높은 sim에 해당하는 ground_truth값이 앞으로 배치됨

    num_gt_per_row = ground_truth.sum(dim=1) # 각 샘플별 gt의 갯수
    ranks = torch.arange(N, device=logits_image.device).unsqueeze(0)

    rank_range = (ranks < num_gt_per_row.unsqueeze(1)) # (존재하는 클래스 갯수 만큼 앞부분을 True)
    rank_range_x1_5 = (ranks < (num_gt_per_row * 1.5).to(dtype=int).unsqueeze(1))
    rank_range_x3 = (ranks < (num_gt_per_row * 3).to(dtype=int).unsqueeze(1))
    rank_range_x10 = (ranks < (num_gt_per_row * 10).to(dtype=int).unsqueeze(1))

    acc_per_row = rank_range & (gt_mask.bool()) # (측정 랭킹 구간) & (sim이 높은 순으로 정렬한 gt_mask)
    acc_per_row_x1_5 = rank_range_x1_5 & (gt_mask.bool())
    acc_per_row_x3 = rank_range_x3 & (gt_mask.bool())
    acc_per_row_x10 = rank_range_x10 & (gt_mask.bool())

    accuracy = acc_per_row.float().sum(dim=1) / num_gt_per_row
    accuracy_x1_5 = acc_per_row_x1_5.float().sum(dim=1) / num_gt_per_row
    accuracy_x3 = acc_per_row_x3.float().sum(dim=1) / num_gt_per_row
    accuracy_x10 = acc_per_row_x10.float().sum(dim=1) / num_gt_per_row

    accuracy = accuracy.mean().item()
    accuracy_x1_5 = accuracy_x1_5.mean().item()
    accuracy_x3 = accuracy_x3.mean().item()
    accuracy_x10 = accuracy_x10.mean().item()

    metrics[f"accuracy"] = accuracy
    metrics[f"accuracy_x1_5"] = accuracy_x1_5
    metrics[f"accuracy_x3"] = accuracy_x3
    metrics[f"accuracy_x10"] = accuracy_x10

    ## 클래스별 평균 유사도 및 표준편차
    image_features = image_features.numpy()
    classes = classes.numpy()
    unique_classes = np.unique(classes)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = image_features[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    metrics[f"sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"sim_stu_mean_by_class"] = np.mean(std_list)

    ## silhouette_score
    metrics[f"silhouette_score_real_class"] = silhouette_score(image_features, classes, metric='cosine')

    kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
    labels = kmeans.fit_predict(image_features)
    metrics[f"silhouette_score_kmeans"] = silhouette_score(image_features, labels, metric='cosine')

    return metrics

def get_class_topk(image_features, prototypes, classes, rel_real):
    metrics = {}

    ## TopK
    logits_image = (image_features @ prototypes.t()).detach().cpu()
    ranks = torch.argsort(logits_image, dim=1, descending=True)

    target_pos = (ranks == classes[:, None]).float()
    self_class_rank = target_pos.argmax(axis=1)
    self_class_rank = self_class_rank.detach().cpu().numpy()

    for k in [1, 2, 5, 10]:
        metrics[f"Class_R@{k}"] = np.mean(self_class_rank < k)

    ## 클래스별 평균 유사도 및 표준편차
    image_features = image_features.numpy()
    classes = classes.numpy()
    unique_classes = np.unique(classes)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = image_features[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    metrics[f"sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"sim_stu_mean_by_class"] = np.mean(std_list)

    # topk 평균 유사도 => 과도하게 뭉치는 정도 계산
    sim_matrix = prototypes @ prototypes.T
    sim_matrix.fill_diagonal_(-1)

    k = int(prototypes.size(0) * 0.1)    # 상위 topk_rate sim
    topk_values, _ = torch.topk(sim_matrix, k=k, dim=-1)
    
    metrics[f"topk_mean_sim"] = topk_values.mean().item()

    # args.rel_real
    proto_matrix = (prototypes @ prototypes.t()).detach().cpu()
    labeled = rel_real != 0
    labeled.fill_diagonal_(False)

    mae_error = np.abs(proto_matrix[labeled] - rel_real[labeled]).mean()
    metrics[f"mae_error"] = mae_error.item()

    ## silhouette_score
    metrics[f"silhouette_score_real_class"] = silhouette_score(image_features, classes, metric='cosine')

    kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
    labels = kmeans.fit_predict(image_features)
    metrics[f"silhouette_score_kmeans"] = silhouette_score(image_features, labels, metric='cosine')

    return metrics



def get_class_clip_metrics(features1, features2, classes):
    metrics = {}
    
    ## 분류 accuracy
    logits = (features1 @ features2.t()).detach().cpu()

    ground_truth = (classes.unsqueeze(0) == classes.unsqueeze(1)).int()
    N = logits.size(0)

    # # 자기 자신 제외
    # logits_no_diag = logits.clone()
    # logits_no_diag.fill_diagonal_(-float("inf"))

    ## 클래스 별 총 데이터수 만큼의 top-k 랭크에서 얼마나 해당 class를 상위에 랭크시켰는지 accuracy 계산
    sorted_idx = logits.argsort(dim=1, descending=True)  # 각 row에서 similarity 순으로 정렬한 index
    gt_mask = ground_truth.gather(1, sorted_idx)  # ground_truth 요소를 sorted_idx의 순서대로 재배열 즉, 높은 sim에 해당하는 ground_truth값이 앞으로 배치됨

    num_gt_per_row = ground_truth.sum(dim=1) # 각 샘플별 gt의 갯수
    ranks = torch.arange(N, device=logits.device).unsqueeze(0)

    rank_range = (ranks < num_gt_per_row.unsqueeze(1)) # (존재하는 클래스 갯수 만큼 앞부분을 True)
    rank_range_x1_5 = (ranks < (num_gt_per_row * 1.5).to(dtype=int).unsqueeze(1))
    rank_range_x3 = (ranks < (num_gt_per_row * 3).to(dtype=int).unsqueeze(1))
    rank_range_x10 = (ranks < (num_gt_per_row * 10).to(dtype=int).unsqueeze(1))

    acc_per_row = rank_range & (gt_mask.bool()) # (측정 랭킹 구간) & (sim이 높은 순으로 정렬한 gt_mask)
    acc_per_row_x1_5 = rank_range_x1_5 & (gt_mask.bool())
    acc_per_row_x3 = rank_range_x3 & (gt_mask.bool())
    acc_per_row_x10 = rank_range_x10 & (gt_mask.bool())

    accuracy = acc_per_row.float().sum(dim=1) / num_gt_per_row
    accuracy_x1_5 = acc_per_row_x1_5.float().sum(dim=1) / num_gt_per_row
    accuracy_x3 = acc_per_row_x3.float().sum(dim=1) / num_gt_per_row
    accuracy_x10 = acc_per_row_x10.float().sum(dim=1) / num_gt_per_row

    accuracy = accuracy.mean().item()
    accuracy_x1_5 = accuracy_x1_5.mean().item()
    accuracy_x3 = accuracy_x3.mean().item()
    accuracy_x10 = accuracy_x10.mean().item()

    metrics[f"accuracy"] = accuracy
    metrics[f"accuracy_x1_5"] = accuracy_x1_5
    metrics[f"accuracy_x3"] = accuracy_x3
    metrics[f"accuracy_x10"] = accuracy_x10

    ## 클래스별 평균 유사도 및 표준편차
    features1 = features1.numpy()
    features2 = features2.numpy()
    classes = classes.numpy()
    unique_classes = np.unique(classes)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = features1[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    metrics[f"im_sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"im_sim_stu_mean_by_class"] = np.mean(std_list)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = features2[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    
    metrics[f"tt_sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"tt_sim_stu_mean_by_class"] = np.mean(std_list)

    ## silhouette_score
    metrics[f"silhouette_score_real_class"] = silhouette_score(features1, classes, metric='cosine')

    # kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
    # labels = kmeans.fit_predict(features1)
    # metrics[f"silhouette_score_kmeans"] = silhouette_score(features1, labels, metric='cosine')

    return metrics

def class_accuracy(logits, classes):
    metrics = {}
    ground_truth = (classes.unsqueeze(0) == classes.unsqueeze(1)).int()
    N = logits.size(0)

    sorted_idx = logits.argsort(dim=1, descending=True)  # 각 row에서 similarity 순으로 정렬한 index
    gt_mask = ground_truth.gather(1, sorted_idx)  # ground_truth 요소를 sorted_idx의 순서대로 재배열 즉, 높은 sim에 해당하는 ground_truth값이 앞으로 배치됨

    num_gt_per_row = ground_truth.sum(dim=1) # 각 샘플별 gt의 갯수
    ranks = torch.arange(N, device=logits.device).unsqueeze(0)

    rank_range = (ranks < num_gt_per_row.unsqueeze(1)) # (존재하는 클래스 갯수 만큼 앞부분을 True)
    # rank_range_x1_5 = (ranks < (num_gt_per_row * 1.5).to(dtype=int).unsqueeze(1))
    rank_range_x3 = (ranks < (num_gt_per_row * 3).to(dtype=int).unsqueeze(1))
    rank_range_x10 = (ranks < (num_gt_per_row * 10).to(dtype=int).unsqueeze(1))

    acc_per_row = rank_range & (gt_mask.bool()) # (측정 랭킹 구간) & (sim이 높은 순으로 정렬한 gt_mask)
    # acc_per_row_x1_5 = rank_range_x1_5 & (gt_mask.bool())
    acc_per_row_x3 = rank_range_x3 & (gt_mask.bool())
    acc_per_row_x10 = rank_range_x10 & (gt_mask.bool())

    accuracy = acc_per_row.float().sum(dim=1) / num_gt_per_row
    # accuracy_x1_5 = acc_per_row_x1_5.float().sum(dim=1) / num_gt_per_row
    accuracy_x3 = acc_per_row_x3.float().sum(dim=1) / num_gt_per_row
    accuracy_x10 = acc_per_row_x10.float().sum(dim=1) / num_gt_per_row

    accuracy = accuracy.mean().item()
    # accuracy_x1_5 = accuracy_x1_5.mean().item()
    accuracy_x3 = accuracy_x3.mean().item()
    accuracy_x10 = accuracy_x10.mean().item()

    metrics[f"accuracy"] = accuracy
    # metrics[f"accuracy_x1_5"] = accuracy_x1_5
    metrics[f"accuracy_x3"] = accuracy_x3
    metrics[f"accuracy_x10"] = accuracy_x10

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale, classes):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

        class_metrics = class_accuracy(logit, classes)
        class_metrics = {f"{name}_{k}":v for k, v in class_metrics.items()}
        metrics.update(class_metrics)

    return metrics


def get_style_clip_metrics(image_style, text_style, logit_scale, classes):
    metrics = {}
    i2t_logits = (logit_scale * image_style @ text_style.t()).detach().cpu()
    t2i_logits = i2t_logits.t().detach().cpu()
    i2i_logits = (logit_scale * image_style @ image_style.t()).detach().cpu()

    logits = {"image_to_text": i2t_logits, "text_to_image": t2i_logits, "image_to_image": i2i_logits}
    for name, logit in logits.items():
        class_metrics = class_accuracy(logit, classes)
        class_metrics = {f"{name}_{k}":v for k, v in class_metrics.items()}
        metrics.update(class_metrics)

    return metrics

