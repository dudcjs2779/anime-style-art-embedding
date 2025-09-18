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