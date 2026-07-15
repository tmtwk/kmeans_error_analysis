import numpy as np

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_generator import create_dataset, get_feature_data


def scale_data(data):
    """
    난이도 지수와 알고리즘 특성 지수를 표준화한다.

    두 특성의 평균을 0, 표준편차를 1에 가깝게 변환하여
    한 특성이 거리 계산에 지나치게 큰 영향을 주는 것을 방지한다.
    """

    # 군집화에 사용할 두 특성만 추출
    feature_data = get_feature_data(data)

    scaler = StandardScaler()

    # 표준화 학습 및 변환
    scaled_data = scaler.fit_transform(feature_data)

    return scaled_data, scaler


def run_kmeans(
    scaled_data,
    n_clusters=3,
    random_state=42
):
    """
    표준화된 데이터에 K-Means 군집화를 적용한다.
    """

    kmeans_model = KMeans(
        n_clusters=n_clusters,

        # 초기 중심점을 무작위로 선택
        init="random",

        # 서로 다른 초기 중심점으로 10번 실행한 뒤
        # 군집 내 거리 제곱합이 가장 작은 결과 선택
        n_init=10,

        # 최대 반복 횟수
        max_iter=300,

        # 같은 결과가 재현되도록 난수 고정
        random_state=random_state
    )

    # 모델 학습과 군집 번호 예측을 동시에 수행
    kmeans_labels = kmeans_model.fit_predict(scaled_data)

    return kmeans_model, kmeans_labels


def get_original_centroids(kmeans_model, scaler):
    """
    표준화된 공간의 K-Means 중심점을
    원래의 1~100 좌표 범위로 되돌린다.
    """

    original_centroids = scaler.inverse_transform(
        kmeans_model.cluster_centers_
    )

    return original_centroids


def run_dbscan(
    scaled_data,
    eps=0.45,
    min_samples=4
):
    """
    표준화된 데이터에 DBSCAN 군집화를 적용한다.

    eps:
        한 데이터의 이웃으로 인정할 최대 거리

    min_samples:
        핵심점이 되기 위해 주변에 필요한 최소 데이터 수
    """

    dbscan_model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    dbscan_labels = dbscan_model.fit_predict(scaled_data)

    return dbscan_model, dbscan_labels


def count_clusters(labels):
    """
    군집 번호 배열에서 실제 군집 수를 계산한다.

    DBSCAN에서 -1은 노이즈를 의미하므로
    군집 수 계산에서 제외한다.
    """

    unique_labels = set(labels)

    cluster_count = len(unique_labels)

    if -1 in unique_labels:
        cluster_count -= 1

    return cluster_count


def count_noise(labels):
    """
    DBSCAN이 -1로 분류한 노이즈 데이터 수를 계산한다.
    """

    return int(np.sum(labels == -1))


def calculate_silhouette(scaled_data, labels):
    """
    실루엣 계수를 계산한다.

    DBSCAN의 경우 노이즈(-1)는 계산에서 제외한다.
    실루엣 계수는 1에 가까울수록 군집이 잘 분리되었음을 의미한다.
    """

    # 노이즈가 아닌 데이터만 선택
    valid_mask = labels != -1

    valid_data = scaled_data[valid_mask]
    valid_labels = labels[valid_mask]

    # 군집이 2개 이상이어야 실루엣 계수 계산 가능
    cluster_count = len(set(valid_labels))

    if cluster_count < 2:
        return None

    # 각 데이터가 하나씩만 남는 비정상적 상황 방지
    if len(valid_data) <= cluster_count:
        return None

    score = silhouette_score(
        valid_data,
        valid_labels
    )

    return score


def cluster_data(
    data,
    n_clusters=3,
    eps=0.45,
    min_samples=4,
    random_state=42
):
    """
    데이터 표준화, K-Means, DBSCAN을 한 번에 실행한다.
    """

    # 1. 데이터 표준화
    scaled_data, scaler = scale_data(data)

    # 2. K-Means 실행
    kmeans_model, kmeans_labels = run_kmeans(
        scaled_data,
        n_clusters=n_clusters,
        random_state=random_state
    )

    # 3. K-Means 중심점을 원래 좌표로 복원
    original_centroids = get_original_centroids(
        kmeans_model,
        scaler
    )

    # 4. DBSCAN 실행
    dbscan_model, dbscan_labels = run_dbscan(
        scaled_data,
        eps=eps,
        min_samples=min_samples
    )

    # 5. 결과를 딕셔너리 형태로 반환
    results = {
        "scaled_data": scaled_data,
        "scaler": scaler,

        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_labels,
        "kmeans_centroids": original_centroids,
        "kmeans_cluster_count": count_clusters(kmeans_labels),
        "kmeans_silhouette": calculate_silhouette(
            scaled_data,
            kmeans_labels
        ),

        "dbscan_model": dbscan_model,
        "dbscan_labels": dbscan_labels,
        "dbscan_cluster_count": count_clusters(dbscan_labels),
        "dbscan_noise_count": count_noise(dbscan_labels),
        "dbscan_silhouette": calculate_silhouette(
            scaled_data,
            dbscan_labels
        )
    }

    return results


def main():
    # 기본 오답 데이터 50개와 이상치 5개 생성
    dataset = create_dataset(
        n_samples=50,
        random_state=42,
        include_outliers=True
    )

    # K-Means와 DBSCAN 실행
    results = cluster_data(
        data=dataset,
        n_clusters=3,
        eps=0.45,
        min_samples=4,
        random_state=42
    )

    print("===== K-Means 실행 결과 =====")
    print(
        "군집 수:",
        results["kmeans_cluster_count"]
    )
    print(
        "중심점 갱신 반복 횟수:",
        results["kmeans_model"].n_iter_
    )
    print(
        "군집 내 거리 제곱합:",
        round(results["kmeans_model"].inertia_, 4)
    )
    print(
        "실루엣 계수:",
        round(results["kmeans_silhouette"], 4)
    )

    print("\nK-Means 중심점")
    print(
        np.round(
            results["kmeans_centroids"],
            2
        )
    )

    print("\n===== DBSCAN 실행 결과 =====")
    print(
        "군집 수:",
        results["dbscan_cluster_count"]
    )
    print(
        "노이즈 수:",
        results["dbscan_noise_count"]
    )

    if results["dbscan_silhouette"] is not None:
        print(
            "노이즈 제외 실루엣 계수:",
            round(results["dbscan_silhouette"], 4)
        )
    else:
        print(
            "노이즈 제외 실루엣 계수:",
            "계산할 수 없음"
        )


if __name__ == "__main__":
    main()