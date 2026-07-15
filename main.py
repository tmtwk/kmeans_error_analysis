from data_generator import (
    create_dataset,
    get_feature_data,
    save_dataset
)

from clustering import cluster_data

from visualization import plot_comparison


def print_dataset_summary(dataset):
    """
    생성된 데이터의 기본 정보를 출력한다.
    """

    base_count = (
        dataset["data_type"] == "base"
    ).sum()

    outlier_count = (
        dataset["data_type"] == "outlier"
    ).sum()

    print("===== 데이터 생성 결과 =====")
    print(f"기본 오답 데이터 수: {base_count}")
    print(f"추가 이상치 수: {outlier_count}")
    print(f"전체 데이터 수: {len(dataset)}")

    print("\n===== 데이터 미리 보기 =====")
    print(dataset.head())


def print_kmeans_result(results):
    """
    K-Means 실행 결과를 출력한다.
    """

    kmeans_model = results["kmeans_model"]

    print("\n===== K-Means 실행 결과 =====")

    print(
        "생성된 군집 수:",
        results["kmeans_cluster_count"]
    )

    print(
        "중심점 갱신 반복 횟수:",
        kmeans_model.n_iter_
    )

    print(
        "군집 내 거리 제곱합:",
        round(kmeans_model.inertia_, 4)
    )

    silhouette = results["kmeans_silhouette"]

    if silhouette is not None:
        print(
            "실루엣 계수:",
            round(silhouette, 4)
        )
    else:
        print(
            "실루엣 계수:",
            "계산할 수 없음"
        )

    print("\nK-Means 중심점")

    for cluster_number, centroid in enumerate(
        results["kmeans_centroids"],
        start=1
    ):
        difficulty = centroid[0]
        algorithm_feature = centroid[1]

        print(
            f"Cluster {cluster_number}: "
            f"난이도 지수 {difficulty:.2f}, "
            f"알고리즘 특성 지수 {algorithm_feature:.2f}"
        )


def print_dbscan_result(results):
    """
    DBSCAN 실행 결과를 출력한다.
    """

    print("\n===== DBSCAN 실행 결과 =====")

    print(
        "생성된 군집 수:",
        results["dbscan_cluster_count"]
    )

    print(
        "노이즈로 분류된 데이터 수:",
        results["dbscan_noise_count"]
    )

    silhouette = results["dbscan_silhouette"]

    if silhouette is not None:
        print(
            "노이즈 제외 실루엣 계수:",
            round(silhouette, 4)
        )
    else:
        print(
            "노이즈 제외 실루엣 계수:",
            "계산할 수 없음"
        )


def add_cluster_results_to_dataset(
    dataset,
    results
):
    """
    K-Means와 DBSCAN의 군집화 결과를
    기존 데이터프레임에 새로운 열로 추가한다.
    """

    result_data = dataset.copy()

    result_data["kmeans_cluster"] = (
        results["kmeans_labels"]
    )

    result_data["dbscan_cluster"] = (
        results["dbscan_labels"]
    )

    return result_data


def main():
    # ----------------------------------------
    # 1. 가상 코딩 테스트 오답 데이터 생성
    # ----------------------------------------
    dataset = create_dataset(
        n_samples=50,
        random_state=42,
        include_outliers=True
    )

    print_dataset_summary(dataset)

    # 군집화에 사용할 데이터 형태 확인
    feature_data = get_feature_data(dataset)

    print("\n군집화 입력 데이터 형태:", feature_data.shape)

    # ----------------------------------------
    # 2. K-Means와 DBSCAN 실행
    # ----------------------------------------
    results = cluster_data(
        data=dataset,
        n_clusters=3,
        eps=0.45,
        min_samples=4,
        random_state=42
    )

    # ----------------------------------------
    # 3. 실행 결과 출력
    # ----------------------------------------
    print_kmeans_result(results)
    print_dbscan_result(results)

    # ----------------------------------------
    # 4. 군집화 결과를 데이터에 추가
    # ----------------------------------------
    result_dataset = add_cluster_results_to_dataset(
        dataset,
        results
    )

    print("\n===== 군집화 결과 미리 보기 =====")
    print(
        result_dataset[
            [
                "sample_id",
                "difficulty_index",
                "algorithm_feature_index",
                "data_type",
                "kmeans_cluster",
                "dbscan_cluster"
            ]
        ].head(10)
    )

    # ----------------------------------------
    # 5. 데이터와 결과를 CSV로 저장
    # ----------------------------------------
    save_dataset(
        dataset,
        "output/error_data.csv"
    )

    save_dataset(
        result_dataset,
        "output/clustering_result.csv"
    )

    print("\n데이터 저장 완료")
    print("- output/error_data.csv")
    print("- output/clustering_result.csv")

    # ----------------------------------------
    # 6. 비교 그래프 생성
    # ----------------------------------------
    plot_comparison(
        data=dataset,
        results=results,
        save_path="output/clustering_result.png",
        show_graph=True
    )


if __name__ == "__main__":
    main()