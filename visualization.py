import os

import matplotlib.pyplot as plt
import numpy as np

from data_generator import (
    FEATURE_COLUMNS,
    create_dataset
)

from clustering import cluster_data


def plot_kmeans_result(
    ax,
    feature_data,
    labels,
    centroids
):
    """
    K-Means 군집화 결과를 지정된 그래프 영역에 출력한다.
    """

    unique_labels = sorted(np.unique(labels))

    # 군집별 데이터 출력
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id

        ax.scatter(
            feature_data[cluster_mask, 0],
            feature_data[cluster_mask, 1],
            s=60,
            alpha=0.8,
            label=f"Cluster {cluster_id + 1}"
        )

    # 군집 중심점 출력
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Centroid"
    )

    ax.set_title(
        "K-Means Clustering",
        fontsize=14
    )

    ax.set_xlabel(
        "Difficulty Index (1-100)",
        fontsize=11
    )

    ax.set_ylabel(
        "Algorithm Feature Index (1-100)",
        fontsize=11
    )

    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)

    ax.grid(
        alpha=0.25
    )

    ax.legend()


def plot_dbscan_result(
    ax,
    feature_data,
    labels
):
    """
    DBSCAN 군집화 결과를 지정된 그래프 영역에 출력한다.

    DBSCAN에서 -1은 어느 군집에도 포함되지 않은
    노이즈 데이터를 의미한다.
    """

    unique_labels = sorted(np.unique(labels))

    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id

        # 노이즈 데이터
        if cluster_id == -1:
            ax.scatter(
                feature_data[cluster_mask, 0],
                feature_data[cluster_mask, 1],
                s=90,
                marker="x",
                linewidths=2,
                label="Noise"
            )

        # 정상 군집 데이터
        else:
            ax.scatter(
                feature_data[cluster_mask, 0],
                feature_data[cluster_mask, 1],
                s=60,
                alpha=0.8,
                label=f"Cluster {cluster_id + 1}"
            )

    ax.set_title(
        "DBSCAN Clustering",
        fontsize=14
    )

    ax.set_xlabel(
        "Difficulty Index (1-100)",
        fontsize=11
    )

    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)

    ax.grid(
        alpha=0.25
    )

    ax.legend()


def plot_comparison(
    data,
    results,
    save_path="output/clustering_result.png",
    show_graph=True
):
    """
    K-Means와 DBSCAN 결과를 나란히 비교하는 그래프를 생성한다.

    Parameters
    ----------
    data:
        data_generator.py에서 생성한 데이터프레임

    results:
        clustering.py의 cluster_data 함수가 반환한 결과

    save_path:
        그래프를 저장할 위치

    show_graph:
        그래프 창을 화면에 표시할지 여부
    """

    # 그래프에 표시할 원래 1~100 범위의 특성 데이터
    feature_data = data[
        FEATURE_COLUMNS
    ].to_numpy()

    # 1행 2열의 비교 그래프 생성
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True
    )

    # 왼쪽 K-Means 그래프
    plot_kmeans_result(
        ax=axes[0],
        feature_data=feature_data,
        labels=results["kmeans_labels"],
        centroids=results["kmeans_centroids"]
    )

    # 오른쪽 DBSCAN 그래프
    plot_dbscan_result(
        ax=axes[1],
        feature_data=feature_data,
        labels=results["dbscan_labels"]
    )

    # 전체 그래프 제목
    figure.suptitle(
        "Clustering Analysis of Coding-Test Error Data",
        fontsize=16
    )

    # 그래프 요소가 겹치지 않도록 조정
    plt.tight_layout()

    # 저장 폴더 생성
    folder_path = os.path.dirname(save_path)

    if folder_path:
        os.makedirs(
            folder_path,
            exist_ok=True
        )

    # 보고서 삽입용 고화질 이미지 저장
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"그래프가 {save_path}에 저장되었습니다."
    )

    if show_graph:
        plt.show()
    else:
        plt.close()


def main():
    # 1. 기본 오답 데이터 50개와 이상치 5개 생성
    dataset = create_dataset(
        n_samples=50,
        random_state=42,
        include_outliers=True
    )

    # 2. K-Means와 DBSCAN 군집화 실행
    results = cluster_data(
        data=dataset,
        n_clusters=3,
        eps=0.45,
        min_samples=4,
        random_state=42
    )

    # 3. 비교 그래프 생성 및 저장
    plot_comparison(
        data=dataset,
        results=results,
        save_path="output/clustering_result.png",
        show_graph=True
    )


if __name__ == "__main__":
    main()