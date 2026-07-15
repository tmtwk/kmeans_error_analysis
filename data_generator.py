import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


# 군집화에 실제로 사용할 열
FEATURE_COLUMNS = [
    "difficulty_index",
    "algorithm_feature_index"
]


def generate_base_data(n_samples=50, random_state=42):
    """
    세 가지 약점 유형을 가진 가상의 코딩 테스트 오답 데이터 생성
    """

    # 가상 오답 유형의 중심점
    centers = [
        [-4, -3],  # 저난도 · 구현/탐색형
        [0, 4],    # 중간난도 · DP/수학형
        [5, -1]    # 고난도 · 구현/탐색형
    ]

    # 각 군집이 퍼지는 정도
    cluster_std = [0.8, 1.0, 0.9]

    # make_blobs를 이용해 정규분포 형태의 데이터 생성
    raw_data, true_labels = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    # 두 특성을 각각 1~100 범위로 변환
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaled_data = scaler.fit_transform(raw_data)

    # 데이터프레임 생성
    data = pd.DataFrame(
        scaled_data,
        columns=FEATURE_COLUMNS
    )

    # 데이터 식별 번호
    data.insert(
        0,
        "sample_id",
        [f"E{i:03d}" for i in range(1, n_samples + 1)]
    )

    # 기본 데이터임을 표시
    data["data_type"] = "base"

    # 데이터 생성 당시의 군집 번호
    # K-Means 학습에는 사용하지 않고 결과 검증에만 사용
    data["true_cluster"] = true_labels

    # 소수점 둘째 자리까지 정리
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].round(2)

    return data


def add_outliers(data):
    """
    K-Means와 DBSCAN 비교를 위한 이상치 추가
    """

    outlier_points = np.array([
        [5, 95],
        [95, 95],
        [50, 5],
        [60, 55],
        [30, 50]
    ], dtype=float)

    outlier_data = pd.DataFrame(
        outlier_points,
        columns=FEATURE_COLUMNS
    )

    start_id = len(data) + 1

    outlier_data.insert(
        0,
        "sample_id",
        [
            f"E{i:03d}"
            for i in range(start_id, start_id + len(outlier_data))
        ]
    )

    outlier_data["data_type"] = "outlier"

    # 이상치는 기존 군집에 속하지 않으므로 -1로 표시
    outlier_data["true_cluster"] = -1

    # 기본 데이터와 이상치 결합
    combined_data = pd.concat(
        [data, outlier_data],
        ignore_index=True
    )

    return combined_data


def create_dataset(
    n_samples=50,
    random_state=42,
    include_outliers=True
):
    """
    기본 데이터 생성과 이상치 추가를 한 번에 실행
    """

    data = generate_base_data(
        n_samples=n_samples,
        random_state=random_state
    )

    if include_outliers:
        data = add_outliers(data)

    return data


def get_feature_data(data):
    """
    K-Means와 DBSCAN에 입력할 두 특성만 반환
    """

    return data[FEATURE_COLUMNS].to_numpy()


def save_dataset(data, file_path="output/error_data.csv"):
    """
    생성된 데이터를 CSV 파일로 저장
    """

    import os

    folder_path = os.path.dirname(file_path)

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    data.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )


def main():
    # 기본 데이터 50개와 이상치 5개 생성
    dataset = create_dataset(
        n_samples=50,
        random_state=42,
        include_outliers=True
    )

    # 데이터 확인
    print("===== 생성된 데이터 미리 보기 =====")
    print(dataset.head())

    print("\n===== 데이터 개수 =====")
    print("기본 데이터:", (dataset["data_type"] == "base").sum())
    print("이상치:", (dataset["data_type"] == "outlier").sum())
    print("전체 데이터:", len(dataset))

    # CSV 저장
    save_dataset(
        dataset,
        "output/error_data.csv"
    )

    print("\n데이터가 output/error_data.csv에 저장되었습니다.")


if __name__ == "__main__":
    main()