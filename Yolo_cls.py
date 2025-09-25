from ultralytics import YOLO
from pathlib import Path # 파일 경로를 더 안전하고 쉽게 다루기 위해 사용합니다.
import torch
import os

# --------------------------------------------------
# ✅ 사용자가 수정해야 할 부분
# --------------------------------------------------
# 1. 학습시킬 데이터셋의 경로를 지정하세요.
DATASET_PATH = 'C:/Users/sega0/Desktop/code/wild/dataset'

# 2. 결과 폴더의 기본 이름을 지정하세요. (예: 'test' -> test1, test2, ...)
EXPERIMENT_BASE_NAME = 'test'

def get_next_experiment_name(base_name, project_dir='runs/classify'):
    """'runs/classify' 폴더를 확인하여 다음 실험 번호를 찾아 이름을 반환합니다."""
    i = 1
    while True:
        exp_name = f"{base_name}{i}"
        if not os.path.exists(os.path.join(project_dir, exp_name)):
            return exp_name
        i += 1



if __name__ == '__main__':
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("경고: CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다. (속도가 매우 느릴 수 있습니다)")

    # ==================================================
    # 1단계: 모델 학습 (Train & Val)
    # ==================================================
    print("--- 1단계: 모델 학습을 시작합니다 ---")

    # 사전 학습된 YOLOv8n 분류 모델을 로드
    model = YOLO('yolov8n-cls.pt')

    # 다음 실행할 실험 이름 가져오기
    next_experiment_name = get_next_experiment_name(EXPERIMENT_BASE_NAME)
    print(f"이번 학습 결과는 'runs/classify/{next_experiment_name}' 폴더에 저장됩니다.")

    # 데이터셋으로 모델 학습 (train/val이 함께 진행됨)
    # 학습 결과는 'runs/classify/...' 폴더에 저장됩니다.
    results = model.train(
        data=DATASET_PATH,
        epochs=50,  # 테스트를 위해 에포크를 줄여도 좋습니다 (예: 3)
        imgsz=224,
        name=next_experiment_name,  # 실행 결과 폴더 이름
        verbose=False

    )

    print("\n--- 학습 완료! ---")