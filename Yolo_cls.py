from ultralytics import YOLO
from pathlib import Path
import torch

# --------------------------------------------------
# ✅ 경로 설정 (수정 없음)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"
SAVE_PATH = BASE_DIR.parent / "runs" # '.../CODE/runs'를 가리킴

# --------------------------------------------------
# ✅ 옵션 설정
# --------------------------------------------------
EXPERIMENT_BASE_NAME = 'test5'
EARLY_STOPPING_PATIENCE = 10
# ✨ [추가] 실시간 데이터 증강 옵션
AUGMENTATION_OPTIONS = {
    'degrees': 15,      # 이미지 회전 각도 범위 (-15 ~ +15)
    'translate': 0.1,   # 이미지 이동 비율
    'scale': 0.1,       # 이미지 크기 조절 비율
    'fliplr': 0.5,      # 50% 확률로 좌우 반전
    'mosaic': 1.0,      # Mosaic 증강 사용 확률 (배경 합성을 통해 일반화 성능 향상)
    'mixup': 0.1        # Mixup 증강 사용 확률 (이미지 섞기)
}
# --------------------------------------------------



def get_next_experiment_name(project_dir: Path, base_name: str) -> str:
    """
    지정된 프로젝트 폴더를 확인하여 다음 실험 이름을 반환합니다.
    먼저 base_name 자체를 확인하고, 존재하면 숫자를 붙여나갑니다.
    """
    # 1. 기본 이름(base_name) 자체를 먼저 확인
    base_exp_path = project_dir / "classify" / base_name
    if not base_exp_path.exists():
        return base_name  # 기본 이름이 비어있으면 그대로 사용

    # 2. 기본 이름이 이미 존재하면, 뒤에 숫자를 붙여서 다음 번호 찾기
    i = 1
    while True:
        exp_name = f"{base_name}{i}"
        if not (project_dir / "classify" / exp_name).exists():
            return exp_name
        i += 1


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("경고: CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다.")

    print("--- 1단계: 모델 학습을 시작합니다 ---")

    model = YOLO('yolov8n-cls.pt')

    # ✨ [수정됨] 함수에 SAVE_PATH를 직접 전달
    next_experiment_name = get_next_experiment_name(SAVE_PATH, EXPERIMENT_BASE_NAME)
    print(f"이번 학습 결과는 '{SAVE_PATH}/classify/{next_experiment_name}' 폴더에 저장됩니다.")

    results = model.train(
        data=DATASET_PATH,
        epochs=75,
        imgsz=224,
        # ✨✨✨ [가장 중요] project 옵션 추가! ✨✨✨
        project=SAVE_PATH,
        # name에는 하위 폴더 이름만 지정
        name=f"classify/{next_experiment_name}",
        verbose=False,
        batch=32,
        patience=EARLY_STOPPING_PATIENCE,
        lr0=0.01,
        **AUGMENTATION_OPTIONS
        
    )

    print("\n--- 학습 완료! ---")