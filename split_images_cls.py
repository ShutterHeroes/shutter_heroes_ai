import os
import shutil
from pathlib import Path
import pandas as pd
import random

# --------------------------------------------------
# ✅ 사용자가 수정해야 할 부분 (상대 경로 방식)
# --------------------------------------------------
# 1. 현재 .py 스크립트 파일이 있는 폴더를 기준 경로로 설정
BASE_DIR = Path(__file__).resolve().parent

# 2. 기준 경로를 바탕으로 나머지 경로들을 설정
ORIGINAL_DATASET_PATH = BASE_DIR.parent / "images" 
NEW_DATASET_PATH = BASE_DIR / "dataset"
CSV_FILE_PATH = BASE_DIR / "종별이미지개수.csv"

# 3. 나머지 설정값들은 그대로 유지
NUM_TEST_IMAGES_PER_CLASS = 5
TRAIN_RATIO = 0.8
NUM_TOP_CLASSES = 10
# --------------------------------------------------


def split_and_overwrite_top_classes(original_path, new_path, num_test, train_ratio, csv_path, num_top):
    """
    지정된 상위 N개 클래스에 대해서만 데이터셋을 새로 분할하고,
    기존 train/val/test 폴더에 있던 해당 클래스 폴더를 덮어쓰는 함수
    """
    # --- CSV 파일에서 상위 폴더 목록 가져오기 ---
    try:
        df = pd.read_csv(csv_path)
        top_folders = df.nlargest(num_top, 'file_count')['folder'].tolist()
        print(f"✅ CSV 파일에서 덮어쓸 상위 {num_top}개 폴더를 성공적으로 불러왔습니다.")
        print(" - 대상 폴더:", top_folders)
    except Exception as e:
        print(f"❌ CSV 파일 처리 중 오류가 발생했습니다: {e}")
        return

    # 경로 객체 생성
    original_path = Path(original_path)
    new_path = Path(new_path)
    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'

    # train/val/test 기본 폴더가 없으면 생성
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # --- CSV에서 불러온 상위 클래스들만 처리 ---
    for class_name in top_folders:
        class_dir = original_path / class_name
        
        if not class_dir.is_dir():
            print(f"\n⚠️ 경고: 원본 폴더 '{original_path}'에서 '{class_name}' 클래스를 찾을 수 없어 건너뜁니다.")
            continue
            
        print(f"\n'{class_name}' 클래스 처리 중...")

        # --- 1단계: 기존 폴더 삭제 (덮어쓰기 준비) ---
        target_folders = [
            train_path / class_name,
            val_path / class_name,
            test_path / class_name
        ]
        
        for folder in target_folders:
            if folder.exists():
                shutil.rmtree(folder)
        print(f"  - 기존 '{class_name}' 폴더를 삭제했습니다.")

        # --- 2단계: 새 폴더 생성 및 데이터 분할/복사 ---
        for folder in target_folders:
            folder.mkdir()
        
        images = list(class_dir.glob('*.*'))
        random.shuffle(images)

        if len(images) < num_test:
            print(f"  - ⚠️ 경고: 전체 이미지({len(images)}개)가 테스트 개수({num_test}개)보다 적습니다.")
            continue

        # Test 데이터 분리
        test_images = images[:num_test]
        remaining_images = images[num_test:]
        for img in test_images:
            shutil.copy(img, test_path / class_name)
        
        # Train/Val 데이터 분리
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        for img in val_images:
            shutil.copy(img, val_path / class_name)

        print(f"  - Test: {len(test_images)}개, Train: {len(train_images)}개, Val: {len(val_images)}개 이미지 복사 완료.")


if __name__ == '__main__':
    split_and_overwrite_top_classes(
        ORIGINAL_DATASET_PATH,
        NEW_DATASET_PATH,
        NUM_TEST_IMAGES_PER_CLASS,
        TRAIN_RATIO,
        CSV_FILE_PATH,
        NUM_TOP_CLASSES
    )
    print("\n✅ 지정된 모든 클래스의 분할 및 덮어쓰기가 완료되었습니다.")