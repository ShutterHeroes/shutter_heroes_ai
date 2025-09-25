import os
import random
import shutil
from pathlib import Path

# --------------------------------------------------
# ✅ 사용자가 수정해야 할 부분
# --------------------------------------------------
# 1. 분할할 원본 이미지가 담긴 폴더 경로 (클래스별 폴더 포함)
ORIGINAL_DATASET_PATH = r"C:\Users\sega0\Desktop\code\wild\original"

# 2. 분할된 train/val/test 데이터셋을 저장할 새로운 폴더 경로
NEW_DATASET_PATH = r"C:\Users\sega0\Desktop\code\wild\dataset" # 이 폴더는 비어있어야 합니다.

# 3. Test 세트로 분리할 이미지 개수 (각 클래스별)
NUM_TEST_IMAGES_PER_CLASS = 5

# 4. (Test 이미지를 제외한 나머지에서) Train/Val 분할 비율
TRAIN_RATIO = 0.8
# --------------------------------------------------

def split_dataset_with_test(original_path, new_path, num_test, train_ratio):
    """
    데이터셋을 train/val/test 폴더로 분할하여 복사하는 함수
    """
    # 경로 객체 생성
    original_path = Path(original_path)
    new_path = Path(new_path)
    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'

    # 기존 폴더가 있다면 삭제 (주의!)
    if new_path.exists():
        shutil.rmtree(new_path)
    
    # train, val, test 폴더 생성
    train_path.mkdir(parents=True)
    val_path.mkdir(parents=True)
    test_path.mkdir(parents=True)
    print(f"'{new_path}' 폴더를 생성하고 train/val/test 하위 폴더를 만들었습니다.")

    # 원본 데이터셋의 클래스 폴더 목록 가져오기
    class_dirs = [d for d in original_path.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n'{class_name}' 클래스 처리 중...")

        # 새로운 경로들에 클래스 폴더 생성
        (train_path / class_name).mkdir()
        (val_path / class_name).mkdir()
        (test_path / class_name).mkdir()

        # 현재 클래스의 모든 이미지 파일 목록 가져오기
        images = list(class_dir.glob('*.*'))
        random.shuffle(images) # 데이터를 무작위로 섞기 (중요!)

        # --- 1단계: Test 데이터 분리 ---
        if len(images) < num_test:
            print(f"  - 경고: '{class_name}' 클래스의 전체 이미지({len(images)}개)가 테스트 개수({num_test}개)보다 적습니다.")
            continue
        
        test_images = images[:num_test]
        remaining_images = images[num_test:] # 테스트용을 제외한 나머지 이미지들

        for img in test_images:
            shutil.copy(img, test_path / class_name)
        print(f"  - Test: {len(test_images)}개 이미지 복사 완료.")

        # --- 2단계: 남은 데이터로 Train/Val 분리 ---
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        
        for img in val_images:
            shutil.copy(img, val_path / class_name)
        
        print(f"  - Train: {len(train_images)}개, Val: {len(val_images)}개 이미지 복사 완료.")

if __name__ == '__main__':
    split_dataset_with_test(ORIGINAL_DATASET_PATH, NEW_DATASET_PATH, NUM_TEST_IMAGES_PER_CLASS, TRAIN_RATIO)
    print("\n✅ 모든 데이터셋 분할(train/val/test)이 완료되었습니다.")