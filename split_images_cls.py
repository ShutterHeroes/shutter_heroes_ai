import shutil
from pathlib import Path
import pandas as pd
import random
import json

# --------------------------------------------------
# ✅ 설정 부분
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# 원본 이미지 폴더
ORIGINAL_DATASET_PATH = BASE_DIR.parent / "images"
# 기본 베이스가 될, 미리 분할된 train/val 폴더
FIXED_DATA_PATH = BASE_DIR.parent / "other"
# 최종 결과물이 저장될 폴더
NEW_DATASET_PATH = BASE_DIR / "dataset"

# 추가 처리 대상을 정할 CSV 파일
CSV_FILE_PATH = BASE_DIR / "종별이미지개수.csv"
# Test 데이터를 고정할 JSON 파일
PREDEFINED_TEST_JSON_PATH = BASE_DIR / "exclude_files.json"

# 각종 옵션
TRAIN_RATIO = 0.8
NUM_TOP_CLASSES = 10
RANDOM_SEED = 42

# --------------------------------------------------

def create_combined_dataset(original_path, fixed_path, new_path, csv_path, json_path, num_top, train_ratio, random_seed):
    random.seed(random_seed)
    
    # --- ✨ [수정된] 1단계: 'other' 폴더의 클래스들을 'dataset'으로 선별 복사 ---
    print("--- 1단계: 'other' 폴더의 클래스들을 'dataset'으로 복사 시작 ---")
    if new_path.exists():
        shutil.rmtree(new_path)
        print(f" - 기존 '{new_path.name}' 폴더 삭제 완료.")
        
    # 빈 dataset 폴더와 train, val, test 하위 폴더를 먼저 생성
    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)

    # 'other/train' 안의 클래스 폴더들을 'dataset/train'으로 하나씩 복사
    source_train = fixed_path / 'train'
    if source_train.is_dir():
        for class_folder in source_train.iterdir():
            if class_folder.is_dir(): # 폴더인 경우에만 복사
                print(f" - train: '{class_folder.name}' 클래스 복사 중...")
                shutil.copytree(class_folder, train_path / class_folder.name)
    
    # 'other/val' 안의 클래스 폴더들을 'dataset/val'으로 하나씩 복사
    source_val = fixed_path / 'val'
    if source_val.is_dir():
        for class_folder in source_val.iterdir():
            if class_folder.is_dir(): # 폴더인 경우에만 복사
                print(f" - val: '{class_folder.name}' 클래스 복사 중...")
                shutil.copytree(class_folder, val_path / class_folder.name)
                
    print("✅ 'other' 폴더 내 클래스 복사 완료.")

    # --- 2단계 & 3단계: CSV/JSON 정보로 Test 생성 및 Train/Val 추가 ---
    print("\n--- 2 & 3단계: CSV 목록 기준으로 데이터 추가 작업 시작 ---")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            test_files_set = set(json.load(f)['exclude'])
        print(f"✅ '{json_path.name}' 파일에서 {len(test_files_set)}개의 고정 Test 이미지 목록을 불러왔습니다.")
    except Exception as e:
        print(f"❌ Test 목록 파일('{json_path.name}') 처리 중 오류: {e}")
        return

    try:
        df = pd.read_csv(csv_path)
        top_folders = df.nlargest(num_top, 'file_count')['folder'].tolist()
        print(f"✅ CSV 파일에서 추가 처리할 상위 {num_top}개 폴더를 선정했습니다.")
        print(" - 대상 폴더:", top_folders)
    except Exception as e:
        print(f"❌ CSV 파일 처리 중 오류: {e}")
        return

    for class_name in top_folders:
        print(f"\n▶ '{class_name}' 클래스 추가 처리 중...")
        class_dir = original_path / class_name
        
        if not class_dir.is_dir():
            print(f" - ⚠️ 경고: 원본 폴더 '{class_dir}'를 찾을 수 없어 건너뜁니다.")
            continue

        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

        all_images = list(class_dir.glob('*.*'))
        predefined_test_images = [img for img in all_images if img.name in test_files_set]
        remaining_images = [img for img in all_images if img.name not in test_files_set]
        
        for img in predefined_test_images:
            shutil.copy(img, test_path / class_name)

        random.shuffle(remaining_images)
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        for img in val_images:
            shutil.copy(img, val_path / class_name)
        
        print(f" - ✅ 처리 완료: Test({len(predefined_test_images)}개), Train({len(train_images)}개), Val({len(val_images)}개) 이미지 추가.")

# --------------------------------------------------

if __name__ == '__main__':
    create_combined_dataset(
        ORIGINAL_DATASET_PATH,
        FIXED_DATA_PATH,
        NEW_DATASET_PATH,
        CSV_FILE_PATH,
        PREDEFINED_TEST_JSON_PATH,
        NUM_TOP_CLASSES,
        TRAIN_RATIO,
        RANDOM_SEED
    )
    print("\n✅ 모든 작업이 완료되었습니다.")