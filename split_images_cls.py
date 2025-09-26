import os
import shutil
from pathlib import Path
import pandas as pd
import random
import json

# --------------------------------------------------
# ✅ 설정 부분 (경로 및 옵션)
# --------------------------------------------------
# 현재 .py 스크립트 파일이 있는 폴더를 기준 경로로 설정
BASE_DIR = Path(__file__).resolve().parent

# 원본 이미지 폴더, 새로 만들 데이터셋 폴더, CSV 파일, JSON 파일 경로 설정
ORIGINAL_DATASET_PATH = BASE_DIR.parent / "images" 
NEW_DATASET_PATH = BASE_DIR / "dataset"
CSV_FILE_PATH = BASE_DIR / "종별이미지개수.csv"
# ✨ 'test' 폴더에 고정할 이미지 목록이 담긴 JSON 파일
PREDEFINED_TEST_JSON_PATH = BASE_DIR / "exclude_files.json" 

# Train / Validation 분할 비율 (나머지 이미지에 대해 적용)
TRAIN_RATIO = 0.8 
# 처리할 상위 클래스 개수
NUM_TOP_CLASSES = 10

def load_predefined_test_files(json_path):
    """JSON 파일에서 'test' 폴더에 고정할 파일 목록을 읽어옵니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 'exclude' 키를 그대로 사용하되, 이제부터 이 목록은 'test'용으로 사용됩니다.
        if 'exclude' in data and isinstance(data['exclude'], list):
            print(f"✅ '{json_path.name}' 파일에서 {len(data['exclude'])}개의 고정 Test 이미지 목록을 불러왔습니다.")
            return data['exclude']
        else:
            print(f"⚠️ 경고: JSON 파일에 'exclude' 키가 없거나 리스트 형식이 아닙니다.")
            return []
    except FileNotFoundError:
        print(f"⚠️ 경고: 고정 Test 목록 파일('{json_path.name}')을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ 경고: '{json_path.name}' 파일의 형식이 올바르지 않습니다.")
        return []

# --------------------------------------------------

def split_data_with_predefined_test(original_path, new_path, train_ratio, csv_path, num_top, predefined_test_files):
    """
    JSON 파일 목록을 Test 셋으로 먼저 할당하고,
    나머지 이미지들을 Train/Val로 분할하는 함수
    """
    # 빠른 조회를 위해 list를 set으로 변환
    test_files_set = set(predefined_test_files)

    try:
        df = pd.read_csv(csv_path)
        top_folders = df.nlargest(num_top, 'file_count')['folder'].tolist()
        print(f"✅ CSV 파일에서 처리할 상위 {num_top}개 폴더를 성공적으로 불러왔습니다.")
        print(" - 대상 폴더:", top_folders)
    except Exception as e:
        print(f"❌ CSV 파일 처리 중 오류가 발생했습니다: {e}")
        return

    original_path = Path(original_path)
    new_path = Path(new_path)
    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'

    # train/val/test 기본 폴더 생성
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_name in top_folders:
        print(f"\n▶ '{class_name}' 클래스 처리 시작...")
        class_dir = original_path / class_name
        
        if not class_dir.is_dir():
            print(f" - ⚠️ 경고: 원본 폴더 '{class_dir}'를 찾을 수 없어 건너뜁니다.")
            continue
        
        # 덮어쓰기를 위해 기존 폴더 삭제 후 다시 생성
        target_folders = [train_path / class_name, val_path / class_name, test_path / class_name]
        for folder in target_folders:
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()

        # ✨ 핵심 로직: 이미지를 '고정 test' 그룹과 '나머지' 그룹으로 분리
        all_images = list(class_dir.glob('*.*'))
        
        predefined_test_images = []
        remaining_images = []

        for img in all_images:
            if img.name in test_files_set:
                predefined_test_images.append(img)
            else:
                remaining_images.append(img)
        
        # 1. 고정 test 이미지들을 test 폴더로 복사
        for img in predefined_test_images:
            shutil.copy(img, test_path / class_name)
        
        # 2. 나머지 이미지들을 train/val로 분할
        random.shuffle(remaining_images)
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        for img in val_images:
            shutil.copy(img, val_path / class_name)

        print(f" - ✅ 복사 완료: Train({len(train_images)}), Val({len(val_images)}), Test({len(predefined_test_images)})")

if __name__ == '__main__':
    # 1. JSON 파일에서 고정할 test 파일 목록을 불러오기
    predefined_test_file_list = load_predefined_test_files(PREDEFINED_TEST_JSON_PATH)
    
    # 2. 메인 함수 실행
    split_data_with_predefined_test(
        ORIGINAL_DATASET_PATH,
        NEW_DATASET_PATH,
        TRAIN_RATIO,
        CSV_FILE_PATH,
        NUM_TOP_CLASSES,
        predefined_test_file_list
    )
    print("\n✅ 모든 작업이 완료되었습니다.")