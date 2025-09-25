import os
import shutil
from pathlib import Path
import pandas as pd
import random
import json # ✨ [추가됨] JSON 라이브러리 import

# --------------------------------------------------
# ✅ 설정 부분 (경로 및 옵션)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DATASET_PATH = BASE_DIR.parent / "images" 
NEW_DATASET_PATH = BASE_DIR / "dataset"
CSV_FILE_PATH = BASE_DIR / "종별이미지개수.csv"
EXCLUDE_JSON_PATH = BASE_DIR / "exclude_files.json"

NUM_TEST_IMAGES_PER_CLASS = 5
TRAIN_RATIO = 0.8
NUM_TOP_CLASSES = 10

def load_exclude_list(json_path):
    """JSON 파일에서 제외할 파일 목록을 읽어옵니다."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'exclude' in data and isinstance(data['exclude'], list):
            print(f"✅ '{json_path.name}' 파일에서 {len(data['exclude'])}개의 제외 목록을 불러왔습니다.")
            return data['exclude']
        else:
            print(f"⚠️ 경고: JSON 파일에 'exclude' 키가 없거나 리스트 형식이 아닙니다.")
            return []
    except FileNotFoundError:
        print(f"⚠️ 경고: 제외 목록 파일('{json_path.name}')을 찾을 수 없습니다. 빈 목록으로 진행합니다.")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ 경고: '{json_path.name}' 파일의 형식이 올바르지 않습니다.")
        return []

# --------------------------------------------------

# ✨ [수정됨] exclude_list 인자를 받도록 함수 시그니처 수정
def split_and_overwrite_top_classes(original_path, new_path, num_test, train_ratio, csv_path, num_top, exclude_list):
    """
    지정된 상위 N개 클래스에 대해서만 데이터셋을 새로 분할하고,
    기존 train/val/test 폴더에 있던 해당 클래스 폴더를 덮어쓰는 함수
    """
    exclude_set = set(exclude_list)

    try:
        df = pd.read_csv(csv_path)
        top_folders = df.nlargest(num_top, 'file_count')['folder'].tolist()
        print(f"✅ CSV 파일에서 덮어쓸 상위 {num_top}개 폴더를 성공적으로 불러왔습니다.")
        print(" - 대상 폴더:", top_folders)
    except Exception as e:
        print(f"❌ CSV 파일 처리 중 오류가 발생했습니다: {e}")
        return

    original_path = Path(original_path)
    new_path = Path(new_path)
    train_path = new_path / 'train'
    val_path = new_path / 'val'
    test_path = new_path / 'test'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for class_name in top_folders:
        print(f"\n▶ '{class_name}' 클래스 처리 시작...")
        class_dir = original_path / class_name
        
        if not class_dir.is_dir():
            print(f"  - ⚠️ 경고: 원본 폴더 '{class_dir}'를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        target_folders = [
            train_path / class_name,
            val_path / class_name,
            test_path / class_name
        ]
        
        for folder in target_folders:
            if folder.exists():
                shutil.rmtree(folder)
        print(f"  - 기존 '{class_name}' 폴더를 삭제했습니다.")

        for folder in target_folders:
            folder.mkdir()
        
        # ✨ [수정됨] 제외 목록에 있는 파일 필터링 로직 추가
        all_images_in_folder = list(class_dir.glob('*.*'))
        images = [img for img in all_images_in_folder if img.name not in exclude_set]
        
        excluded_count = len(all_images_in_folder) - len(images)
        if excluded_count > 0:
            print(f"  - 제외 목록에 따라 {excluded_count}개의 파일을 건너뜁니다.")
        
        random.shuffle(images)

        if len(images) < num_test:
            print(f"  - ⚠️ 경고: (제외 파일 제외 후) 분할할 이미지가 테스트 개수({num_test}개)보다 적습니다.")

        test_images = images[:num_test]
        remaining_images = images[num_test:]
        
        split_point = int(len(remaining_images) * train_ratio)
        train_images = remaining_images[:split_point]
        val_images = remaining_images[split_point:]

        for img in train_images:
            shutil.copy(img, train_path / class_name)
        for img in val_images:
            shutil.copy(img, val_path / class_name)
        for img in test_images:
            shutil.copy(img, test_path / class_name)

        print(f"  - ✅ 복사 완료: Train({len(train_images)}), Val({len(val_images)}), Test({len(test_images)})")


if __name__ == '__main__':
    # ✨ [수정됨] 스크립트 시작 시 JSON 파일에서 목록을 먼저 불러옴
    exclude_file_list = load_exclude_list(EXCLUDE_JSON_PATH)
    
    # ✨ [수정됨] 불러온 목록을 함수의 마지막 인자로 전달
    split_and_overwrite_top_classes(
        ORIGINAL_DATASET_PATH,
        NEW_DATASET_PATH,
        NUM_TEST_IMAGES_PER_CLASS,
        TRAIN_RATIO,
        CSV_FILE_PATH,
        NUM_TOP_CLASSES,
        exclude_file_list 
    )
    print("\n✅ 모든 작업이 완료되었습니다.")