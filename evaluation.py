import os
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont 
import pandas as pd

# --------------------------------------------------
# ✅ 사용자가 수정해야 할 부분
# --------------------------------------------------
MODEL_PATH = r"C:\Users\sega0\Desktop\folder\code\wild\runs\classify\test11\weights\best.pt"
TEST_DATASET_PATH = r"C:\Users\sega0\Desktop\folder\code\wild\dataset\test"
RESULTS_SAVE_PATH = Path(MODEL_PATH).parent.parent
VISUALIZED_IMAGES_SAVE_DIR = RESULTS_SAVE_PATH / "visualized_predictions"
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
# --------------------------------------------------

def main():
    model_path = Path(MODEL_PATH)
    test_path = Path(TEST_DATASET_PATH)

    try:
        VISUALIZED_IMAGES_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✨ 시각화된 이미지를 저장할 폴더가 생성 또는 확인되었습니다:\n  -> {VISUALIZED_IMAGES_SAVE_DIR}")
    except Exception as e:
        print(f"❌ 오류: 이미지 저장 폴더 생성에 실패했습니다. 경로를 확인하거나 다른 경로를 지정해 주세요: {e}")
        return

    if not model_path.exists() or not test_path.exists():
        print(f"❌ 오류: 모델 또는 테스트 데이터셋 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    
    print(f"모델을 로드합니다: {model_path.name}")
    model = YOLO(model_path)
    print("✅ 모델 로드 완료.")

    total_images = 0
    total_correct = 0
    results_data = []

    try:
        font = ImageFont.truetype(FONT_PATH, 20) 
        small_font = ImageFont.truetype(FONT_PATH, 16)
    except IOError:
        print(f"⚠️ 경고: 폰트를 로드할 수 없습니다. 기본 폰트가 사용됩니다: {FONT_PATH}")
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    class_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        true_label = class_dir.name
        print(f"\n{'='*50}\n📂 클래스 '{true_label}'의 이미지 예측을 시작합니다.\n{'='*50}")

        class_total = 0
        class_correct = 0

        for image_path in class_dir.glob('*.*'):
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            try:
                results = model.predict(image_path, verbose=False)
                result = results[0]
                
                # ✨ 수정된 부분: topk() 대신 top5와 top5conf 속성 사용
                pred_label = model.names[result.probs.top1]
                pred_confidence = result.probs.top1conf.item()

                is_correct = (true_label == pred_label)
                if is_correct:
                    class_correct += 1
                    total_correct += 1
                
                class_total += 1
                total_images += 1
                
                print(f"  - 파일: {image_path.name} | 예측: '{pred_label}' | 신뢰도: {pred_confidence*100:.2f}% | 결과: {'✅ 정답' if is_correct else '❌ 오답'}")

                try:
                    img = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    # ✨ 수정된 부분: topk() 대신 top5와 top5conf 속성을 결합하여 사용
                    top5_indices = result.probs.top5
                    top5_confs = result.probs.top5conf
                    
                    text_lines = [f"{model.names[idx]} {conf.item():.2f}" for idx, conf in zip(top5_indices, top5_confs)]
                    
                    text_height_per_line = small_font.getbbox("Tg")[3] - small_font.getbbox("Tg")[1]
                    total_text_height = len(text_lines) * (text_height_per_line + 2)
                    
                    box_start_x, box_start_y = 10, 10
                    box_end_x = box_start_x + 250
                    box_end_y = box_start_y + total_text_height + 10
                    
                    draw.rectangle([box_start_x, box_start_y, box_end_x, box_end_y], fill=(0, 0, 0, 128))

                    y_offset = box_start_y + 5
                    for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
                        text_to_draw = f"{model.names[idx]} {conf.item():.2f}"
                        text_color = "red" if model.names[idx] == true_label else "white"
                        draw.text((box_start_x + 5, y_offset), text_to_draw, font=small_font, fill=text_color)
                        y_offset += (text_height_per_line + 2)

                    save_path = VISUALIZED_IMAGES_SAVE_DIR / f"{true_label}_{image_path.name}"
                    print(f"    - 이미지를 다음 경로에 저장합니다:\n      -> {save_path}")
                    img.save(save_path)
                except Exception as img_e:
                    print(f"    - 파일: {image_path.name} | ⚠️ 이미지 시각화/저장 중 오류 발생: {img_e}")
                    
            except Exception as e:
                print(f"  - 파일: {image_path.name} | ⚠️ 예측 중 오류 발생: {e}")

        class_accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"\n👉 '{true_label}' 클래스 예측 완료: {class_total}개 중 {class_correct}개 정답 (정확도: {class_accuracy:.2f}%)")
        results_data.append([true_label, class_total, class_correct, class_accuracy])

    print(f"\n\n{'='*60}\n🏆 최종 예측 결과 요약\n{'='*60}")
    df = pd.DataFrame(results_data, columns=['클래스', '총 이미지 수', '정답 수', '정확도 (%)'])
    df = df.sort_values(by='정확도 (%)', ascending=False)
    print(df.to_string(index=False))

    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"\n\n📊 전체 정확도: {total_images}개 중 {total_correct}개 정답 ({overall_accuracy:.2f}%)")

    csv_save_path = RESULTS_SAVE_PATH / 'prediction_summary.csv'
    try:
        df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        print(f"\n\n💾 결과가 CSV 파일로 성공적으로 저장되었습니다.")
        print(f"  -> 저장 위치: {csv_save_path}")
    except Exception as e:
        print(f"\n\n❌ CSV 파일 저장 중 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    main()