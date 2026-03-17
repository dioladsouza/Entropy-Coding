import os
import csv
import json
import time
import glob
import cabac_coder as cabbage
import golomb_rice_coder as gluten
import huffman_coder as hummus  # 新加入的 Huffman
from PIL import Image

# 1. 自动遍历 test_images 文件夹下的所有图片
test_images = glob.glob("test_images/*.png") + glob.glob("test_images/*.jpg")

# 2. 确保结果图片保存到 result_images 文件夹
output_dir = "result_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 如果没有图片，生成一个基础测试图
if not test_images:
    base_img = "test_images/test_image.png"
    if not os.path.exists("test_images"): os.makedirs("test_images")
    gluten.generate_test_image(base_img, size=256)
    test_images = [base_img]

results = []

def run_benchmark():
    for img_path in test_images:
        base_name = os.path.basename(img_path)
        print(f"\n" + "#"*60)
        print(f" PROCESSING: {base_name}")
        print("#"*60)
        
        # --- 运行 CABAC (cabbage) ---
        try:
            res_cabbage = cabbage.evaluate(img_path)
            results.append({
                "Image": base_name,
                "Algorithm": "CABAC",
                "CR": round(res_cabbage['compression_ratio'], 3),
                "Enc_Time_ms": round(res_cabbage['encode_time']*1000, 2),
                "Dec_Time_ms": round(res_cabbage['decode_time']*1000, 2),
                "Memory_KB": round(res_cabbage['encode_peak_kb'], 2),
                "Status": "PASS" if res_cabbage['lossless'] else "FAIL"
            })
        except Exception as e:
            print(f"CABAC failed: {e}")

        # --- 运行 Golomb-Rice (gluten) ---
        try:
            res_gluten = gluten.evaluate(img_path) 
            results.append({
                "Image": base_name,
                "Algorithm": "Golomb-Rice",
                "CR": round(res_gluten['compression_ratio'], 3),
                "Enc_Time_ms": round(res_gluten.get('encode_time', 0)*1000, 2),
                "Dec_Time_ms": round(res_gluten.get('decode_time', 0)*1000, 2),
                "Memory_KB": round(res_gluten.get('encode_peak_kb', 0), 2),
                "Status": "PASS"
            })
        except Exception as e:
            print(f"Golomb-Rice failed: {e}")

        # --- 运行 Huffman (hummus) ---
        try:
            res_hummus = hummus.evaluate(img_path)
            results.append({
                "Image": base_name,
                "Algorithm": "Huffman",
                "CR": round(res_hummus['compression_ratio'], 3),
                "Enc_Time_ms": round(res_hummus.get('encode_time', 0)*1000, 2),
                "Dec_Time_ms": round(res_hummus.get('decode_time', 0)*1000, 2),
                "Memory_KB": round(res_hummus.get('encode_peak_kb', 0), 2),
                "Status": "PASS" if res_hummus.get('lossless') else "FAIL"
            })
        except Exception as e:
            print(f"Huffman failed: {e}")

    # 3. 保存 CSV 文件
    keys = ["Image", "Algorithm", "CR", "Enc_Time_ms", "Dec_Time_ms", "Memory_KB", "Status"]
    with open('benchmark_results.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    # 4. 保存 JSON 文件
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n" + "="*60)
    print(f"BENCHMARK COMPLETE!")
    print(f"  - Results saved to: benchmark_results.csv & .json")
    print(f"  - Reconstructed images in: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()