import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm  #用于显示进度条
from datasets import load_from_disk, load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
# 模型名称 (QwenGuard-v1.2-7B)
MODEL_ID = "/HDD0/zhangyuheng/safety_analysis/QwenGuard-v1.2-7B" 

# 数据集配置
DATASETS = {
    "mm_safetybench": {
        "root": "/HDD0/zhangyuheng/safety_analysis/mm_safetybench",
        "output": "mm_safetybench_qwenguard_results",
        "image_save_dir": "/HDD0/zhangyuheng/safety_analysis/mm_safetybench/saved_images",
        "subdir_filter": "SD_TYPO"
    },
    "beaver_tails": {
        "root": "/HDD0/zhangyuheng/safety_analysis/beaver_tails_data",
        "output": "qwenguard_results",
        "image_save_dir": "/HDD0/zhangyuheng/safety_analysis/beaver_tails_data/saved_images",
        "subdir_filter": None
    }
}

# 默认数据集
default_dataset = "mm_safetybench"

# 内存管理配置
SAVE_INTERVAL = 10  # 每处理10个样本保存一次
CLEAR_CACHE_INTERVAL = 5  # 每处理5个样本清空一次缓存
# ===========================================

def run_inference(dataset_name):
    # 1. 验证数据集名称
    if dataset_name not in DATASETS:
        print(f"错误: 数据集 {dataset_name} 不存在。可用数据集: {list(DATASETS.keys())}")
        return
    
    # 2. 获取数据集配置
    dataset_config = DATASETS[dataset_name]
    DATASET_ROOT = dataset_config["root"]
    OUTPUT_ROOT = dataset_config["output"]
    IMAGE_SAVE_DIR = dataset_config["image_save_dir"]
    SUBDIR_FILTER = dataset_config["subdir_filter"]
    
    print(f"\n使用数据集: {dataset_name}")
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"结果保存路径: {OUTPUT_ROOT}")
    print(f"图片保存路径: {IMAGE_SAVE_DIR}")
    if SUBDIR_FILTER:
        print(f"子目录过滤: {SUBDIR_FILTER}")
    
    # 3. 设置内存管理
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 4. 创建图片保存目录
    if not os.path.exists(IMAGE_SAVE_DIR):
        os.makedirs(IMAGE_SAVE_DIR)

    # 5. 创建结果保存根目录
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # 6. 加载模型和处理器
    print(f"\n正在加载模型: {MODEL_ID} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 如果显卡不支持 FlashAttn，可改为 None
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 7. 获取所有类别目录
    categories = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d)) and d != 'saved_images']
    print(f"\n找到 {len(categories)} 个类别: {categories}")
    
    # 8. 遍历每个类别
    for category in categories:
        category_path = os.path.join(DATASET_ROOT, category)
        
        # 获取类别下的子目录
        if SUBDIR_FILTER:
            subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d)) and d == SUBDIR_FILTER]
            print(f"类别 {category} 下的 {SUBDIR_FILTER} 子目录: {subdirs}")
        else:
            subdirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
            print(f"类别 {category} 下的所有子目录: {subdirs}")
        
        # 遍历每个子目录
        for subdir in subdirs:
            subdir_path = os.path.join(category_path, subdir)
            output_dir = os.path.join(OUTPUT_ROOT, category, subdir)
            
            # 创建类别和子目录结果目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            output_json = os.path.join(output_dir, f"{category}_{subdir}_results.json")
            
            # 检查是否已经存在结果文件，如果存在则跳过
            if os.path.exists(output_json):
                print(f"\n跳过类别: {category}/{subdir} (结果文件已存在)")
                continue
            
            print(f"\n处理类别: {category}/{subdir}")
            print(f"数据集路径: {subdir_path}")
            print(f"结果保存路径: {output_json}")
            
            # 加载数据集
            try:
                dataset = load_from_disk(subdir_path)
                # 假设数据集直接是一个数据集对象
                eval_data = dataset
            except Exception as e:
                print(f"数据集加载失败，请检查路径: {e}")
                continue

            # 9. 检查是否存在已处理的结果
            results = []
            start_idx = 0
            if os.path.exists(output_json):
                try:
                    with open(output_json, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    start_idx = len(results)
                    print(f"已找到部分结果，从索引 {start_idx} 开始处理...")
                except Exception as e:
                    print(f"加载已存在结果失败，重新开始处理: {e}")
                    results = []
                    start_idx = 0
            else:
                print(f"开始处理，共 {len(eval_data)} 条数据...")

            # 10. 遍历数据集进行推理 (从start_idx开始)
            for idx in tqdm(range(start_idx, len(eval_data)), total=len(eval_data)-start_idx):
                try:
                    item = eval_data[idx]
                    
                    # --- 提取内容 ---
                    question_text = item.get('question', '')
                    
                    # 检查是否有图片
                    if 'image' in item:
                        pil_image = item['image']  # 这是一个 PIL.Image 对象
                        
                        # --- 保存图片到本地 ---
                        # 为了在 JSON 中引用，我们需要把图片存成文件
                        image_filename = f"img_qwenguard_{dataset_name}_{category}_{subdir}_{idx}.jpg"
                        image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
                        # 将 PIL 图片保存下来
                        pil_image.save(image_path)

                        # --- 构建 Qwen2.5-VL 的输入格式 ---
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": pil_image, # 直接传入 PIL 对象
                                    },
                                    {"type": "text", "text": question_text},
                                ],
                            }
                        ]
                    else:
                        # 纯文本输入
                        image_path = None
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question_text},
                                ],
                            }
                        ]

                    # --- 预处理 ---
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)

                    # --- 模型生成 ---
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                    
                    # --- 解码输出 ---
                    # 去掉输入部分的 token，只保留模型新生成的回答
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                    # --- 收集结果 ---
                    result_entry = {
                        "id": idx,
                        "question": question_text,
                        "image": image_path,  # JSON 中存路径，而不是图片本身
                        "answer": output_text
                    }
                    results.append(result_entry)

                    # (可选) 每处理10条就打印一次，防止程序假死
                    if idx % 10 == 0:
                        print(f"\n[类别: {category}/{subdir}, 示例 {idx}] Q: {question_text[:30]}... -> A: {output_text[:30]}...")
                    
                    # --- 内存管理 ---
                    if (idx + 1) % CLEAR_CACHE_INTERVAL == 0:
                        torch.cuda.empty_cache()
                    
                    # --- 定期保存结果 ---
                    if (idx + 1) % SAVE_INTERVAL == 0:
                        with open(output_json, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)
                        print(f"\n已保存 {idx + 1} 条结果到 {output_json}")
                except Exception as e:
                    print(f"处理示例 {idx} 时出错: {e}")
                    # 跳过出错的示例，继续处理下一个
                    continue

            # 11. 保存最终结果
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            print(f"\n类别 {category}/{subdir} 处理完成！结果已保存至 {output_json}")
            
            # 处理完一个子目录后清空缓存
            torch.cuda.empty_cache()
    
    print(f"\n所有类别处理完成！结果保存在 {OUTPUT_ROOT} 目录下")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行 QwenGuard 模型推理")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=default_dataset, 
        choices=list(DATASETS.keys()),
        help=f"选择要使用的数据集 (默认: {default_dataset})"
    )
    args = parser.parse_args()
    
    # 运行推理
    run_inference(args.dataset)