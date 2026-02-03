import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 图表输出目录
OUTPUT_DIR = "mm_safetybench_evaluation_charts"

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 模型名称映射
MODEL_MAP = {
    "qwen2.5-vl": "qwen2.5-vl",
    "qwenguard": "qwenguard",
    "proguard": "proguard"
}

# 评估结果目录
EVALUATIONS_DIR = "mm_safetybench_model_evaluations"

# 存储所有评估结果
total_scores = {
    "qwen2.5-vl": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []},
    "qwenguard": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []},
    "proguard": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []}
}

# 存储每个风险类别的评估结果
category_scores = {}

# 风险类别名称映射（用于图表显示）
category_name_map = {
    "EconomicHarm": "Economic Harm",
    "Financial_Advice": "Financial Advice",
    "Fraud": "Fraud",
    "Gov_Decision": "Government Decision",
    "HateSpeech": "Hate Speech",
    "Health_Consultation": "Health Consultation",
    "Illegal_Activitiy": "Illegal Activity",
    "Legal_Opinion": "Legal Opinion",
    "Malware_Generation": "Malware Generation",
    "Physical_Harm": "Physical Harm",
    "Political_Lobbying": "Political Lobbying",
    "Privacy_Violence": "Privacy Violence",
    "Sex": "Sexual Content"
}

def load_evaluation_file(file_path):
    """加载单个评估文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_category(file_path):
    """从文件路径中提取风险类别和子目录"""
    # 从路径中提取类别名称和子目录
    dir_path = os.path.dirname(file_path)
    subdir = os.path.basename(dir_path)
    category = os.path.basename(os.path.dirname(dir_path))
    return f"{category}/{subdir}"

def process_evaluation_data(data, category):
    """处理评估数据"""
    # 初始化类别分数存储
    if category not in category_scores:
        category_scores[category] = {
            "qwen2.5-vl": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []},
            "qwenguard": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []},
            "proguard": {"qwen2.5-vl": [], "qwenguard": [], "proguard": []}
        }
    
    for item in data:
        if "evaluations" not in item:
            continue
        
        evaluations = item["evaluations"]
        
        for evaluator, eval_results in evaluations.items():
            if evaluator not in MODEL_MAP:
                continue
            evaluator_key = MODEL_MAP[evaluator]
            
            for model, result in eval_results.items():
                if model not in MODEL_MAP:
                    continue
                model_key = MODEL_MAP[model]
                
                if "score" in result:
                    score = result["score"]
                    # 添加到总分数
                    total_scores[evaluator_key][model_key].append(score)
                    # 添加到类别分数
                    category_scores[category][evaluator_key][model_key].append(score)

def calculate_averages(scores_dict):
    """计算平均分数"""
    averages = {
        "qwen2.5-vl": {"qwen2.5-vl": 0, "qwenguard": 0, "proguard": 0},
        "qwenguard": {"qwen2.5-vl": 0, "qwenguard": 0, "proguard": 0},
        "proguard": {"qwen2.5-vl": 0, "qwenguard": 0, "proguard": 0}
    }
    
    for evaluator, models in scores_dict.items():
        for model, scores in models.items():
            if scores:
                averages[evaluator][model] = np.mean(scores)
    
    return averages

def calculate_model_averages(averages):
    """计算每个模型在多模型评估下的均分"""
    model_averages = {}
    for model in MODEL_MAP.values():
        scores = []
        for evaluator in MODEL_MAP.values():
            scores.append(averages[evaluator][model])
        model_averages[model] = np.mean(scores)
    return model_averages

def generate_summary(averages, category_averages=None):
    """生成总结结果"""
    summary = {"evaluations": {}}
    
    for evaluator, models in averages.items():
        summary["evaluations"][evaluator] = {}
        for model, avg_score in models.items():
            summary["evaluations"][evaluator][model] = {
                "score": round(avg_score, 2),
                "reason": f"Average safety risk score: {round(avg_score, 2)}"
            }
    
    # 如果提供了类别平均分数，添加到总结中
    if category_averages:
        summary["category_evaluations"] = category_averages
    
    return summary

def plot_results(averages, title, filename):
    """生成图表"""
    # 准备数据
    evaluators = list(averages.keys())
    models = list(averages[evaluators[0]].keys())
    
    # 创建子图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置参数
    bar_width = 0.25
    index = np.arange(len(models))
    
    # 绘制条形图
    for i, evaluator in enumerate(evaluators):
        scores = [averages[evaluator][model] for model in models]
        ax.bar(index + i * bar_width, scores, bar_width, label=evaluator)
    
    # 设置图表属性（英文）
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Safety Risk Score (1-5)')
    ax.set_title(title)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 保存图表到指定目录
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_model_averages(model_averages):
    """生成模型均分图表"""
    # 准备数据
    models = list(model_averages.keys())
    scores = [model_averages[model] for model in models]
    
    # 创建子图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制条形图
    bars = ax.bar(models, scores, color=['blue', 'green', 'red'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 设置图表属性（英文）
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Safety Risk Score (1-5)')
    ax.set_title('Average Safety Risk Score Across All Evaluators')
    ax.set_ylim(0, 5)
    ax.grid(axis='y', alpha=0.3)
    
    # 保存图表到指定目录
    output_path = os.path.join(OUTPUT_DIR, 'model_averages_summary.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(input_file=None):
    """主函数"""
    # 如果提供了输入文件，只处理该文件
    if input_file:
        if os.path.exists(input_file):
            print(f"处理文件: {input_file}")
            data = load_evaluation_file(input_file)
            # 提取类别信息
            category = extract_category(input_file)
            process_evaluation_data(data, category)
        else:
            print(f"错误: 文件 {input_file} 不存在")
            return
    else:
        # 遍历所有评估文件
        if not os.path.exists(EVALUATIONS_DIR):
            print(f"错误: 目录 {EVALUATIONS_DIR} 不存在")
            return
        
        files_found = False
        for root, dirs, files in os.walk(EVALUATIONS_DIR):
            for file in files:
                if file.endswith('_evaluations.json'):
                    files_found = True
                    file_path = os.path.join(root, file)
                    category = extract_category(file_path)
                    print(f"处理文件: {file_path} (类别: {category})")
                    data = load_evaluation_file(file_path)
                    process_evaluation_data(data, category)
        
        if not files_found:
            print(f"警告: 目录 {EVALUATIONS_DIR} 中没有找到评估文件")
            return
    
    # 检查是否有评估数据
    if not total_scores or all(all(len(scores) == 0 for scores in models.values()) for models in total_scores.values()):
        print("错误: 没有找到有效的评估数据")
        return
    
    # 计算整体平均值
    overall_averages = calculate_averages(total_scores)
    
    # 计算每个风险类别的平均值
    category_averages = {}
    for category, scores in category_scores.items():
        category_averages[category] = calculate_averages(scores)
    
    # 计算每个模型在多模型评估下的均分
    model_averages = calculate_model_averages(overall_averages)
    
    # 生成总结
    summary = generate_summary(overall_averages, category_averages)
    
    # 保存总结结果
    with open('mm_safetybench_evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 为每个风险类别生成图表
    for category, avg_scores in category_averages.items():
        category_display_name = category_name_map.get(category, category)
        title = f"Safety Risk Evaluation by {category_display_name}"
        # 替换文件名中的斜杠为下划线，避免创建不存在的目录结构
        safe_category_name = category.replace('/', '_')
        filename = f"evaluation_{safe_category_name}.png"
        plot_results(avg_scores, title, filename)
    
    # 生成整体评估图表
    plot_results(overall_averages, "Overall Safety Risk Evaluation", "overall_evaluation.png")
    
    # 生成模型均分图表
    plot_model_averages(model_averages)
    
    print("\n评估总结完成！")
    print("总结结果已保存到: mm_safetybench_evaluation_summary.json")
    print(f"整体评估图表已保存到: {OUTPUT_DIR}/overall_evaluation.png")
    print(f"模型均分图表已保存到: {OUTPUT_DIR}/model_averages_summary.png")
    print(f"各风险类别图表已保存到: {OUTPUT_DIR}/evaluation_*.png")
    
    # 打印整体平均分数
    print("\nOverall average safety risk scores:")
    print("-" * 60)
    print(f"{'Evaluator':<15} {'Model':<15} {'Avg Score':<10}")
    print("-" * 60)
    for evaluator, models in overall_averages.items():
        for model, avg_score in models.items():
            print(f"{evaluator:<15} {model:<15} {avg_score:<10.2f}")
    print("-" * 60)
    
    # 打印每个模型在多模型评估下的均分
    print("\nAverage safety risk scores across all evaluators:")
    print("-" * 60)
    print(f"{'Model':<15} {'Avg Score':<10}")
    print("-" * 60)
    for model, avg_score in model_averages.items():
        print(f"{model:<15} {avg_score:<10.2f}")
    print("-" * 60)
    
    # 计算每个风险类别的总体平均得分（所有评估者和模型的平均值）
    category_overall_scores = {}
    for category, avg_scores in category_averages.items():
        total_score = 0
        count = 0
        for evaluator, models in avg_scores.items():
            for model, avg_score in models.items():
                total_score += avg_score
                count += 1
        if count > 0:
            category_overall_scores[category] = total_score / count
    
    # 按总体平均得分对风险类别进行排序（从高到低）
    sorted_categories = sorted(category_overall_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 打印每个风险类别的平均分数
    print("\nAverage safety risk scores by category:")
    print("-" * 80)
    print(f"{'Category':<20} {'Evaluator':<15} {'Model':<15} {'Avg Score':<10}")
    print("-" * 80)
    for category, _ in sorted_categories:
        category_display = category_name_map.get(category, category)
        avg_scores = category_averages[category]
        for evaluator, models in avg_scores.items():
            for model, avg_score in models.items():
                print(f"{category_display:<20} {evaluator:<15} {model:<15} {avg_score:<10.2f}")
    print("-" * 80)
    
    # 打印按总体平均得分排序的风险类别
    print("\nRisk categories sorted by overall average safety risk score (highest to lowest):")
    print("-" * 60)
    print(f"{'Category':<30} {'Overall Avg Score':<15}")
    print("-" * 60)
    for category, overall_score in sorted_categories:
        category_display = category_name_map.get(category, category)
        print(f"{category_display:<30} {overall_score:<15.2f}")
    print("-" * 60)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成评估总结和图表")
    parser.add_argument(
        "input_file", 
        type=str, 
        nargs='?', 
        default=None,
        help="可选的评估文件路径，如果提供则只处理该文件"
    )
    args = parser.parse_args()
    
    # 运行主函数
    main(args.input_file)
