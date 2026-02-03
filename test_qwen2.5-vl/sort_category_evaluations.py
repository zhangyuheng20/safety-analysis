import json

# 加载评估总结文件
with open('evaluation_summary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取风险类别评估结果
category_evaluations = data.get('category_evaluations', {})

# 计算每个风险类别的总体平均得分
category_overall_scores = {}
for category, avg_scores in category_evaluations.items():
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

# 打印排序结果
print("Risk categories sorted by overall average safety risk score (highest to lowest):")
print("-" * 80)
print(f"{'Category':<50} {'Overall Avg Score':<15}")
print("-" * 80)
for category, overall_score in sorted_categories:
    print(f"{category:<50} {overall_score:<15.2f}")
print("-" * 80)

# 生成排序后的 JSON 文件
sorted_category_data = {
    "sorted_categories": [
        {
            "category": category,
            "overall_score": round(overall_score, 2),
            "details": category_evaluations[category]
        }
        for category, overall_score in sorted_categories
    ],
    "overall_evaluations": data.get('evaluations', {})
}

# 保存排序结果到新文件
with open('sorted_category_evaluations.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_category_data, f, ensure_ascii=False, indent=2)

print("\n排序结果已保存到: sorted_category_evaluations.json")
