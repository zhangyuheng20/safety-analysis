from datasets import load_from_disk

# 这里填你之前保存的那个文件夹路径
# 注意：是文件夹路径，不是具体的 .arrow 文件名
dataset_path = "./beaver_tails_data" 

try:
    # 1. 加载数据
    dataset = load_from_disk(dataset_path)
    
    print("数据加载成功！")
    print(dataset)
    
    # 2. 查看第一条数据
    print("\n第一条数据样本：")
    print(dataset['evaluation'][0])
    
except Exception as e:
    print(f"加载出错: {e}")