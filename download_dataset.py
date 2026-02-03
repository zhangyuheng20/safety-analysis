from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
import os

def download_all_configs(dataset_name, save_path_root="./mm_safetybench"):
    print(f"正在获取 {dataset_name} 的所有子集配置列表...")
    
    try:
        # 1. 自动获取所有子集名称 (Config Names)
        # 这会自动拉取报错信息里看到的那个列表
        configs = get_dataset_config_names(dataset_name)
        print(f"检测到 {len(configs)} 个子集: {configs}")
        
        all_splits = []

        # 2. 循环下载每个子集
        for config in configs:
            print(f"\n--- 正在处理子集: {config} ---")
            # 下载该子集
            ds = load_dataset(dataset_name, config)
            
            # 3. 保存逻辑 (有两种选择，这里演示分别保存)
            # 方式 A: 每个子集存一个文件夹
            subset_path = os.path.join(save_path_root, config)
            ds.save_to_disk(subset_path)
            print(f"子集 {config} 已保存到: {subset_path}")
            
            # (可选) 如果你想最后合并成一个大文件，可以将 train split 加入列表
            # if 'train' in ds:
            #     all_splits.append(ds['train'])

        print(f"\n全部下载完成！数据保存在: {save_path_root}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 运行此函数将自动下载 BeaverTails-V 的所有分类
    download_all_configs("PKU-Alignment/MM-SafetyBench")