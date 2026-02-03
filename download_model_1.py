import os

# 1. 强制设置国内镜像环境变量 (代码里设置，比终端更稳)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# --- 配置区 ---
REPO_ID = "AIML-TUDA/QwenGuard-v1.2-7B" # 你要下载的模型ID
LOCAL_DIR = "./QwenGuard-v1.2-7B"   # 下载到本地的文件夹路径
TOKEN = None                           # 如果下载 Llama3 等需要权限的模型，填入你的 HF Token

# --- 核心逻辑 ---

print(f"正在准备下载: {REPO_ID}")
print(f"保存路径: {LOCAL_DIR}")
print("正在尝试连接镜像站...")

try:
    # 2. 调用 snapshot_download
    # 它的功能等同于 CLI 的 download 命令
    # resume_download=True (新版默认开启) 确保中断后可以续传
    model_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        token=TOKEN,
        local_dir_use_symlinks=False,  # 下载真实文件，而不是快捷方式 (Windows/Linux通用推荐)
        resume_download=True,          # 断点续传
        max_workers=8                  # 多线程下载加速
    )
    print(f"\n下载成功！模型已保存在: {model_path}")

except Exception as e:
    print(f"\n下载出错: {e}")
    print("建议：如果网络依然不稳定，请尝试重新运行脚本，它会自动从断开处继续。")