"""
Description: 使用modelscope下载glm-4-9b-chat。
Notes: 
模型会自动保存在 '/root/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat/' 目录。
"""
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('ZhipuAI/glm-4-9b-chat')