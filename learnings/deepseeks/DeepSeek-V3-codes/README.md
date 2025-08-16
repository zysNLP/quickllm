# 调试方式

1. 从HuggingFace中下载deepseek-v3除了模型以外的py、json等文件放到：deepseek-v3文件夹中

2. 在inference/model.py中加入相关加载的if __name__==__main__方法，然后运行inference/model.py调试
