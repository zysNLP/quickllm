import os
import pandas as pd
import requests
from tqdm import tqdm
import concurrent.futures
from loguru import logger
import hashlib

def get_md5(url):
    """获取URL的MD5值作为文件名"""
    return hashlib.md5(url.encode()).hexdigest()

def download_image(url, save_path):
    """下载单个图片"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        logger.error(f"下载图片失败 {url}: {str(e)}")
    return False

def download_images_batch(urls, save_dir):
    """并发下载多个图片"""
    os.makedirs(save_dir, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for url in urls:
            if isinstance(url, str) and url.strip():
                # 使用URL的MD5值作为文件名
                fname = get_md5(url) + os.path.splitext(url)[1]
                save_path = os.path.join(save_dir, fname)
                if not os.path.exists(save_path):
                    futures.append(executor.submit(download_image, url, save_path))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="下载图片"):
            future.result()

def main():
    # 设置路径
    dir_quickllm = "/data2/users/yszhang/quickllm"
    excel_path = f"{dir_quickllm}/learnings/dim_grpo/SoulTable_三品类尺寸待标注数据_标注测试0326_p12.xlsx"
    image_cache_dir = f"{dir_quickllm}/learnings/dim_grpo/image_cache_new"
    
    # 创建日志
    logger.add(os.path.join(image_cache_dir, "download.log"), rotation="10 MB")
    logger.info(f"开始下载图片到: {image_cache_dir}")
    
    # 读取Excel文件
    df = pd.read_excel(excel_path)
    logger.info(f"读取Excel文件: {excel_path}, 共{len(df)}条数据")
    
    # 收集所有图片URL
    fields = ["sku_img", "spu_img_1", "spu_img_2", "spu_img_3", "spu_img_4"]
    all_urls = set()
    for field in fields:
        urls = df[field].dropna().unique()
        all_urls.update(urls)
        logger.info(f"字段 {field} 有 {len(urls)} 个唯一URL")
    
    logger.info(f"总共需要下载 {len(all_urls)} 张图片")
    
    # 下载图片
    download_images_batch(all_urls, image_cache_dir)
    
    # 验证下载结果
    downloaded_files = len([f for f in os.listdir(image_cache_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    logger.info(f"下载完成，共下载 {downloaded_files} 个文件")

if __name__ == "__main__":
    main() 