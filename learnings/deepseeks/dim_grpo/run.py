import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
from pathlib import Path
import hashlib
from loguru import logger
import json
import re
import numpy as np

# 设置环境变量，确保只使用指定的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_md5(url):
    """获取URL的MD5值作为文件名"""
    if not isinstance(url, str):
        url = str(url)
    return hashlib.md5(url.encode()).hexdigest()

def load_minicpm(model_path):
    """加载MiniCPM模型，使用与训练时相同的配置"""
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 启用低CPU内存使用
        device_map="cuda:0"  # 使用gpu 0号卡
    )
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 将模型移到GPU
    model = model.cuda()
    model.eval()
    
    # 设置模型参数为不需要梯度
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 显式加载与模型配套的图像处理器
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, image_processor

def build_prompt(input_text, images):
    """构建提示词，与训练时保持一致"""
    prompt = (
        f"基于以下产品信息，包括文本和多张图片，提取产品的长、宽、高尺寸。\n"
        f"产品文本描述: {input_text}\n"
        "请严格按照以下JSON格式输出，不要包含任何额外说明或markdown标记：\n"
        "{\"length\": 数字, \"width\": 数字, \"height\": 数字, \"unit\": \"cm\"}"
    )
    return prompt

def calculate_reward(answer, sample_label):
    """
    计算奖励，与训练时使用相同的函数
    分别对label和predict的l,w,h排序，计算排序后对应位置的差值
    reward = 1 - 平均相对误差
    """
    def parse_and_sort(label_str):
        """解析并排序"""
        numbers = re.findall(r'\d+\.?\d*', label_str)
        if len(numbers) >= 3:
            return sorted([float(x) for x in numbers[:3]], reverse=True)  # 从大到小排序
        elif len(numbers) == 2:
            return sorted([float(x) for x in numbers[:2]], reverse=True)
        elif len(numbers) == 1:
            return [float(numbers[0])]
        return None
    
    def parse_answer_and_sort(answer_str):
        """解析answer并排序"""
        try:
            answer_dict = json.loads(answer_str)
            vals = []
            if 'length' in answer_dict:
                vals.append(float(answer_dict['length']))
            if 'width' in answer_dict:
                vals.append(float(answer_dict['width']))
            if 'height' in answer_dict:
                vals.append(float(answer_dict['height']))
            return sorted(vals, reverse=True) if vals else None
        except:
            return parse_and_sort(answer_str)
    
    # 解析并排序
    label_sorted = parse_and_sort(sample_label)
    answer_sorted = parse_answer_and_sort(answer)
    
    if not label_sorted or not answer_sorted:
        return 0.0
    
    # 计算排序后对应位置的相对误差
    min_len = min(len(label_sorted), len(answer_sorted))
    total_error = 0.0
    
    for i in range(min_len):
        label_val = label_sorted[i]
        answer_val = answer_sorted[i]
        if label_val > 0:
            rel_error = abs(answer_val - label_val) / label_val
            total_error += rel_error
    
    avg_error = total_error / min_len
    reward = max(0.0, 1.0 - avg_error)
    
    return reward

def load_trained_model(model_path, checkpoint_path):
    """加载训练后的模型"""
    try:
        model, tokenizer, image_processor = load_minicpm(model_path)
        
        # 加载训练后的权重
        logger.info(f"加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.eval().cuda()
        
        # 清理内存
        torch.cuda.empty_cache()
        
        logger.success("训练后模型加载成功")
        return model, tokenizer, image_processor
        
    except Exception as e:
        logger.error(f"加载训练后模型失败: {e}")
        return None, None, None

def predict_with_model(model, tokenizer, image_processor, text, images, max_length=64):
    """使用模型进行预测，添加异常处理"""
    try:
        prompt_str = build_prompt(text, images)
        user_msgs = [{"role": "user", "content": prompt_str, "images": images}]
        
        with torch.no_grad():
            answer = model.chat(
                msgs=user_msgs, tokenizer=tokenizer,
                max_new_tokens=max_length, temperature=0.7
            )
        
        return answer
        
    except Exception as e:
        logger.warning(f"预测失败: {e}")
        return "{\"length\": 0, \"width\": 0, \"height\": 0, \"unit\": \"cm\"}"

def extract_numbers(answer):
    """提取答案中的数值，与训练时保持一致"""
    try:
        answer_dict = json.loads(answer)
        numbers = []
        if 'length' in answer_dict:
            numbers.append(str(answer_dict['length']))
        if 'width' in answer_dict:
            numbers.append(str(answer_dict['width']))
        if 'height' in answer_dict:
            numbers.append(str(answer_dict['height']))
        return ' '.join(numbers)
    except:
        # 如果JSON解析失败，使用正则表达式提取数字
        numbers = re.findall(r'\d+\.?\d*', answer)
        return ' '.join(numbers)

def main():
    # 配置路径
    dir_quickllm = "/data2/users/yszhang/quickllm"
    model_path = f"{dir_quickllm}/models/OpenBMB/MiniCPM-o-2_6"
    data_path = f"{dir_quickllm}/learnings/dim_grpo/SoulTable_三品类尺寸待标注数据_标注测试0326_p12.xlsx"
    image_cache_dir = f"{dir_quickllm}/learnings/dim_grpo/image_cache_new"
    checkpoint_path = f"{dir_quickllm}/outputs/dim_grpo_minicpm_final_correct/model_step100.pt"
    
    # 检查checkpoint是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # 内存优化设置
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 加载数据集
    try:
        df = pd.read_excel(data_path)
        logger.info(f"加载数据集: {data_path}, 共{len(df)}条")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return
    
    # 加载图片缓存
    image_cache_dir = Path(image_cache_dir)
    if not image_cache_dir.exists():
        logger.error(f"图片缓存目录不存在: {image_cache_dir}")
        return
        
    existing_images = {f.name for f in image_cache_dir.glob('*') if f.is_file()}
    logger.info(f"找到 {len(existing_images)} 个图片文件")
    
    # 加载原始模型
    logger.info("加载原始模型...")
    try:
        original_model, original_tokenizer, original_image_processor = load_minicpm(model_path)
        logger.success("原始模型加载成功")
    except Exception as e:
        logger.error(f"加载原始模型失败: {e}")
        return
    
    # 加载训练后的模型
    trained_model, trained_tokenizer, trained_image_processor = load_trained_model(model_path, checkpoint_path)
    if trained_model is None:
        logger.error("无法加载训练后的模型，退出")
        return
    
    # 测试前10条数据
    test_count = min(10, len(df))
    logger.info(f"开始测试前{test_count}条数据...")
    
    results = []
    improved_count = 0
    total_original_reward = 0.0
    total_trained_reward = 0.0
    
    for idx in range(test_count):
        try:
            row = df.iloc[idx]
            
            # 获取图片
            fields = ["sku_img", "spu_img_1", "spu_img_2", "spu_img_3", "spu_img_4"]
            img_paths = []
            for f in fields:
                url = row[f]
                if isinstance(url, str) and url.strip():
                    url_filename_part = url.split('/')[-1]
                    fname = get_md5(url) + os.path.splitext(url_filename_part)[1]
                    if fname in existing_images:
                        img_paths.append(image_cache_dir / fname)
            
            images = [Image.open(p).convert('RGB') for p in img_paths if os.path.exists(p)]
            text = str(row["text"])
            label = str(row["label"]).replace("[", "").replace("]", "").replace(",", " ").replace("cm", "").strip()
            
            if not images:
                logger.warning(f"样本{idx+1}没有有效图片，跳过")
                continue
            
            logger.info(f"处理样本 {idx+1}: {text[:50]}...")
            
            # 使用原始模型预测
            original_answer = predict_with_model(original_model, original_tokenizer, original_image_processor, text, images)
            
            # 使用训练后的模型预测
            trained_answer = predict_with_model(trained_model, trained_tokenizer, trained_image_processor, text, images)
            
            # 计算奖励，使用与训练时相同的函数
            original_reward = calculate_reward(original_answer, label)
            trained_reward = calculate_reward(trained_answer, label)
            
            # 提取数值
            original_numbers = extract_numbers(original_answer)
            trained_numbers = extract_numbers(trained_answer)
            
            # 判断是否有改进
            improvement = trained_reward - original_reward
            if improvement > 0:
                improved_count += 1
            
            # 累计奖励
            total_original_reward += original_reward
            total_trained_reward += trained_reward
            
            # 输出结果
            print(f"\n样本 {idx+1}:")
            print(f"标签: {label}")
            print(f"原始: {original_numbers} (奖励: {original_reward:.3f})")
            print(f"训练: {trained_numbers} (奖励: {trained_reward:.3f})")
            print(f"改进: {improvement:+.3f}")
            
            results.append({
                'idx': idx + 1,
                'label': label,
                'original_answer': original_answer,
                'trained_answer': trained_answer,
                'original_reward': original_reward,
                'trained_reward': trained_reward,
                'improvement': improvement
            })
            
            # 清理内存
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"处理样本{idx+1}时出错: {e}")
            continue
    
    # 打印总结
    print("\n" + "="*60)
    print("总结:")
    print("="*60)
    
    if results:
        avg_original_reward = total_original_reward / len(results)
        avg_trained_reward = total_trained_reward / len(results)
        avg_improvement = (total_trained_reward - total_original_reward) / len(results)
        
        print(f"样本数: {len(results)}")
        print(f"原始模型平均奖励: {avg_original_reward:.3f}")
        print(f"训练后模型平均奖励: {avg_trained_reward:.3f}")
        print(f"平均改进: {avg_improvement:+.3f}")
        print(f"改进样本数: {improved_count}/{len(results)} ({improved_count/len(results)*100:.1f}%)")
        
        # 详细分析
        print("\n详细分析:")
        print("-"*30)
        perfect_original = sum(1 for r in results if r['original_reward'] >= 0.9)
        perfect_trained = sum(1 for r in results if r['trained_reward'] >= 0.9)
        print(f"高精度样本 (≥0.9): 原始{perfect_original}个 → 训练后{perfect_trained}个")
        
        low_original = sum(1 for r in results if r['original_reward'] < 0.3)
        low_trained = sum(1 for r in results if r['trained_reward'] < 0.3)
        print(f"低精度样本 (<0.3): 原始{low_original}个 → 训练后{low_trained}个")
        
        # 找出改进最大的样本
        if results:
            best_improvement = max(results, key=lambda x: x['improvement'])
            print(f"\n改进最大的样本: 样本{best_improvement['idx']}")
            print(f"  标签: {best_improvement['label']}")
            print(f"  原始奖励: {best_improvement['original_reward']:.3f}")
            print(f"  训练后奖励: {best_improvement['trained_reward']:.3f}")
            print(f"  改进: {best_improvement['improvement']:+.3f}")
    else:
        print("没有成功处理的样本")
    
    # 清理内存
    torch.cuda.empty_cache()
    logger.success("测试完成")

if __name__ == "__main__":
    main() 