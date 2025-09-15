import matplotlib
matplotlib.use('Agg')  # 必须在导入 plt 前设置！
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
import pandas as pd
import jieba
import re
import jieba.posseg as pseg
from collections import Counter
from wordcloud import WordCloud
import numpy as np
from io import BytesIO
import base64
from datetime import datetime
import os
import logging
import requests
import json


app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# base64图片转url
def generate_url(base64_image):
    postUrl = r'https://api.imgbb.com/1/upload'
    api_key = 'cd252b3a315af679db9b6f10dbe1eff9'
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0'
    req = requests.post(f'{postUrl}?key={api_key}', data = {'image': base64_image},
                        headers = {'user-agent': user_agent})
    js = req.json()
    image_url = js['data']['image']['url']
    return image_url


# 初始化jieba分词器（只在启动时执行一次）
def init_jieba():
    # 设备名词库
    DEVICE_NAMES = ["空调", "电脑", "打印机", "照明灯", "插座", "开关",
                    "网线", "门禁", "服务器", "软件", "设备", "实验室",
                    "焊接车间", "天花板", "管路", "线缆", "减震器", "隔音墙", "咖啡机", "灯", "键盘", "鼠标", "网络"]

    # 问题描述词库
    PROBLEM_WORDS = ["漏水", "漏电", "损坏", "打不开", "报错", "停止工作",
                     "卡纸", "不亮", "无法开机", "失灵", "脱落", "断裂",
                     "异响", "无法打印", "连接异常", "没反应", "坏", "烧",
                     "停", "关", "开", "启", "拆", "整", "贴", "无反应"]

    # 添加自定义词典
    for word in DEVICE_NAMES + PROBLEM_WORDS:
        jieba.add_word(word, freq=1000)


# 执行初始化
init_jieba()


# 文本清洗函数
def clean_text(text):
    # 移除特殊符号但保留字母数字和中文字符
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# 精准分词与词性标注函数
def precise_cut(text):
    words = pseg.cut(text)
    return words


# 构建设备-问题组合词
def build_compound_terms(terms):
    compounds = []
    i = 0
    while i < len(terms) - 1:
        # 如果当前词是设备名，下一个词是问题描述，则组合
        if terms[i] in DEVICE_NAMES and terms[i + 1] in PROBLEM_WORDS:
            compounds.append(f"{terms[i]}{terms[i + 1]}")
            i += 2  # 跳过下一个词
        else:
            # 单独添加有意义的词
            if terms[i] in PROBLEM_WORDS or terms[i] in DEVICE_NAMES:
                compounds.append(terms[i])
            i += 1

    # 处理最后一个词
    if i < len(terms) and (terms[i] in PROBLEM_WORDS or terms[i] in DEVICE_NAMES):
        compounds.append(terms[i])

    return compounds





# 生成自定义形状词云
def generate_custom_wordcloud(word_freq):
    try:
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, 'fonts', 'msyh.ttc')

        # 检查字体文件是否存在
        if not os.path.exists(font_path):
            # 尝试备用路径
            font_path = '/usr/share/fonts/truetype/microsoft/microsoft-yahei.ttf'
            if not os.path.exists(font_path):
                # 最后尝试Windows路径
                font_path = 'C:/Windows/Fonts/msyh.ttc'
                if not os.path.exists(font_path):
                    raise FileNotFoundError("中文字体文件缺失，请确保msyh.ttc存在于fonts目录")

        logger.info(f"使用字体路径: {font_path}")

        # 配置词云
        wc = WordCloud(
            font_path=font_path,
            background_color='white',
            max_words=50,
            colormap='Reds',
            contour_width=1,
            contour_color='#1f77b4',
            scale=2,
            random_state=42,
            width=600,
            height=400,
            margin=5
        )

        # 生成词云
        wc.generate_from_frequencies(word_freq)

        # 创建图像缓冲区
        img_buffer = BytesIO()
        plt.figure(figsize=(6, 4), dpi=100)
        plt.imshow(wc, interpolation='lanczos')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        logger.error(f"生成词云时出错: {str(e)}")
        raise


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        # 接收JSON数据
        data = request.json

        # 验证输入数据
        if not data or 'records' not in data:
            logger.warning("无效的请求数据: 缺少'records'字段")
            return jsonify({'error': '无效的请求数据'}), 400

        # 创建数据帧
        records = data['records'][0]['entity']['Power BI values']
        dataset = pd.DataFrame(records)

        # 检查必要字段
        if '项目/问题点描述' not in dataset.columns:
            logger.warning("数据中缺少'项目/问题点描述'列")
            return jsonify({'error': '数据中缺少"项目/问题点描述"列'}), 400

        # 获取异常描述列的所有文本
        text = '\n'.join(dataset['项目/问题点描述'].astype(str).tolist())

        # 如果文本为空
        if not text.strip():
            logger.warning("'项目/问题点描述'列内容为空")
            return jsonify({'error': '项目/问题点描述内容为空'}), 400

        # 清洗文本
        cleaned_text = clean_text(text)
        logger.info(f"清洗后文本长度: {len(cleaned_text)}")

        # 执行分词与词性标注
        word_objects = precise_cut(cleaned_text)

        # 提取名词和动词
        nouns_verbs = []
        for word, flag in words:
            if flag.startswith('n') or flag.startswith('v'):
                nouns_verbs.append(word)

        # 构建复合词
        compound_terms = build_compound_terms(nouns_verbs)

        # 过滤掉纯数字和尺寸描述
        filtered_terms = []
        for term in compound_terms:
            # 跳过纯数字和尺寸描述（如6mm, 10mm等）
            if re.match(r'^\d+\.?\d*[mm|cm|m|kg|g]?$', term):
                continue
            # 跳过单字词
            if len(term) == 1:
                continue
            filtered_terms.append(term)

        # 词频统计
        term_counts = Counter(filtered_terms)
        logger.info(f"高频词汇: {term_counts.most_common(10)}")

        # 生成词云图像
        image_buffer = generate_custom_wordcloud(term_counts)

        # 将图像转换为Base64字符串 - 这是更可靠的解决方案
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        # 将Base64转成图片URL
        imageUrl = generate_url(image_base64)

        # 返回结果（包含Base64图像和词频数据）
        return jsonify({
            'image_base64': image_base64,
            'word_freq': term_counts.most_common(20),
            'status': 'success',
            'ticket_type': 'Maintenance',
            'image_Url': imageUrl,
            'created_at': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"API处理错误: {str(e)}", exc_info=True)
        return jsonify({
            'error': '处理过程中发生错误',
            'details': str(e)
        }), 500


# 移除cleanup路由 - 使用Base64后不再需要临时文件

if __name__ == '__main__':
    # 生产环境应设置debug=False
    app.run(host='0.0.0.0', port=5090, debug=False)