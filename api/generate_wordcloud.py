# api/generate_wordcloud.py

import matplotlib
matplotlib.use('Agg')  # 必须在导入 plt 前设置
import matplotlib.pyplot as plt

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
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi



# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB 连接配置
def get_mongo_client():
    try:
        # 从环境变量获取 MongoDB 连接字符串
        mongo_uri = "mongodb+srv://admin:Diotec2005@cluster0.dpxn7cl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        client = MongoClient(mongo_uri)
        # 测试连接
        client.admin.command('ping')
        logger.info("成功连接到 MongoDB")
        return client
    except Exception as e:
        logger.error(f"MongoDB 连接失败: {str(e)}")
        return None

# 初始化 MongoDB 连接
mongo_client = get_mongo_client()


# 初始化jieba分词器（只在启动时执行一次）
def init_jieba():
    # 设备/零件前缀词库
    DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                     "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                     "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                     "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点", "锡膏"]

    # 添加设备相关词汇
    for prefix in DEVICE_PREFIX:
        jieba.add_word(prefix, freq=1000)

    # 添加否定性复合词
    negative_terms = [
        "不合格", "不良率", "不良品", "不工作", "不转动", "不推料", "不下料", "不焊",
        "未焊住", "未检出", "未测试", "未达标", "未完成", "未通过", "无效果", "无响应",
        "无法启动", "无法运行", "无法测试", "无法检出", "异常报警", "异常停机", "异常高",
        "异常低", "异常波动", "异常偏高", "异常偏低"
    ]
    for term in negative_terms:
        jieba.add_word(term, freq=1000)

    # 添加复合故障词
    faults = ["断脚", "漏油", "漏晶", "卡料", "压伤", "死机", "翘脚", "报警频繁", "吸不起", "不下料",
              "缺角", "印不到", "印字深浅", "偏位", "竖芯片", "粘连", "堵料", "反极性", "打弯管", "测不到",
              "短路率", "本体破损", "盖带断裂", "不转动", "掉锡", "刮花", "不推料", "不合模", "氧化", "压痕浅",
              "错位", "未焊住", "虚焊", "脱焊", "侧立", "反贴", "假焊", "冷焊"]
    for fault in faults:
        jieba.add_word(fault, freq=1000)

    # 电性不良相关词
    elec_faults = ["IR不良", "VF不良", "VB不良", "CP超标", "AP不良", "O/S不良",
                   "漏电流", "耐压不足", "电性不良", "参数超标"]
    for fault in elec_faults:
        jieba.add_word(fault, freq=1000)


# 执行初始化
init_jieba()


# 高级文本清洗函数
def clean_text(text):
    # 移除HTML标签、数字、英文、特殊符号
    text = re.sub(r'<[^>]+>', '', text)  # HTML标签
    text = re.sub(r'[a-zA-Z0-9\s+\.\%\/\、\°\&\;\-]+', ' ', text)  # 数字和英文
    text = re.sub(r'[\W]', ' ', text)  # 特殊符号
    text = re.sub(r'\s+', ' ', text)  # 多个空格
    return text.strip()


# 精准分词与词性标注函数
def precise_cut(text):
    words = pseg.cut(text)
    return words


# 智能过滤规则 - 专注不合格问题
def is_valid_term(word_obj):
    word, flag = word_obj.word, word_obj.flag

    # 过滤正面表述和合格相关词
    positive_terms = ["达标", "正常", "OK", "通过", "良品", "Pass"]
    if any(pt in word for pt in positive_terms):
        return None

    # 过滤计量/单位词
    units = ["Kpcs", "K", "pcs", "kg", "%", "mv", "V", "A", "mA", "Ω", "℃"]
    if word in units:
        return None

    # 过滤中性词
    neutral_terms = ["调试", "设置", "操作", "调整", "显示", "运行", "需要"]
    if word in neutral_terms:
        return None

    # 良率问题统一归并
    if "良率" in word and "不良率" not in word:
        return "良率问题"
    if "良率" in word:
        return "良率问题"

    # 电性不良问题归并
    if re.search(r'(IR|VF|VB|CP|AP|O/S).*(不良|大|小|高|低|超标)', word):
        return re.sub(r'(合格|正常|Pass)', '不良', word)

    # 保留复合技术问题
    DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                     "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                     "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                     "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点", "锡膏"]

    if word in DEVICE_PREFIX or any(word.startswith(p) for p in DEVICE_PREFIX):
        return word

    # 保留特定技术故障词
    tech_faults = ["死机", "漏油", "卡料", "断脚", "压伤", "翘脚", "吸不起",
                   "测不到", "短路率", "堵料", "反极性", "氧化", "压痕浅", "未焊住",
                   "虚焊", "脱焊", "假焊", "冷焊", "侧立", "反贴"]
    if word in tech_faults:
        return word

    # 保留否定性复合词
    if word in ["不合格", "不良率", "不良品", "不工作", "不转动", "不推料", "不下料"]:
        return word

    # 保留长度在2-4字的名词性故障描述
    if 2 <= len(word) <= 4 and flag.startswith('n'):
        return word

    return None


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
        plt.figure(figsize=(12, 9), dpi=100)
        plt.imshow(wc, interpolation='lanczos')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight', pad_inches=0)
        plt.close()

        img_buffer.seek(0)
        return img_buffer

    except Exception as e:
        logger.error(f"生成词云时出错: {str(e)}")
        raise


# 保存词云到 MongoDB
def save_wordcloud_to_mongo(image_base64):
    try:
        if not mongo_client:
            logger.error("MongoDB 连接不可用")
            return False

        # 获取数据库和集合
        db = mongo_client.get_database('mydb')
        collection = db['wordcloud']

        # 准备数据
        wordcloud_data = {
            'image_base64': image_base64,
            'created_at': datetime.now()
        }

        # 使用固定ID进行覆盖写入
        result = collection.update_one(
            {'_id': 'latest_wordcloud'},
            {'$set': wordcloud_data},
            upsert=True  # 如果不存在则插入
        )

        if result.upserted_id or result.modified_count > 0:
            logger.info("词云数据成功保存到 MongoDB")
            return True
        else:
            logger.warning("词云数据保存到 MongoDB 但未更改")
            return True

    except Exception as e:
        logger.error(f"保存词云到 MongoDB 失败: {str(e)}")
        return False



# ======================
# Vercel Serverless 入口
# ======================
def handler(request):
    if request.method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        }

    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": {"error": "Method not allowed"}
        }

    try:
        data = request.json
        if not data or 'records' not in data:
            return {
                "statusCode": 400,
                "body": {"error": "缺少 records 字段"}
            }

        records = data['records'][0]['entity']['Power BI values']
        dataset = pd.DataFrame(records)

        if '异常描述' not in dataset.columns:
            return {
                "statusCode": 400,
                "body": {"error": "缺少 '异常描述' 列"}
            }

        text = '\n'.join(dataset['异常描述'].astype(str).tolist())
        if not text.strip():
            return {
                "statusCode": 400,
                "body": {"error": "异常描述为空"}
            }

        cleaned_text = clean_text(text)
        word_objects = precise_cut(cleaned_text)

        filtered_terms = []
        for wo in word_objects:
            result = is_valid_term(wo)
            if result == "良率问题":
                filtered_terms.append("良率问题")
            elif result:
                filtered_terms.append(result)

        # 构建复合词（设备+故障）
        DEVICE_PREFIX = ["转进头", "油压机", "转一", "转二", "马达", "吸嘴", "芯片", "框架", "封装",
                         "焊片", "炉子", "模具", "皮带", "链条", "直震", "圆振", "引直", "极测",
                         "封一", "封二", "轨一", "轨二", "测一", "测二", "测试", "影像", "印字", "盖带",
                         "设备", "焊炉", "镭射", "拉刀", "夹子", "顶针", "气缸", "阀", "台", "机", "炉", "锡", "锡点",
                         "锡膏"]

        compound_terms = []
        for i in range(len(filtered_terms) - 1):
            if (filtered_terms[i] in DEVICE_PREFIX and
                    filtered_terms[i + 1] not in DEVICE_PREFIX and
                    "良率问题" not in filtered_terms[i + 1] and
                    not filtered_terms[i + 1].endswith(('问题', '故障'))):
                compound_terms.append(filtered_terms[i] + filtered_terms[i + 1])

        term_counts = Counter(compound_terms + filtered_terms)
        image_buffer = generate_custom_wordcloud(term_counts)
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')

        # 保存到 MongoDB（可选）
        if mongo_client:
            save_wordcloud_to_mongo(image_base64)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": {
                "image_base64": image_base64,
                "word_freq": term_counts.most_common(20),
                "status": "success"
            }
        }

    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        return {
            "statusCode": 500,
            "body": {
                "error": "处理失败",
                "details": str(e)
            }
        }
