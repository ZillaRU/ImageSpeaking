# 看图说话 powered by BM1684X
## 功能描述
本仓库是基于BM1684X平台的**看图说话**的功能演示。
看图说话（Image Caption）技术使计算机能够**用自然语言描述图像内容**，广泛应用于辅助视觉障碍人士、社交媒体、电子商务、智能信息检索、医疗图像分析、无人驾驶汽车、智能家居监控等领域，提高信息可访问性，增强用户体验，促进安全与效率。

## 安装指南
1. 拉取代码，`git clone https://github.com/ZillaRU/ImageSpeaking.git`。
2. 进入项目目录，`cd ImageSpeaking`。
3. 安装依赖，`pip3 install -r requirements.txt`。
4. 下载bmodel，并解压（`tar xzvf hik_llm.tar.gz`）放在该项目根目录的`bmodel`文件夹。
   ```bash
   pip3 install dfss -U
   python3 -m dfss --url=open@sophgo.com:/aigc/hik_llm.tar.gz
   ```
6. 运行demo，`python3 app.py -n [描述语句的个数，默认为1]`，模型加载完毕后终端会显示端口号，浏览器访问`本机ip:端口号`即可。
    <img width="960" alt="示例" src="https://github.com/ZillaRU/ImageSpeaking/assets/25343084/f722efbc-ea2c-4e74-b556-d43cf42dedb7">

## 耗时统计
此为应用跑在服务器上测试的耗时数据，cpu耗时相比airbox可能偏短。未计入网络延迟。
|操作|耗时 (秒)|备注|
|---|---|---|
|image preprocess【CPU】|0.078|可用TPU加速。|
|SWIN【TPU，FP16】|0.045||
|tagging_head_infer【TPU，FP16】|0.021||
|tag_encoder_infer【TPU，FP32】|0.011||
|bert_infer_first【TPU，FP16】|0.011||
|bert_cls_infer_first【TPU，FP16】|0.006||
|BERT modules【TPU，FP16】|0.252||
|*看图说话Total*|0.436||
