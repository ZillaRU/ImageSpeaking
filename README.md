# 看图说话 on 1684X
##### 功能描述：看图说话（Image Caption）技术使计算机能够通过自然语言描述图像内容，广泛应用于辅助视觉障碍人士、社交媒体、电子商务、智能信息检索、医疗图像分析、无人驾驶汽车、智能家居监控等领域，提高信息可访问性，增强用户体验，促进安全与效率。

1. `git clone https://github.com/ZillaRU/ImageSpeaking.git`
2. `pip3 install -r requirements.txt`
3. 把tpu_perf包的infer.py替换为`./replace-this-file/infer.py`。
    ```bash
    cp ./replace-this-file/infer.py /home/linaro/.local/lib/python3.8/site-packages/tpu_perf/ 
    ```
    若用了virtualenv，不是该路径可`python -c "import tpu_perf.infer as infer; print(infer.__file__)"`看下。
    这一步是因为，pipeline中用到GroundingDINO fp16模型，用sail推理会引起芯片Fault必须用tpu_perf。
4. 下载bmodel，并解压放在该项目根目录的`bmodel`文件夹。
   ```bash
   pip3 install dfss -U
   python3 -m dfss --url=open@sophgo.com:/aigc/hik_llm.tar.gz
   ```
6. 运行demo。`python3 app.py -n [描述的数量]`，模型加载完毕后终端会显示端口号，浏览器访问`本机ip:端口号`即可。

