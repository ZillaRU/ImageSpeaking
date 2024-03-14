# 看图说话 on 1684X
TPU memory占用大概是6600M多点。
1. `git clone https://github.com/ZillaRU/AnnoAnything.git`
2. `pip3 install -r requirements.txt`
3. 把tpu_perf包的infer.py替换为`./replace-this-file/infer.py`。
    ```bash
    cp ./replace-this-file/infer.py /home/linaro/.local/lib/python3.8/site-packages/tpu_perf/ 
    ```
    若用了virtualenv，不是该路径可`python -c "import tpu_perf.infer as infer; print(infer.__file__)"`看下。
    这一步是因为，pipeline中用到GroundingDINO fp16模型，用sail推理会引起芯片Fault必须用tpu_perf。
4. 下载bmodel，并解压放在该项目根目录的`bmodel`文件夹。 [Google drive](https://drive.google.com/drive/folders/1WFfq32nKCYhEwJvCZV5XYw9sFRriZYqP?usp=sharing)
5. 运行demo，`python3 app.py`。

