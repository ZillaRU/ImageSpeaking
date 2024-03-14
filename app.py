# -*- coding: UTF-8 -*-
import os
import argparse
import gradio as gr
import numpy as np
import torch
from img_speaking_pipeline import ImageSpeakingPipeline as ISPipeline
from PIL import Image
import time


parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('-n', '--number', type=int, default=3, help="the number of sentences")
args = parser.parse_args()

pipeline = ISPipeline()

NUM_OF_DESCRIPTION=args.number

def describe_image(img_path, num=NUM_OF_DESCRIPTION):
    if img_path is None:
        return '[ INVALID INPUT ]'
    img_description, tags = pipeline(img_path, num_return_sequences=num)
    print(img_description, tags)
    return "\n".join([i.replace("a picture of ","") for i in img_description])


# Description
title = f"<center><strong><font size='8'>看图说话💬 powered by 1684x <font></strong></center>"

default_example = ["./resources/image/demo1.jpg", "./resources/image/demo3.jpg"]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


with gr.Blocks(css=css, title="看图说话") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    description_p = """ # 使用方法

            1. 上传图像。
            2. 点击“描述”。
            """
    with gr.Row():
        with gr.Column():
            img_inputs = gr.Image(label="选择图片", value=default_example[0], sources=['upload'], type='filepath')
        with gr.Column():
            img_descrip_text = gr.Textbox(label='💬', text_size=6, lines=3, interactive=False)
            

    # Submit & Clear
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    btn_p = gr.Button(
                        "描述", variant="primary"
                    )
                    clear_btn_p = gr.Button("清空", variant="secondary")


        with gr.Column():
            # Description
            gr.Markdown(description_p)

    btn_p.click(
        describe_image, inputs=[img_inputs], outputs=[img_descrip_text]
    )
    def clear():
        return [None, None]

    clear_btn_p.click(clear, outputs=[img_inputs, img_descrip_text])


demo.queue()
demo.launch(ssl_verify=False, server_name="0.0.0.0")
