#!/bin/bash
pip3 install dfss -U
python3 -m dfss --url=open@sophgo.com:/aigc/hik_llm.tar.gz
tar xzvf hik_llm.tar.gz
mv data/ImageSpeaking/bmodel .
rm -rf hik_llm.tar.gz ./data