# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/26

import os

def get_project_root():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    return project_dir