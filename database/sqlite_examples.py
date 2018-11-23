# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/22

import sqlite3
import nltk

def sql_query(dbname, query):
    path = nltk.data.find('corpora/city_database/city.db')
    print(path)

    query='SELECT City FROM city_table WHERE Country="china";'
    connect = sqlite3.connect(str(path))
    cur = connect.cursor()
    return cur.execute(query)
