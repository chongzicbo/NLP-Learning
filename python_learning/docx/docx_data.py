"""
Author: cheng bo bo.cheng@mdpi.com
Date: 2023-08-24 14:52:43
LastEditors: cheng bo bo.cheng@mdpi.com
LastEditTime: 2023-08-24 16:06:24
FilePath: /dev/mytest/docx_data.py
Description: 
"""
import pymysql
import os
import shutil


def get_connection(host: str, username: str, port: int, password: str, database: str):
    conn = pymysql.connect(
        host=host, user=username, port=port, passwd=password, database=database
    )
    return conn


conn = get_connection(
    host="10.10.0.86",
    username="readonlys",
    port=3306,
    password="Iij6AeCo",
    database="mdpipub",
)
cursor = conn.cursor()
dst_dir = "/home/bocheng/data/docx/not_formatted2"
os.makedirs(dst_dir,exist_ok=True)


def filter_key(key):
    return key.startswith("manu")


def prepare_data1():
    with open("/home/bocheng/dev/mytest/docx_list.txt", "r") as fr:
        for line in fr:
            if "done" in line:
                continue
            jorunal_name, id = line.split("-")
            jorunal_name = jorunal_name.strip()
            id = id.strip()

            sql = f"select hash_key,article_author_id from submission_manuscript where id={id} limit 1"
            cursor.execute(sql)
            hash_key, article_author_id = cursor.fetchone()
            article_dir = (
                f"/data/dms/submissions/User - {article_author_id}/{hash_key.strip()}/"
            )
            if not os.path.exists(article_dir):
                print(f"{article_dir} does not exists")
                continue
            try:
                for file in sorted(
                    filter(filter_key, os.listdir(article_dir)),
                    key=lambda f: int(f.split(".")[1][1:]),
                ):
                    if file.startswith("manuscript.v") and file.endswith("docx"):
                        filepath = os.path.join(article_dir, file)
                        dst_path = os.path.join(dst_dir, line.strip() + ".docx")
                        shutil.copy(filepath, dst_path)
                        print(f"{file} was copied to {dst_path}")
                        break
            except ValueError as exception:
                print(os.listdir(article_dir))
                break

            # break

def prepare_data2():
    with open("/home/bocheng/data/docx/hashkey_author_id_1000.csv", "r") as fr:
        for line in fr:
            if "done" in line:
                continue
            id, hash_key,article_author_id,jorunal_name = line.split(",")
            jorunal_name = jorunal_name.strip()
            id = id.strip()

            # sql = f"select hash_key,article_author_id from submission_manuscript where id={id} limit 1"
            # cursor.execute(sql)
            # hash_key, article_author_id = cursor.fetchone()
            article_dir = (
                f"/data/dms/submissions/User - {article_author_id}/{hash_key.strip()}/"
            )
            if not os.path.exists(article_dir):
                print(f"{article_dir} does not exists")
                continue
            try:
                for file in sorted(
                    filter(filter_key, os.listdir(article_dir)),
                    key=lambda f: int(f.split(".")[1][1:]),
                ):
                    if file.startswith("manuscript.v") and file.endswith("docx"):
                        filepath = os.path.join(article_dir, file)
                        filename=f"{jorunal_name}-{id}"
                        dst_path = os.path.join(dst_dir,filename + ".docx")
                        shutil.copy(filepath, dst_path)
                        print(f"{file} was copied to {dst_path}")
                        break
            except ValueError as exception:
                print(os.listdir(article_dir))
                break


if __name__ =="__main__":
    prepare_data2()