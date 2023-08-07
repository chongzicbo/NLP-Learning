#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author;data: zengquanlei;04/07/22


# 通过命令行参数获取输入文件名（包含路径）和输出文件名（包含路径）
def main(argv):
    global inputfile_name
    global outputfile_name
    inputfile_name = ""
    outputfile_name = ""
    try:
        opts, args = getopt.getopt(argv, "hvi:o:", ["ifile_path=", "ofile_path="])
    except getopt.GetoptError:
        print("pmid2pmcid.py -i <inputfile_name> -o <outputfile_name>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("pmid2pmcid.py -i <inputfile_name> -o <outputfile_name>")
            sys.exit()
        elif opt == "-v":
            print("pmid2pmcid.py V1.0 date: 04/07/2022")
            sys.exit()
        elif opt in ("-i", "--ifile_path"):
            inputfile_name = arg
        elif opt in ("-o", "--ofile_path"):
            outputfile_name = arg
    print("输入文件名称为：", inputfile_name)
    print("输出文件名称为：", outputfile_name)


if __name__ == "__main__":
    import csv
    import requests
    import getopt
    import sys
    from bs4 import BeautifulSoup
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main(sys.argv[1:])

# 读取待爬取的pmid清单
PMID_list = []
with open(inputfile_name, "r", newline="", encoding="utf-8") as f1:
    csv_reader = csv.reader(f1)
    for row in csv_reader:
        PMID_list.append(row[0])

# 爬取参数设置
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
}
requests.adapters.DEFAULT_RETRIES = 5
s = requests.session()
s.keep_alive = False

convert_url = r"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={}"


# 构建获取pmid的函数
def convert_pmid(url):
    pmcid_list = []
    res = requests.get(url, headers=headers, verify=False, timeout=(60, 60))
    soup = BeautifulSoup(res.text, "lxml")
    items = soup.find_all("record")
    for item in items:
        if item["pmcid"]:
            pmcid_list.append(item["pmcid"])
    return pmcid_list


# pmid转换pmcid
all_pmcid_list = []
for pmid in PMID_list:
    try:
        pmcid_list = convert_pmid(convert_url.format(pmid))
        all_pmcid_list.append([pmid] + pmcid_list)
    except:
        continue

# 将pmcid_list写入
with open(outputfile_name, "w", newline="", encoding="utf-8") as file3:
    csv_writer3 = csv.writer(file3)
    for row in all_pmcid_list:
        csv_writer3.writerow(row)
