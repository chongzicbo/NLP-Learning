#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: file_spyder.py
@time: 2022/6/30 9:21
"""

from __future__ import unicode_literals
import requests
import json
import os
import threading
import argparse
import logging

"""
pmid接口：https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17284678&retmode=xml
pmcid接口："https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC4775221" #全文接口

#以下接口，传入pmcid时，如果pmcid前包含"PMC"字符，则去掉
https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:212403&metadataPrefix=pmc
https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:8888445&metadataPrefix=oai_dc
https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:6763696&metadataPrefix=pmc
https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:152494&metadataPrefix=pmc_fm
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir",
    default="E:\\working\\huada_bgi\\data\\other_data\\付费文献\\for_test2",
    type=str,
    # required=True,
    help="filepath saved",
)

parser.add_argument(
    "--input_file_list",
    default="E:\\working\\huada_bgi\\data\\test_data\\for_test2.txt",
    type=str,
    # required=True,
    help="待爬取的文献清单",
)
parser.add_argument(
    "--log_file",
    default="file_spyder_for_test.log",
    type=str,
    # required=True,
    help="日志文件",
)


# save_dir = "E:\\working\\huada_bgi\\data\\other_data\\付费文献\\"


class myThread(threading.Thread):
    def __init__(self, threadID, name, s, e, text_list):
        """

        :param threadID:  线程编号
        :param name: 线程名称
        :param s: 线程处理的数据起始索引
        :param e:线程处理的数据终止索引
        :param text_list: 待处理的数据列表 list
        :param result_dic: 保存的结果dic
        """
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.s = s
        self.e = e
        self.text_list = text_list

    def run(self):
        print("Starting " + self.name + time.ctime(), end="")
        print(" From %d to %d" % (self.s, self.e))
        # threadLock.acquire()
        processSection(self.name, self.s, self.e, self.text_list)


def processSection(name, s, e, text_list):
    """
    :param name:
    :param s:
    :param e:
    :param text_list:
    :param result_dic:
    :return:
    """
    for i in range(s, e):
        processText(name, i, text_list[i])


def processText(name, number, pmcid):
    download_by_pmcid(save_dir=save_dir, pmcid=pmcid)
    print("Thread %s: have processed page %s " % (name, number))


def write_failed(filepath, id):
    with open(filepath, mode="a", encoding="utf-8") as fw:
        fw.write(id + "\n")


def download_by_pmcid(save_dir, pmcid):
    """
    根据pmcid下载文献：先拿到xml，然后调用it接口转换成json格式
    :param save_dir:
    :param pmcid:
    :return:
    """
    if pmcid.startswith("PMC"):
        new_pmcid = pmcid[3:]
    else:
        new_pmcid = pmcid
    url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:%s&metadataPrefix=oai_dc"
        % new_pmcid
    )
    url = (
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:%s&metadataPrefix=pmc"
        % new_pmcid
    )
    # url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:%s&metadataPrefix=pmc_fm" % new_pmcid
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%s" % pmcid
    )
    # xml_res = requests.get(
    #     url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=%s" % pmcid)
    xml_res = requests.get(url=url)
    # print(xml_res.text)
    if "error=" in xml_res.text:
        print("The following PMCID is not available: %s" % pmcid)
        logging.info("The following PMCID is not available: %s" % pmcid)
        write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
        return
    elif "<error>" in xml_res.text:
        print("Exception from Backend: bePMCXML, Couldn't+resolve failed %s" % pmcid)
        logging.info(
            "Exception from Backend: bePMCXML, Couldn't+resolve failed %s" % pmcid
        )
        write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
        return
    elif xml_res.text is None:
        print("文献 %s 下载失败" % pmcid)
        logging.info("文献 %s 下载失败" % pmcid)
        write_failed(os.path.join(save_dir, "failed_for_test.txt"), pmcid)
        return
    else:
        # print(xml_res.text)
        url = "http://10.227.4.138/lit-anno/api/litAnno/article/transfer2json"
        header = {"Content-Type": "application/json"}
        content = {"content": xml_res.text}
        content = json.dumps(content)
        # print(content)
        re = requests.post(url=url, headers=header, data=content)
        if re.status_code == 500:
            print("xml获取成功，但是xml转json解析失败")
            logging.info("xml获取成功，但是xml转json解析失败")
            write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
            return

        res = json.loads(re.text)
        if res["retCode"] == 500:
            print("xml获取成功，但是xml转json解析失败")
            logging.info("xml获取成功，但是xml转json解析失败")
            write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
            return
        try:
            annotations = {}
            if res["result"] is not None:
                abstract = is_abstract(res)
                annotations["doi"] = res["result"].get("doi", "")
                annotations["passages"] = res["result"].get("passages", "")
                annotations["pmc"] = pmcid
                annotations["pmid"] = res["result"].get("pmid", "")
                annotations["title"] = res["result"].get("title", "")
                filename = pmcid + ".json"
                # print(annotations)
                if abstract:
                    save_dir = save_dir + "/abstract"
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    with open(
                        os.path.join(save_dir, filename), encoding="utf-8", mode="w"
                    ) as fw:
                        fw.write(json.dumps(annotations, ensure_ascii=False))
                    # print("文献 %s 已经下载成功" % pmcid)
                else:
                    save_dir = save_dir + "/fulltext"
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    with open(
                        os.path.join(save_dir, filename), encoding="utf-8", mode="w"
                    ) as fw:
                        fw.write(json.dumps(annotations, ensure_ascii=False))
            else:
                print("文献 %s 下载失败" % pmcid)
                logging.info("文献 %s 下载失败" % pmcid)
                write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
        except json.decoder.JSONDecodeError:
            logging.info("decoder error:", re.text)
            logging.info("文献 %s 下载失败" % pmcid)
            write_failed(os.path.join(save_dir, "failed_pmcid.txt"), pmcid)
            # print(json.dumps(res))


def get_xml_pmid(pmid):
    xml_res = requests.get(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&retmode=xml"
        % pmid
    )
    print(xml_res.text)


def download_by_pmid(save_dir, pmid):
    xml_res = requests.get(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&retmode=xml"
        % pmid
    )
    url = "http://10.227.4.138/lit-anno/api/litAnno/article/transfer2json"
    print(xml_res.text)
    header = {"Content-Type": "application/json"}
    content = {"content": xml_res.text}
    content = json.dumps(content)
    re = requests.post(url=url, headers=header, data=content)
    res = json.loads(re.text)
    # print(json.dumps(res))
    annotations = {}
    if res["result"] is not None:
        annotations["doi"] = res["result"].get("doi", "")
        annotations["passages"] = res["result"].get("passages", "")
        annotations["pmc"] = res["result"].get("pmcId", "")
        annotations["pmid"] = res["result"].get("pmid", "")
        annotations["title"] = res["result"].get("title", "")
        filename = "pmid-" + pmid + ".json"
        # print(annotations)
        with open(os.path.join(save_dir, filename), encoding="utf-8", mode="w") as fw:
            fw.write(json.dumps(annotations, ensure_ascii=False))
            # print("文献 %s 已经下载成功" % pmcid)
    else:
        print("文献 %s 已经下载失败" % pmid)
        print(json.dumps(res))


def is_abstract(res):
    """
    判断文献是否仅包含摘要
    :param res:
    :return:
    """
    passages = res["result"].get("passages", None)
    sections = []
    if passages is not None:
        for passage in passages:
            sections.append(passage["metas"]["section"].lower())
    if "introduction" not in sections and "reference" not in sections:
        return True
    return False


def get_all_exist_file(save_dir):
    g = os.walk(save_dir)
    pmcid_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            pmcid_list.append(file_name[0:-5])
    return pmcid_list


# 判断文献是否已经存在
def exists(pmcid_list, id):
    if id in pmcid_list:
        logging.info("%s 文献已经存在" % id)
        return True
    return False


if __name__ == "__main__":
    import time

    start_time = time.time()
    args = parser.parse_args()
    save_dir = (
        args.save_dir
    )  # "E:\\working\\huada_bgi\\data\\other_data\\付费文献\\pmcid\\"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_file_path = os.path.join(args.save_dir, args.log_file)
    logging.basicConfig(filename=log_file_path, level=logging.ERROR)

    # get_xml_pmid("15342251")
    # download_by_pmid(save_dir, pmid="27435154")
    download_by_pmcid(save_dir, pmcid="PMC8710260")
    # totalThread = 3
    # print(exists(save_dir,"PMC2654455"))
    with open(args.input_file_list) as fr:
        lines = fr.readlines()
        lines = [line.split(",")[0] for line in lines]
        lines = [line.strip() for line in lines]
        logging.info("待下载解析的pmcid全文文献共：%s 篇" % len(lines))
        # gap = int(len(lines) / totalThread)
        pmcid_list = get_all_exist_file(save_dir)
        remain_set = set(lines).difference(set(pmcid_list))
        logging.info("还剩余 %s 篇文献未下载" % len(remain_set))
        for i, line in enumerate(list(remain_set)):
            # if exists(pmcid_list, line.strip()):
            #     continue
            if i % 1000 == 1:
                end_time = time.time()
                logging.info("已经下载 %s 篇文献,共花费时间 %s 秒" % (i, (end_time - start_time)))
            if len(line.strip()) > 0:
                download_by_pmcid(save_dir, line.strip())
    # with open("E:\\working\\huada_bgi\\data\\produce\\all_cnv_pmid_nopmcid_list-20220704.csv") as fr:
    #     lines = fr.readlines()
    #     lines = [line.strip() for line in lines]
    #     print("待下载解析文献共：%s 篇" % len(lines))
    #     gap = int(len(lines) / totalThread)
    #     for line in lines:
    #         if len(line.strip()) > 0:
    #             download_by_pmid(save_dir, line.strip())
    #             break

    # threadLock = threading.Lock()  # 锁
    # threads = []
    #
    # for i in range(totalThread):
    #     thread = 'thread%s' % i
    #     if i == 0:
    #         thread = myThread(0, "Thread-%s " % i, 0, gap, lines)
    #     elif totalThread == i + 1:
    #         thread = myThread(i, "Thread-%s" % i, i * gap, len(lines), lines)
    #     else:
    #         thread = myThread(i, "Thread-%s " % i, i * gap, (i + 1) * gap, lines)
    #     threads.append(thread)  # 添加线程到列表
    #
    # for i in range(totalThread):
    #     threads[i].start()
    #
    # for t in threads:
    #     t.join()
    #
    # print("Exiting Main Thread")
    end_time = time.time()
    print("共花费时间：%s 秒" % (end_time - start_time))
