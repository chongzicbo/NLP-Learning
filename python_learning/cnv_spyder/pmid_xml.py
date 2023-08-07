#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: pmid_xml.py
@time: 2022/7/12 17:25
"""

"""
根据pmid获取到xml文件，然后对xml文件进行解析，生成it格式的json格式文件
"""

import argparse
import json
import logging
import os
import xml
from xml.dom.minidom import parseString
from xml.dom.minidom import Element, Text

import requests

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir",
    default="E:\\working\\huada_bgi\\data\\other_data\\付费文献\\pmid",
    type=str,
    # required=True,
    help="filepath saved",
)

parser.add_argument(
    "--input_file_list",
    default="E:\\working\\huada_bgi\\data\\produce\\all_cnv_pmid_nopmcid_list-20220704.csv",
    type=str,
    # required=True,
    help="待爬取的文献清单",
)
parser.add_argument(
    "--log_file",
    default="file_spyder.log",
    type=str,
    # required=True,
    help="日志文件",
)


def write_failed(filepath, id):
    with open(filepath, mode="a", encoding="utf-8") as fw:
        fw.write(id + "\n")


def get_all_exist_file(save_dir):
    g = os.walk(save_dir)
    pmcid_list = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            pmcid_list.append(file_name[0:-5])
    return pmcid_list


def get_node_data(data: list, node):
    """ "
    递归获取节点数据
    """
    if isinstance(node, Text):
        data.append(node.data)
    elif isinstance(node, Element):
        for n in node.childNodes:
            if isinstance(n, Text):
                data.append(n.data)
            return get_node_data(data, n.firstChild)


def download_by_pmid(save_dir, pmid):
    pmc_dict = {}
    xml_res = requests.get(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&retmode=xml"
        % pmid
    )
    if "error=" in xml_res.text:
        print("The following PMCID is not available: %s" % pmid)
        logging.info("The following PMCID is not available: %s" % pmid)
        write_failed(os.path.join(save_dir, "failed_pmid.txt"), pmid)
        return
    # print(xml_res.text)
    try:
        dom = parseString(xml_res.text)
    except xml.parsers.expat.ExpatError:
        logging.info("pmid %s 文献下载失败" % pmid)
        write_failed(os.path.join(save_dir, "failed_pmid.txt"), pmid)
        return
    root = dom.documentElement
    # title = ""
    # title = root.getElementsByTagName("ArticleTitle")[0].data
    # title = root.getElementsByTagName("ArticleTitle")[0].firstChild.data
    # try:
    #     title = root.getElementsByTagName("ArticleTitle")[0].firstChild.data
    # except (AttributeError, IndexError):
    #     logging.info("%s 文献下载失败" % pmid)
    #     print(xml_res.text)
    # 获取期刊名称
    journal_nodes = root.getElementsByTagName("Title")
    journal_ls = []
    if len(journal_nodes) > 0:
        for node in journal_nodes[0].childNodes:
            get_node_data(journal_ls, node)
    journal_title = " ".join(journal_ls)

    journalTitle_dic = {"journalTitle": journal_title}
    journalMeta = [journalTitle_dic]
    journalMeta_dic = {"journalMeta": journalMeta}

    # 获取文章出版日期
    PubMedPubDate_nodes = root.getElementsByTagName("PubMedPubDate")
    PubMedPubDate = []
    for child in PubMedPubDate_nodes:
        date = {"dateType": "received"}
        for c in child.childNodes:
            if c.nodeName == "Year":
                date["year"] = c.firstChild.data
            elif c.nodeName == "Month":
                date["month"] = c.firstChild.data
            elif c.nodeName == "Day":
                date["day"] = c.firstChild.data

        PubMedPubDate.append(date)
    PubmedData_dic = {"history": PubMedPubDate}

    author_nodes = root.getElementsByTagName("Author")
    contribGroup = []
    for child in author_nodes:
        name = {}
        for c in child.childNodes:
            if c.nodeName == "LastName":
                surname = c.firstChild.data
                name["surname"] = surname
            elif c.nodeName == "ForeName":
                foreName = c.firstChild.data
                name["givenNames"] = foreName
        contrib = {}
        contrib["contribType"] = "author"
        contrib["name"] = name
        contribGroup.append(contrib)
    contribGroup_dic = {"contribGroup": contribGroup}
    articleMeta = [contribGroup_dic]
    articleMeta.append(PubmedData_dic)
    articleMeta_dic = {"articleMeta": articleMeta}

    title_nodes = root.getElementsByTagName("ArticleTitle")
    title_ls = []
    if len(title_nodes) > 0:
        for node in title_nodes[0].childNodes:
            get_node_data(title_ls, node)
    title = " ".join(title_ls)
    # if len(title_nodes) > 0:
    #     for node in title_nodes[0].childNodes:
    #         if node.firstChild is None:
    #             try:
    #                 title_ls.append(node.data)
    #             except AttributeError as e:
    #                 logging.info("%s : %s" % (pmid, e))
    #         elif len(node.childNodes):
    #             for n in node.childNodes:
    #                 if n.firstChild is None:
    #                     title_ls.append(n.data)
    #                 else:
    #                     title_ls.append(n.firstChild.data)
    #         else:
    #             title_ls.append(node.firstChild.data)
    #     # print(abstract_nodes)
    #     title = " ".join(title_ls)
    # else:
    #     title = ""

    abstract_nodes = root.getElementsByTagName("AbstractText")

    abstract_ls = []
    if len(abstract_nodes) > 0:
        for node in abstract_nodes[0].childNodes:
            get_node_data(abstract_ls, node)
    abstract = " ".join(abstract_ls)
    # if len(abstract_nodes) > 0:
    #     for node in abstract_nodes[0].childNodes:
    #         if node.firstChild is None:
    #             try:
    #                 abstract_ls.append(node.data)
    #             except AttributeError:
    #                 pass
    #         elif len(node.childNodes):
    #             for n in node.childNodes:
    #                 if n.firstChild is None:
    #                     abstract_ls.append(n.data)
    #                 else:
    #                     abstract_ls.append(n.firstChild.data)
    #         else:
    #             abstract_ls.append(node.firstChild.data)
    #     # print(abstract_nodes)
    #     abstract = " ".join(abstract_ls)
    # else:
    #     abstract = ""

    doi = ""
    articleIds = root.getElementsByTagName("ArticleId")
    for a_id in articleIds:
        if a_id.getAttribute("IdType") == "doi":
            if a_id.firstChild is not None:
                doi = a_id.firstChild.data
                break
    # print(title)
    # print(abstract)
    # print(doi)
    pmc_dict["doi"] = doi
    pmc_dict["pmid"] = pmid
    pmc_dict["title"] = title

    passages = []

    # title
    passage1 = {}
    metas1 = {}
    metas1["section"] = "Title"
    metas1["articleMeta"] = articleMeta_dic  # 作者信息
    metas1["journalMeta"] = journalMeta_dic  # 期刊信息
    passage1["metas"] = metas1
    passage1["offset"] = 0
    passage1["length"] = len(title)
    passage1["text"] = title
    passage1["annotations"] = []
    passages.append(passage1)

    # abstract
    passage2 = {}
    metas2 = {}
    metas2["section"] = "Abstract"
    passage2["metas"] = metas2
    passage2["offset"] = len(title)
    passage2["length"] = len(abstract)
    passage2["text"] = abstract
    passage2["annotations"] = []
    passages.append(passage2)

    pmc_dict["passages"] = passages
    pmc_dict["pmc"] = ""

    filename = pmid + ".json"
    with open(os.path.join(save_dir, filename), encoding="utf-8", mode="w") as fw:
        fw.write(json.dumps(pmc_dict, ensure_ascii=False))
    return json.dumps(pmc_dict, ensure_ascii=False)


if __name__ == "__main__":
    import time

    start_time = time.time()
    args = parser.parse_args()

    # pmc_json = download_by_pmid(args.save_dir, "16496165")

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    logging.basicConfig(
        filename=os.path.join(args.save_dir, args.log_file), level=logging.INFO
    )
    print(download_by_pmid(args.save_dir, "1614233"))
    with open(args.input_file_list) as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines]
        logging.info("待下载解析的pmcid全文文献共：%s 篇" % len(lines))
        pmid_list = get_all_exist_file(save_dir)
        remain_set = set(lines).difference(set(pmid_list))
        logging.info("还剩余 %s 篇文献未下载" % len(remain_set))
        for i, line in enumerate(list(remain_set)[:]):
            # if exists(pmcid_list, line.strip()):
            #     continue
            if i % 1000 == 1:
                end_time = time.time()
                logging.info("已经下载 %s 篇文献,共花费时间 %s 秒" % (i, (end_time - start_time)))
            if len(line.strip()) > 0:
                try:
                    download_by_pmid(save_dir, line.strip())
                except Exception:
                    print(line, "下载失败")
                    logging.info("%s 下载失败" % line)
    end_time = time.time()
    print("共花费时间：%s 秒" % (end_time - start_time))
