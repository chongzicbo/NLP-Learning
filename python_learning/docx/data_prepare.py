import docx
import os
import re
from copy import deepcopy

docx_dir = "/data/bocheng/autoformated/500_ms"


def extract_abstract_keyword(doc):
    para_index = {}
    for i, paragraph in enumerate(doc.paragraphs):
        if paragraph.text.strip().lower().startswith("abstract"):
            abstract_index = i
            para_index["abstract_index"] = i
        if paragraph.text.strip().lower().startswith("keyword"):
            keyword_index = i
            para_index["keyword_index"] = i

    abstracts = doc.paragraphs[
        para_index["abstract_index"] : para_index["keyword_index"]
    ]
    keywords = doc.paragraphs[para_index["keyword_index"]]
    for abs in abstracts:
        print(abs.text)

    print(keywords.text)


def extract_back_matter(doc):
    para_index = {}
    reference_index = -1
    disclaimer_index = -1
    introduction_index = -1
    conclusion_index = -1
    discussion_index = -1
    keywords_index = -1
    for i, paragraph in enumerate(doc.paragraphs):
        p_text = paragraph.text.strip().lower()
        if p_text.startswith("author contribution") or (
            "author" in p_text[:30] and "contribution" in p_text[:30]
        ):
            para_index["author contributions"] = i

        elif p_text.startswith("funding") or ("funding" in p_text[:10]):
            para_index["funding"] = i
        elif (
            p_text.startswith("data availability statement")
            or ("data availability statement" in p_text)
            or ("data" in p_text[:30] and "availability" in p_text[:30])
        ):
            para_index["data availability"] = i
        elif p_text.startswith("acknowledgment") or (
            "acknowled" in p_text[:30] and len(p_text) < 30
        ):
            para_index["acknowledgments"] = i
        elif p_text.startswith("abbreviation") and len(p_text) < 20:
            para_index["abbreviation"] = i

        elif "conflicts of interest" not in para_index and (
            p_text.startswith("conflicts of interest")
            or "conflicts of interest" in p_text[:30]
            or ("interest" in p_text and len(p_text) < 40)
            or ("conflict" in p_text[:30] and "interest" in p_text[:30])
            or "competing interest" in p_text[:60]
        ):
            para_index["conflicts of interest"] = i
        # elif p_text.startswith("disclaimer/publisher") or (
        #     "disclaimer/publisher" in p_text
        # ):
        #     para_index["disclaimer/publisher"] = i
        elif p_text.startswith("disclaimer/publisher") or (
            "disclaimer/publisher" in p_text
        ):
            # para_index["disclaimer/publisher"] = i
            disclaimer_index = i
        elif p_text.startswith("institutional review") or (
            "institutional review" in p_text[:40]
        ):
            para_index["institutional review"] = i
        elif "ethical" in p_text and "declaration" in p_text and len(p_text) < 30:
            para_index["ethical declaration"] = i

        elif p_text.startswith("supplementary materials") or (
            p_text.startswith("supplementary") and len(p_text) < 40
        ):
            para_index["supplementary materials"] = i
        elif p_text.startswith("informed consent") or (
            "consent" in p_text[:30] and "publication" in p_text[:30]
        ):
            para_index["informed consent"] = i
        elif "declaration" in p_text and len(p_text) < 20:
            para_index["declarations"] = i
        elif p_text.startswith("appendix") or (
            "appendix" in p_text and len(p_text) < 15
        ):
            para_index["appendix"] = i
        elif (
            (p_text.startswith("reference") and len(p_text) < 30)
            or p_text.endswith("references")
            or p_text.endswith("reference")
            or ("reference" in p_text and len(p_text) < 20)
            or ("literature" in p_text and len(p_text) < 15)
            or ("bibliography" in p_text and len(p_text) < 20)
        ):
            reference_index = i
        elif (
            p_text.endswith("introduction")
            or ("introduction" in p_text and len(p_text) < 30)
            or (p_text.startswith("background") and len(p_text) < 20)
        ):
            introduction_index = i
        elif (
            p_text.startswith("discussion")
            or p_text.endswith("discussion")
            or p_text.endswith("discussions")
        ) and len(p_text) < 50:
            discussion_index = i
        elif (p_text.startswith("conclusion") or p_text.endswith("conclusion")) and len(
            p_text
        ) < 50:
            conclusion_index = i
        elif p_text.startswith("keywords"):
            keywords_index = i
    # 没匹配到Introduction段落，则以keywords的后一段作为Introduction
    if introduction_index == -1 and keywords_index != -1:
        introduction_index = keywords_index + 1
    reference_end_index = -1
    # if para_index and sorted(para_index.values())[0] < introduction_index:
    #     para_index = {}
    # 把一些判断错误的back matter过滤掉
    para_index_cp = deepcopy(para_index)
    for p_i in para_index_cp:
        if introduction_index < para_index[p_i] and (
            para_index[p_i] < discussion_index or para_index[p_i] < conclusion_index
        ):
            para_index.pop(p_i)
        elif para_index[p_i] < introduction_index:
            para_index.pop(p_i)
    # main text (conclusion)-> reference
    # 没有back matter，有reference
    if not para_index and reference_index != -1:
        main_text = doc.paragraphs[introduction_index:reference_index]
        backmatter = []
        reference_end_index = update_reference(
            reference_index, doc.paragraphs[reference_index:]
        )
        references = (
            doc.paragraphs[reference_index:reference_end_index]
            if reference_end_index != -1
            else doc.paragraphs[reference_index:]
        )
    # 有reference 有back matter
    elif para_index and reference_index != -1:
        indexs = sorted(para_index.values())
        # main text (conclusion)-> backmatter -> reference -> disclaimer/publisher 这种情况
        if disclaimer_index != -1 and reference_index != -1:
            if reference_index < disclaimer_index and reference_index > indexs[-1]:
                backmatter = doc.paragraphs[indexs[0] : reference_index]
                main_text = doc.paragraphs[introduction_index : indexs[0]]
                references = doc.paragraphs[reference_index:disclaimer_index]
            # main text (conclusion) -> reference -> backmatter-> disclaimer/publisher 这种情况
            elif reference_index < indexs[0] and disclaimer_index > indexs[-1]:
                backmatter = doc.paragraphs[indexs[0] : disclaimer_index + 1]
                main_text = doc.paragraphs[introduction_index:reference_index]
                references = doc.paragraphs[reference_index : indexs[0]]
        elif reference_index != -1:
            # main text (conclusion)-> backmatter -> reference  这种情况 没有 disclaimer/publisher agriculture-2469420
            if reference_index > indexs[-1]:
                backmatter = doc.paragraphs[indexs[0] : reference_index]
                main_text = doc.paragraphs[introduction_index : indexs[0]]
                reference_end_index = update_reference(
                    reference_index, doc.paragraphs[reference_index:]
                )
                references = (
                    doc.paragraphs[reference_index:reference_end_index]
                    if reference_end_index != -1
                    else doc.paragraphs[reference_index:]
                )  # 这种情况下，references后面还可能存在一些别的东西比如图表附录，需要进一步判断
            # main text (conclusion) -> reference -> backmatter   这种情况 没有 disclaimer/publisher applsci-2453016
            elif reference_index < indexs[0]:
                backmatter = doc.paragraphs[indexs[0] :]
                main_text = doc.paragraphs[introduction_index:reference_index]
                references = doc.paragraphs[reference_index : indexs[0]]

    # for p in backmatter:
    #     print(p.text)
    for p in main_text[0:3] + main_text[-10:]:
        print(p.text)
    print(
        "============================================================================================================"
    )
    for p in backmatter:
        print(p.text)
    print(
        "============================================================================================================"
    )
    for p in [references[0]] + references[-10:]:
        print(p.text)


def update_reference(reference_start, reference_para):
    for i, p in enumerate(reference_para):
        if (
            p.text.strip().lower().startswith("figure")
            or p.text.strip().lower().startswith("fig.")
            or p.text.strip().lower().startswith("table")
        ):
            return reference_start + i
    return -1


if __name__ == "__main__":
    docx_path = "/data/bocheng/autoformated/not_formated/biomolecules-2514096.docx"
    doc = docx.Document(docx_path)
    # extract_abstract_keyword(doc)
    extract_back_matter(doc)
