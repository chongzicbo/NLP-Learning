import streamlit as st

about_message = """
# About
## testing!
:smile:
"""

st.set_page_config(
    page_title="Streamlit example",
    page_icon="/data/bocheng/data/wordcounter/ic_fluent_text_word_count_24_regular.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://www.baidu.com/",
        "Report a bug": None,
        "About": about_message,
    },
)
st.image("/data/bocheng/data/wordcounter/ic_fluent_text_word_count_24_regular.png")


def t1():
    st.text("t1-ing!")


###组件
a = st.checkbox(
    "test_checkbox",
    value=False,
    key=None,
    help="testing",
    on_change=None,
    args=None,
    kwargs=None,
)

# 按钮
b = st.button(label="button", key=None, help="testing!", on_click=None)

# 下载按钮
c = st.download_button(
    label="download_button",
    data="testttt",
    file_name="test_.md",
    help="testing!",
    on_click=None,
)

# 单选框
d = st.radio(
    label="What's your favorite movie genre",
    options=("Comedy", "Drama", "Documentary"),
    index=2,
    help="testing!",
)

# 下拉选项
e = st.selectbox(
    "slectbox", ("Comedy", "Drama", "Documentary"), index=2, help="testing!"
)

# 多选
f = st.multiselect(
    "multiselect",
    ("Comedy", "Drama", "Documentary"),
    default=["Drama"],
    help="testing!",
)

# 滑动条
g = st.slider(
    label="slider",
    min_value=-10,
    max_value=10,
    value=-2,
    step=1,
    help="testing!",
    on_change=t1,
)

# 选择滑动条
h = st.select_slider(
    label="select_slider", options=[1, "test2", 3], value=3, help="testing!"
)

# 文本框
i = st.text_input(
    label="text_input", max_chars=30, value="test1", help="testing!", placeholder="请输入"
)

# 数字选择框
j = st.number_input(
    "number_input", min_value=-10, max_value=10, value=2, step=2, help="testing"
)

# 文本区域
k = st.text_area(
    "text_area", value="test1", max_chars=60, help="testing!", placeholder="请输入"
)
# 时间选择
from datetime import datetime

dt1 = datetime.today()
dt2 = datetime.today()
l = st.date_input(label="date_input", value=(dt1, dt2))
# 时间选择
m = st.time_input("time_input", value=None, help="testing!")

# 上传按钮
n = st.file_uploader(label="file_uploader", accept_multiple_files=True, help="testing!")

# 拾色器
o = st.color_picker("color_picker", "#00f900")
import streamlit.components.v1 as components

components.iframe(
    src="//player.bilibili.com/player.html?aid=376524564&bvid=BV1wo4y1X7Tk&cid=365010431&page=1",
    width=1080,
    height=720,
    scrolling=False,
)
# 边栏
add_selectbox = st.sidebar.selectbox(
    label="How would you like to be contacted?",
    options=("Email", "Home phone", "Mobile phone"),
    key="t1",
)

# 列布局
col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")
# 展开框
with st.expander(label="expander", expanded=False):
    st.write("tesing")
# container
with st.container():
    st.write("container")

container = st.container()
container.write("containertext1")
st.write("not container")

# 在container中继续调用组件
container.write("containertext2")

with st.empty():
    st.write("not empty here")
import time

empty = st.empty()
empty.text("still not empty")
time.sleep(3)
empty.text("change")
