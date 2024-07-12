import streamlit as st
from PIL import Image
import io
import base64

# 一个隐藏的text_input，用于接收图像的Base64数据
image_data = st.text_input("paste image here", key="paste_image_data")

# 将JavaScript粘贴事件处理脚本嵌入到页面
st.markdown(
    """
    <script>
    document.addEventListener('paste', function(event) {
        var items = (event.clipboardData || event.originalEvent.clipboardData).items;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') === 0) {
                var blob = items[i].getAsFile();
                var reader = new FileReader();
                reader.onloadend = function() {
                    var base64data = reader.result;                
                    var textInput = document.getElementById('paste_image_data');
                    textInput.value = base64data;
                    var event = new Event('change', { bubbles: true });
                    textInput.dispatchEvent(event);
                };
                reader.readAsDataURL(blob);
                break;
            }
        }
    });
    </script>
""",
    unsafe_allow_html=True,
)

# 如果从JavaScript接收到图像数据
if image_data and image_data.startswith("data:image"):
    # 解码Base64数据
    base64_data = image_data.split(",")[1]
    bytes_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(bytes_data))

    # 在这里可以对图像执行进一步的处理
    # ...

    # 将图像显示在Streamlit界面上
    st.image(image, caption="Pasted Image")
