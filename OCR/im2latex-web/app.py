from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# 设置图片上传文件夹
UPLOAD_FOLDER = "data/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)
        print(f"{filename} has been saved.")
        return redirect(url_for("display_image", filename=file.filename))


@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("data", filename="images/" + filename), code=301)


@app.route("/process/<method>", methods=["POST"])
def process_image(method):
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400
    filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filename)

    processed_text = None
    if method == "pix2text":
        processed_text = extract_mathematic_formula_pix2text(filename)
    elif method == "nougat":
        processed_text = extract_mathematic_formula_nougat(filename)
    else:
        return "Invalid processing method", 400

    # After processing the image, send back the processed text
    return processed_text


def extract_mathematic_formula_pix2text(img_path):
    # 假装这里是处理图像并返回Latex字符串的代码
    return "Latex String from pix2text"


def extract_mathematic_formula_nougat(img_path):
    # 假装这里是处理图像并返回Latex字符串的代码
    return "Latex String from nougat"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8504, debug=True)
