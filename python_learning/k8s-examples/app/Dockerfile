FROM python:3.8

# 将工作目录设置为 /app
WORKDIR /app


# 将当前目录下的文件添加到工作目录中
ADD . /app
# 将 app.py 文件复制到镜像中的 /app 目录下
COPY app.py /app/  

# 安装所需的依赖
RUN pip install flask

# 暴露端口
EXPOSE 5001

# 定义环境变量
ENV NAME World

# 运行应用
CMD ["python", "app.py"]