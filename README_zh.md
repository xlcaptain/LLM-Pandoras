# LLM-Workbench


<p align="center">
    <a href="https://github.com/xlcaptain/LLM-Workbench/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/xlcaptain/LLM-Workbench/blob/main/README_zh.md">简体中文</a> |
    </p>
</h4>


LLM-Workbench是一个使用Streamlit进行语言模型训练、微调和可视化的工具包。它非常适合研究人员和AI爱好者使用。

<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/summary.png">

## 🚀 功能
- **🤗 知识库问答**：我们提供多种知识查找方式，其中es使用8.9.0 混合搜索来根据输入搜索相关知识片段，并进行回答。
- **📚 excel问答**：我们使用chatglm3根据问题生成对应的python代码，并且使用kernel内核来执行相关代码，以返回结果。
- **🎓 模型训练**：使用我们的工具包，您可以轻松地训练自己的语言模型。
- **🔧 模型微调**：我们提供了一种简单的方式来微调您的模型，以便更好地适应您的特定任务。
- **📊 模型可视化**：我们的工具包包含了一些可视化工具，可以帮助您更好地理解您的模型。

## 📦 Docker-compose 安装
安装ElasticSearch(请根据docker-compose文件开放服务器对应端口，或自定义修改)：
```angular2html
cd docker/es
docker-compose up -d
```
使用知识库问答，你需要构建相应的索引：
```angular2html

```
方式一：您可以通过以下命令安装LLM-Workbench：
```
cd LLM-Workbench
docker build -t llm-base:v1.0 .
docker-compose up -d 
```
其中，针对使用excel表格问答的情况，需要进入容器中，去指定kernel解释器：
```angular2html
ipython kernel install --name llm --user
```
其中，llm对应的是conda的环境名。
## 🎈 使用

方式二：在安装了LLM-Workbench之后，您可以通过以下命令启动它：
```angular2html
pip install -r requirements.txt
streamlit run chat-box.py
```

然后，您可以在浏览器中打开显示的URL，开始使用LLM-Workbench。

## 🤝 贡献

我们欢迎任何形式的贡献！如果您有任何问题或建议，请随时通过GitHub向我们提出。

## 📄 许可证

LLM-Workbench是在MIT许可证下发布的。有关详细信息，请参阅[LICENSE](LICENSE)文件。

## 📞 联系我们

如果您有任何问题或建议，欢迎通过电子邮件或GitHub向我们提问。