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

LLM-Workbench is a toolkit for training, fine-tuning, and visualizing language models using Streamlit. It is very suitable for researchers and AI enthusiasts.

<img alt="GitHub" src="https://github.com/xlcaptain/LLM-Workbench/blob/main/static/img/summary.png">

## 🚀 Features
- **🤗 Knowledge Base Q&A**: We provide various ways of knowledge retrieval, where es uses 8.9.0 hybrid search to search for relevant knowledge fragments based on input and answer.
- **📚 Excel Q&A**: We use chatglm3 to generate corresponding python code based on the question, and use the kernel to execute the relevant code to return the result.
- **🎓 Model Training**: With our toolkit, you can easily train your own language model.
- **🔧 Model Fine-tuning**: We provide a simple way to fine-tune your model to better adapt to your specific task.
- **📊 Model Visualization**: Our toolkit includes some visualization tools that can help you better understand your model.

## 📦 Docker-compose Installation
Install ElasticSearch (please open the corresponding server port according to the docker-compose file, or customize it): 
```
cd docker/es
docker-compose up -d
```
To use the knowledge base Q&A, you need to build the corresponding index: 

Method one: You can install LLM-Workbench with the following command: 
```
cd LLM-Workbench
docker-compose up -d
```
For the case of using excel table Q&A, you need to enter the container and specify the kernel interpreter: 
```
ipython kernel install --name llm --user
```
Where, llm corresponds to the conda environment name.

## 🎈 Usage

Method two: After installing LLM-Workbench, you can start it with the following command: 
```
pip install -r requirements.txt
streamlit run chat-box.py
```

Then, you can open the displayed URL in your browser to start using LLM-Workbench.

## 🤝 Contribution

We welcome any form of contribution! If you have any questions or suggestions, feel free to raise them on GitHub.

## 📄 License

LLM-Workbench is released under the MIT license. For more details, please see the [LICENSE](LICENSE) file.

## 📞 Contact Us

If you have any questions or suggestions, feel free to ask us via email or GitHub.

