# install for docker-compose
1. 执行下面命令，启动容器。其中，需要预先下载对应的embedding模型权重。
```angular2html
cd text-embedding-inference
docker-compose up -d
```
## 目前text-embedding-inference框架支持：
| MTEB Rank | Model Type  | Model Name                                                               | 
|-----------|-------------|--------------------------------------------------------------------------|
| 1         | BERT        | 	[BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| 2         |             | BAAI/bge-base-en-v1.5                                                    |                         
| 3         |             | llmrails/ember-v1                                                        |                                                     
| 4         |             | 	thenlper/gte-large                                                      |                                                     
| 5         |             | thenlper/gte-base                                                        |                                                     
| 6         |             | 	intfloat/e5-large-v2                                                    |                                                     
| 7         |             | 	BAAI/bge-small-en-v1.5,v_proj                                           |                                                     
| 10        |             | 	intfloat/e5-base-v2                                                     |                                          
| 11        | XLM-RoBERTa | 	intfloat/multilingual-e5-large                                          |                                                     
| N/A       | JinaBERT    | 	jinaai/jina-embeddings-v2-base-en                                       |                                               
| N/A       | JinaBERT    | 	jinaai/jina-embeddings-v2-small-en                                      |

## Api服务

```python
#main.py
    emb = Embeddings()
    result = emb.embed_documents(texts=['text'],api_base="http://ip:port")
```
具体，请查看main.py文件的源码。如果想使用其他形式（不是使用openai）可打开http:ip:port/docs，查看api文档。

## 详情
如需更详细，请参考官方文档：[text-embeddings-inferenc社区。](https://huggingface.co/docs/text-embeddings-inference/index)

