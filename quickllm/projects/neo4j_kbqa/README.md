## 基于Noe4j的知识图谱搭建医疗知识图谱

执行步骤：

### 1.加载neo4j服务

### 2.安装依赖

```shell
pip install -r requirements.txt -i https://pypi.douban.com/simple
```

### 3.构建数据入库

```shell
python build_graph.py
```

### 3.测试运行

```shell
python kbqa_test.py
```