## 基于Noe4j的知识图谱搭建医疗知识图谱

执行步骤：

### 1.加载neo4j服务

```
# noe4j安装和使用流程
1. 在官网https://neo4j.com/deployment-center中，下拉找到Neo4j Desktop部分，注册账号和下载安装文件
2. 本地安装Neo4j Desktop并使用自带的激活码激活
3. 新建一个Neo4j Project，项目名称随意，记录好密码
4. 在浏览器中打开http://localhost:7474/browser/，输入用户名neo4j, 刚刚的密码，启动服务
5. 在以下代码中更新用户名和密码
```

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