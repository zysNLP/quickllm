论文来源：ATC



两个创新点：1. FP6；2.内核

背景：模型参数量越来越大，保证准确率的情况下压缩内存占用。当前主流压精度的方法是只压权重，不压缩激活值，这种能解决99%的权重压缩问题。

目前一些硬件只支持4bit和8bit压缩（哪些？）

评估使用两个指标：1.zero-shot perplexity；越低越好。2.代码生成pass@1

压缩后介于FP16和INT4之间，没有FP16那么多参数，但是比INT4多

FP6和FP16效果差不多，远远优于INT4；

LLaMa70B的DRAM使用率从FP16的130G压缩到49G；LLaMa70B的GPU从FP16的2个GPU压缩到使用一个GPU。token生成速度比FP16和int8分别提升2.4和1.8倍。



![image-20250713101108743](/Users/sunday/Library/Application Support/typora-user-images/image-20250713101108743.png)

![image-20250713101510754](/Users/sunday/Library/Application Support/typora-user-images/image-20250713101510754.png)

![image-20250713102331046](/Users/sunday/Library/Application Support/typora-user-images/image-20250713102331046.png)

![image-20250713104311782](/Users/sunday/Library/Application Support/typora-user-images/image-20250713104311782.png)

![image-20250713112934687](/Users/sunday/Library/Application Support/typora-user-images/image-20250713112934687.png)

![image-20250713120240289](/Users/sunday/Library/Application Support/typora-user-images/image-20250713120240289.png)想·