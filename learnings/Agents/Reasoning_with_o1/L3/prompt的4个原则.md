1.简单直接；

2.跳过COT；

3.使用Structure；Markdown或者xml的tag

4.Show而不是Tell

![1原则](/Users/zys/Documents/a-Models/63Reasoning with o1/L3/1原则.png)

```python
bad_prompt = ("Generate a function that outputs the SMILES IDs for all the molecules involved in insulin."
              "Think through this step by step, and don't skip any steps:"
              "- Identify all the molecules involve in insulin"
              "- Make the function"
              "- Loop through each molecule, outputting each into the function and returning a SMILES ID"
              "Molecules: ")
response = client.chat.completions.create(model=O1_MODEL,messages=[{"role":"user","content": bad_prompt}])
```

