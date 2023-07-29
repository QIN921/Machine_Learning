import numpy as np

# 定义一个数组，每个元素代表一个概率值
probs = np.array([0.2, 0.3, 0.1, 0.25, 0.15])

# 使用np.random.choice()函数根据概率随机选择数组下标
selected_index = np.random.choice(np.arange(len(probs)), p=probs)

print(selected_index)
