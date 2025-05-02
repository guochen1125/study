# string = "aabbxllscscs"
# # 字符串转顺序计数字典
# string_count_dict = {s: string.count(s) for s in sorted(set(string))}
# print(string_count_dict)

import numpy as np


a = np.array([[1, 2, 3, 4], [2, 3, 5, 2], [6, 3, 9, 5]])
print(a)
print(np.mean(a, 0))
print(a-np.mean(a, 0).reshape(-1, 4))
