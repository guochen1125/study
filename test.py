string = "aabbxllscscs"
# 字符串转顺序计数字典
string_count_dict = {s: string.count(s) for s in sorted(set(string))}
print(string_count_dict)


string = "ababababab"

def KMPtable(string):
    res = []
    for index in range(len(string)):
        s = string[: index + 1]
        prefix = [s[: i + 1] for i in range(len(s) - 1)][::-1]
        suffix = [s[i:] for i in range(1, len(s))]
        val = max([len(p) for p, q in zip(prefix, suffix) if p == q], default=0)
        res.append(val)
    return res

print(KMPtable(string))
