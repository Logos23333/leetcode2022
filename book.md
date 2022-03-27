按照题的类型，可以分成数据结构题，算法题，脑筋急转弯题按照题的类型，可以分成数据结构题，算题脑急转。
# 数据结构
## 哈希表
哈希表的思想在于用空间换时间，它访问Key的时间复杂度为O(1)。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| 两数之和 | Easy | https://leetcode-cn.com/problems/two-sum/ |

### 前缀和
前缀和通常被用于“连续子序列之和/积”类型的题目中，它计算序列的前k个数之和并用哈希表存储。
它的思想是，任意连续子数组nums[i:j]之和都可以用total[j]-total[i]表示。
假设数组为nums，长度为n，我们想知道该数组存不存在和为target的“连续子数组”，用前缀和的伪代码如下：
```
m = {}
total = 0 # 保存前缀和
for num in nums:
    total += num
    if target-total in m:
        return True
    m[total] =True
```
注意：前缀和有些时候需要初始化哈希表，因为我们要考虑nums[:i]的情况，具体如何初始化要看题目。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| 和为 K 的子数组 | Medium | https://leetcode-cn.com/problems/subarray-sum-equals-k/ |
| 统计「优美子数组」 | Medium | https://leetcode-cn.com/problems/count-number-of-nice-subarrays/ |
| 路径总和Ⅲ | Medium | https://leetcode-cn.com/problems/path-sum-iii/ |
| 连续数组 | Medium | https://leetcode-cn.com/problems/contiguous-array/ |
## 链表

## 队列
## 栈
### 单调栈
## 队列
## 二叉树

# 算法
## 二分搜索
## 双指针
## 滑动窗口
## dfs
### 备忘录
### 剪枝
### 博弈论
## 动态规划
### 状态压缩
## 排序
### 堆排
### 快排
### 桶排

## 位运算

# 特定类型题
## 组合/排列
## 子序列
## 字符串匹配


# 脑筋急转弯题
