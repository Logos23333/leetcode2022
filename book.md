按照题的类型，可以分成数据结构题，算法题，脑筋急转弯题。

对于一道新题，我们要做的是快速的分析出其类型，找到适用算法或数据结构，为此，我们需要知道每个算法的适用范围和数据结构的特点。

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

```python
m = {0:0} # 哈希表初始化
total = 0 # 保存前缀和
for idx, num in enumerate(nums):
    total += num
    if target-total in m:
        return True
    m[total] = idx
```
注意：前缀和有些时候需要初始化哈希表，因为我们要考虑nums[:i]的情况，具体如何初始化要看题目。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| 和为 K 的子数组 | Medium | https://leetcode-cn.com/problems/subarray-sum-equals-k/ |
| 统计「优美子数组」 | Medium | https://leetcode-cn.com/problems/count-number-of-nice-subarrays/ |
| 路径总和Ⅲ | Medium | https://leetcode-cn.com/problems/path-sum-iii/ |
| 连续数组 | Medium | https://leetcode-cn.com/problems/contiguous-array/ |
## 链表

对于链表，我们需要知道它和数组相比的优点和缺点。

优点：插入一个元素不需要移动其它元素。

缺点：访问元素需要遍历整个链表。

我们还需要熟练掌握链表节点的增/删/查等。

建议直接做剑指offer的链表题，都十分经典。

| 题目                                                         | 难度   | 链接                                                         |
| :----------------------------------------------------------- | ------ | ------------------------------------------------------------ |
| [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/) | Easy   | https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/ |
| [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/) | Easy   | https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/   |
| [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/) | Easy   | https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/ |
| [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/) | Medium | https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/ |
| [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/) | Medium | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/ |
| [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/) | Easy   | https://leetcode-cn.com/problems/intersection-of-two-linked-lists/ |
| [445. 两数相加 II](https://leetcode-cn.com/problems/add-two-numbers-ii/) | Medium | https://leetcode-cn.com/problems/add-two-numbers-ii/         |

### 快慢指针

链表题中有一部分可以用快慢指针做，比较常见，建议熟练掌握。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/) | Easy   | https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/ |
| [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/) | Medium | https://leetcode-cn.com/problems/linked-list-cycle-ii/       |
|                                                              |        |                                                              |

## 二叉树

### 完全二叉树

完全二叉树常被用来实现堆。用数组实现完全二叉树时有以下性质，idx节点对应的左节点索引为2\*idx + 1，右子树索引为2\*idx + 2。

###  二叉搜索树

二叉搜索树的性质是左节点值小于根节点，右节点值大于根节点。

## 队列
## 栈
### 单调栈
## 队列
# 算法



## 二分搜索

特点：在已排序数组中搜索值target

二分搜索模板：

```python
def search(arr, k):
    n = len(arr)
    i, j = 0, n - 1
    while i<j:
        m = i + (j - i)//2
        if arr[m]>k:
            j = m-1
        elif arr[m]<k:
            i = m+1
        else:
            return m
    return -1
```



| 题目                                                         | 难度 | 链接                                                         |
| ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | Easy | https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/ |
| [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/) | Easy | https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/     |
|                                                              |      |                                                              |



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

堆也被叫做优先队列，常被用来解决“第K大的数”等类型的问题。

堆排模板：

```python
def topDown(arr, i, size):
    l = i*2+1 # 左节点索引
    r = i*2+2 # 右节点索引
    max_idx = i
    if l<size and arr[l]>arr[max_idx]:
        max_idx = l
    if r<size and arr[r]>=arr[max_idx]:
        max_idx = r
    if i!=max_idx:
        arr[i], arr[max_idx] = arr[max_idx], arr[i]
        topDown(arr, max_idx, size) # 递归 topDown


def heapSort(arr):
    for i in range(len(arr)//2-1, -1, -1): # 从最后一个有子节点的节点开始topDown
        topDown(arr, i, len(arr))
    # 此时已经构建了一个最大堆，堆顶为该数组最大值
    for i in range(len(arr)-1, -1, -1):
        arr[0], arr[i] = arr[i], arr[0] # 将最大值交换到数组尾部
        topDown(arr, 0, i) # 保持堆的特性，做一次topDown
```



### 快排

快排的思路是：每次选择一个pivot，移动pivot，直到pivot左边的元素比其小，pivot右边的元素比其大。

快排模板：

```python
def quick(arr, start, end):
    if start>=end:
        return
    
    i, j = start, end
    # 这里我们选择第一个元素作为pivot
    # 随机选择可以降低复杂度
    pivot = arr[start]
    while i<j:
        while i<j and arr[j]>=pivot:
            j-=1
        arr[i] = arr[j]
        while i<j and arr[i]<pivot:
            i+=1
        arr[j] = arr[i]
       
    arr[i] = pivot
    quick(arr, start, i-1)
    quick(arr, i+1, end)
```



### 桶排

## 位运算

# 特定类型题
## 组合/排列
## 子序列
## 字符串匹配

## 二叉树路径和


# 脑筋急转弯题
