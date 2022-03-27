按照题的类型，可以分成数据结构题，算法题，脑筋急转弯题。

对于一道新题，我们要做的是快速的分析出其类型，找到适用算法或数据结构，为此，我们需要知道每个算法的适用范围和数据结构的特点。

# 数据结构
## 哈希表
哈希表的思想在于用空间换时间，它访问Key的时间复杂度为O(1)。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| [1. 两数之和](https://leetcode-cn.com/problems/two-sum/) | Easy | https://leetcode-cn.com/problems/two-sum/ |

### 前缀和
前缀和通常被用于“连续子序列之和/积”类型的题目中，它计算序列的前k个数之和并用哈希表存储。
它的思想是，任意连续子数组nums[i:j]之和都可以用total[j]-total[i]表示。
假设数组为nums，长度为n，我们想知道该数组存不存在和为target的“连续子数组”，用前缀和的伪代码如下：

```python
m = {0:-1} # 哈希表初始化
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
| [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) | Medium | https://leetcode-cn.com/problems/subarray-sum-equals-k/ |
| [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/) | Medium | https://leetcode-cn.com/problems/count-number-of-nice-subarrays/ |
| [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/) | Medium | https://leetcode-cn.com/problems/path-sum-iii/ |
| [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/) | Medium | https://leetcode-cn.com/problems/contiguous-array/ |
#### [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

```python
class Solution(object):
    def subarraySum(self, nums, k):
        n = len(nums)
        total = 0
        mapping = {0:1} # 哈希表初始化
        res = 0
        for num in nums:
            total += num
            res += mapping.get(total-k, 0)
            mapping[total] = mapping.get(total, 0) + 1
        return res
```

#### [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/)

```python
class Solution(object):
    def numberOfSubarrays(self, nums, k):
        n = len(nums)
        total, res = 0, 0
        m = {0: 1}
        for idx, num in enumerate(nums):
            if num%2:
                total+=1
            if total-k in m:
                res += m[total-k]
            m[total] = m.get(total, 0) + 1
        return res
```

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix = {0:1}

        def dfs(root, cur):
            if not root:
                return 0
            
            cur+=root.val # 更新前缀和
            res = prefix.get(cur-targetSum, 0) # 查询当前是否有满足题意的路径

            # dfs搜索
            prefix[cur] = prefix.get(cur, 0) + 1
            res += dfs(root.left, cur)
            res += dfs(root.right, cur)
            prefix[cur] -= 1

            return res
        
        return dfs(root, 0)
```

#### 525. 连续数组

```python
class Solution(object):
    def findMaxLength(self, nums):
        pre_dict = {0: -1}
        ret = pre_sum = 0
        for index, num in enumerate(nums):
            pre_sum += -1 if num == 0 else 1
            if pre_sum in pre_dict:
                ret = max(ret, index - pre_dict[pre_sum])
            else:
                pre_dict[pre_sum] = index
        return ret
```

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

滑动窗口可以分为变长滑动窗口和固定窗口大小的滑动窗口。

滑动窗口的适用范围：右指针向右移动时total一定变小/大，左指针向右移动时total一定变大/小，典型的适用场景是 正整数数组求和，正整数数组求积，注意，一定是正整数数组，如果是整数数组的话，right向右移动并不能保证子数组单调增或单调减，所以滑动窗口此时就并不适用了。

体会二种滑动窗口，一种要找到大于target的最短子数组，一种要找到小于target的最长子数组。

滑动窗口模板：

```python
# 找最短子数组大于target
i,j = 0, 0
total = 0
res = float('inf')
while j<n:
    total += nums[j] # 根据题意，total会有所变化
    while total>target: # 缩小左界，直到不满足题意
        res = min(res, j-i+1)
        total-=nums[i]
        i+=1
    j+=1 # 扩大右界
```

```python
# 找最长子数组小于target
i,j = 0, 0
total = 0
res = float('-inf')
while j<n:
    total += nums[j]
    while total>target: # 缩小左界，直到满足题意
        total-=nums[i]
        i+=1
    if total<target:
        res = max(res, j-i+1)
    j+=1 # 扩大右界
```

在第一种滑动窗口中，因为要寻找最短，所以是通过缩小左界去寻找的，而在第二种滑动窗口中，因为要寻找最长，所以是通过扩大右界去寻找的。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/) | Easy   | https://leetcode-cn.com/problems/maximum-average-subarray-i/ |
| [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/) | Medium | https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/ |
| [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/) | Medium | https://leetcode-cn.com/problems/minimum-size-subarray-sum/  |
|                                                              |        |                                                              |
| [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/) | Hard   | https://leetcode-cn.com/problems/sliding-window-maximum/     |
|                                                              |        |                                                              |
| [713. 乘积小于K的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/) | Medium | https://leetcode-cn.com/problems/subarray-product-less-than-k/ |
| [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/) | Medium | https://leetcode-cn.com/problems/permutation-in-string/      |
|                                                              |        |                                                              |
| [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/) | Hard   | https://leetcode-cn.com/problems/minimum-window-substring/   |
| [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/) | Medium | https://leetcode-cn.com/problems/contains-duplicate-iii/     |
|                                                              |        |                                                              |



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
