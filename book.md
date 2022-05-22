按照题的类型，可以分成数据结构题，算法题，脑筋急转弯题。

对于一道新题，我们要做的是快速的分析出其类型，找到适用算法或数据结构，为此，我们需要知道每个算法的适用范围和数据结构的特点。

- [数据结构](#数据结构)
  - [哈希表](#哈希表)
      - [1. 两数之和](#1-两数之和)
    - [前缀和](#前缀和)
      - [560. 和为 K 的子数组](#560-和为-k-的子数组)
      - [1248. 统计「优美子数组」](#1248-统计优美子数组)
      - [437. 路径总和 III](#437-路径总和-iii)
      - [525. 连续数组](#525-连续数组)
  - [链表](#链表)
    - [快慢指针](#快慢指针)
  - [二叉树](#二叉树)
    - [完全二叉树](#完全二叉树)
    - [二叉搜索树](#二叉搜索树)
      - [剑指 Offer 54. 二叉搜索树的第k大节点](#剑指-offer-54-二叉搜索树的第k大节点)
      - [剑指 Offer 33. 二叉搜索树的后序遍历序列](#剑指-offer-33-二叉搜索树的后序遍历序列)
      - [剑指 Offer 36. 二叉搜索树与双向链表](#剑指-offer-36-二叉搜索树与双向链表)
    - [遍历二叉树](#遍历二叉树)
      - [105. 从前序与中序遍历序列构造二叉树](#105-从前序与中序遍历序列构造二叉树)
      - [剑指 Offer 32 - I. 从上到下打印二叉树](#剑指-offer-32---i-从上到下打印二叉树)
      - [剑指 Offer 32 - II. 从上到下打印二叉树 II](#剑指-offer-32---ii-从上到下打印二叉树-ii)
      - [剑指 Offer 32 - III. 从上到下打印二叉树 III](#剑指-offer-32---iii-从上到下打印二叉树-iii)
      - [剑指 Offer 37. 序列化二叉树](#剑指-offer-37-序列化二叉树)
    - [路径总和题](#路径总和题)
      - [112. 路径总和](#112-路径总和)
      - [113. 路径总和 II](#113-路径总和-ii)
      - [437. 路径总和 III](#437-路径总和-iii-1)
      - [剑指 Offer II 051. 节点之和最大的路径](#剑指-offer-ii-051-节点之和最大的路径)
    - [其它题](#其它题)
      - [剑指 Offer 55 - I. 二叉树的深度](#剑指-offer-55---i-二叉树的深度)
      - [剑指 Offer 55 - II. 平衡二叉树](#剑指-offer-55---ii-平衡二叉树)
      - [剑指 Offer 26. 树的子结构](#剑指-offer-26-树的子结构)
      - [剑指 Offer 27. 二叉树的镜像](#剑指-offer-27-二叉树的镜像)
      - [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](#剑指-offer-68---i-二叉搜索树的最近公共祖先)
      - [剑指 Offer 68 - II. 二叉树的最近公共祖先](#剑指-offer-68---ii-二叉树的最近公共祖先)
  - [队列](#队列)
  - [栈](#栈)
    - [单调栈](#单调栈)
  - [队列](#队列-1)
- [算法](#算法)
  - [二分搜索](#二分搜索)
      - [35. 搜索插入位置](#35-搜索插入位置)
      - [34. 在排序数组中查找元素的第一个和最后一个位置](#34-在排序数组中查找元素的第一个和最后一个位置)
      - [剑指 Offer 53 - II. 0～n-1中缺失的数字](#剑指-offer-53---ii-0n-1中缺失的数字)
      - [33. 搜索旋转排序数组](#33-搜索旋转排序数组)
      - [81. 搜索旋转排序数组 II](#81-搜索旋转排序数组-ii)
      - [153. 寻找旋转排序数组中的最小值](#153-寻找旋转排序数组中的最小值)
      - [154. 寻找旋转排序数组中的最小值 II](#154-寻找旋转排序数组中的最小值-ii)
      - [69. x 的平方根](#69-x-的平方根)
  - [双指针](#双指针)
  - [滑动窗口](#滑动窗口)
  - [dfs](#dfs)
    - [博弈论](#博弈论)
    - [状态压缩](#状态压缩)
      - [526. 优美的排列](#526-优美的排列)
      - [698. 划分为k个相等的子集](#698-划分为k个相等的子集)
  - [动态规划](#动态规划)
    - [路径问题](#路径问题)
      - [62. 不同路径](#62-不同路径)
      - [63. 不同路径 II](#63-不同路径-ii)
      - [64. 最小路径和](#64-最小路径和)
      - [120. 三角形最小路径和](#120-三角形最小路径和)
      - [931. 下降路径最小和](#931-下降路径最小和)
      - [1289. 下降路径最小和  II](#1289-下降路径最小和--ii)
      - [1575. 统计所有可行路径](#1575-统计所有可行路径)
      - [576. 出界的路径数](#576-出界的路径数)
      - [1301. 最大得分的路径数目](#1301-最大得分的路径数目)
    - [0-1背包问题模板](#0-1背包问题模板)
      - [dp[N\]\[V\+1]模板](#dpnv1模板)
      - [dp[2\]\[V\+1]模板](#dp2v1模板)
      - [⭐dp\[V\+1]模板](#dpv1模板)
    - [0-1背包问题](#0-1背包问题)
      - [416. 分割等和子集](#416-分割等和子集)
    - [完全背包问题模板](#完全背包问题模板)
      - [dpN模板](#dpn模板)
      - [dp2模板](#dp2模板)
      - [dp[V+1\]模板](#dpv1模板-1)
      - [⭐dp[V+1\]模板(优化)](#dpv1模板优化)
    - [完全背包问题](#完全背包问题)
      - [279. 完全平方数](#279-完全平方数)
      - [322. 零钱兑换](#322-零钱兑换)
      - [518. 零钱兑换 II](#518-零钱兑换-ii)
  - [排序](#排序)
    - [堆排](#堆排)
    - [快排](#快排)
    - [桶排](#桶排)
  - [位运算](#位运算)
      - [191. 位1的个数](#191-位1的个数)
      - [260. 只出现一次的数字 III](#260-只出现一次的数字-iii)
      - [剑指 Offer 65. 不用加减乘除做加法](#剑指-offer-65-不用加减乘除做加法)
      - [29. 两数相除](#29-两数相除)
- [特定类型题](#特定类型题)
  - [组合/排列](#组合排列)
  - [子序列](#子序列)
  - [字符串匹配](#字符串匹配)
  - [二叉树路径和](#二叉树路径和)
  - [模拟题](#模拟题)
- [脑筋急转弯题](#脑筋急转弯题)

# 数据结构
## 哈希表
哈希表的思想在于用空间换时间，它访问Key的时间复杂度为$O(1)$。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| [1. 两数之和](https://leetcode-cn.com/problems/two-sum/) | Easy | https://leetcode-cn.com/problems/two-sum/ |

#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mapping = {}
        for idx, num in enumerate(nums):
            if target-num in mapping:
                return [mapping[target-num], idx]
            mapping[num] = idx
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

### 前缀和

前缀和通常被用于“连续子序列之和/积”类型的题目中，它计算序列的前k个数之和并用哈希表存储。
它的思想是，任意连续子数组`nums[i:j]`之和都可以用`total[j]-total[i]`表示。
假设数组为`nums`，长度为`n`，我们想知道该数组存不存在和为target的“连续子数组”，用前缀和的模板如下：

```python
m = {0:-1} # 哈希表初始化
total = 0 # 当前前缀和
for idx, num in enumerate(nums):
    total += num
    if target-total in m:
        return True
    m[total] = idx
```
注意：前缀和有些时候需要初始化哈希表，因为我们要考虑`nums[:i]`的情况，具体如何初始化要看题目。

| 题目 | 难度 | 链接 |
| --- | --- | --- |
| [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/) | Medium | https://leetcode-cn.com/problems/subarray-sum-equals-k/ |
| [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/) | Medium | https://leetcode-cn.com/problems/count-number-of-nice-subarrays/ |
| [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/) | Medium | https://leetcode-cn.com/problems/path-sum-iii/ |
| [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/) | Medium | https://leetcode-cn.com/problems/contiguous-array/ |
#### [560. 和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

思路：任意连续子数组nums[i:j]之和都可以用total[j]-total[i]表示。

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

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [1248. 统计「优美子数组」](https://leetcode-cn.com/problems/count-number-of-nice-subarrays/)

思路：哈希表`(i, number)`记录有`i`个奇数数字的数组个数为`number`。应用了前缀和后，这道题就变成了[1. 两数之和](https://leetcode-cn.com/problems/two-sum/)。

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

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

思路：任意路径之和都可以用前缀和之差表示。

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix = {0:1}

        def dfs(root, cur):
            if not root:
                return 0
            
            cur+=root.val # 当前前缀和
            res = prefix.get(cur-targetSum, 0) # 查询当前是否有满足题意的路径
            prefix[cur] = prefix.get(cur, 0) + 1 # 更新哈希表

            # dfs搜索
            res += dfs(root.left, cur)
            res += dfs(root.right, cur)
            prefix[cur] -= 1

            return res
        
        return dfs(root, 0)
```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(n)$

#### [525. 连续数组](https://leetcode-cn.com/problems/contiguous-array/)

思路：`total[i]`表示列表`[:i+1]`中0和1的数量。若`total[j]`和`total[i]`的值一样，则代表`nums[i+1:j+1]`0和1的数量相同（互相抵消）。

哈希表记录的是total对应的索引`index`，要注意哈希表的初始化，记录索引的时候要初始化为`-1`。

```python
class Solution(object):
    def findMaxLength(self, nums):
        m = {0: -1}
        res, total = 0, 0
        for index, num in enumerate(nums):
            total += -1 if num == 0 else 1
            if total in m:
                res = max(res, index - m[total])
            else:
                m[total] = index
        return res
```

#### [304. 二维区域和检索 - 矩阵不可变](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/)

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0 for i in range(n+1)] for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                self.prefix[i][j] = matrix[i-1][j-1] + self.prefix[i-1][j] + self.prefix[i][j-1] - self.prefix[i-1][j-1]


    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        row1, col1, row2, col2 = row1+1, col1+1, row2+1, col2+1
        return self.prefix[row2][col2] - self.prefix[row2][col1-1] - self.prefix[row1-1][col2] + self.prefix[row1-1][col1-1]
```

时间复杂度：$O(m*n)$，检索的时间复杂度为$O(1)$

空间复杂度：$O(m*n)$

#### [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

```python
class NumArray:

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.prefix = [0 for i in range(n+1)]
        for i in range(1, n+1):
            self.prefix[i] = self.prefix[i-1] + nums[i-1]


    def sumRange(self, left: int, right: int) -> int:
        return self.prefix[right+1]-self.prefix[left]
```

时间复杂度：$O(n)$，检索的时间复杂度为$O(1)$

空间复杂度：$O(n)$

#### [523. 连续的子数组和](https://leetcode-cn.com/problems/continuous-subarray-sum/)

这题有两个地方和普通前缀和不一样：

1. 乍一看用不了前缀和，但分析可知要使得连续子数组和为k的倍数，就是使sum[j]和sum[i]余k相同
2. 对连续子数组长度有要求，所以我们要保存最前方的值（旧值），而不是新值

```python
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        mapping = {0:-1}
        total = 0
        for idx, num in enumerate(nums):
            total += num
            target = total%k
            lst_idx = mapping.get(target, idx)
            if lst_idx == idx:
                mapping[target] = idx
            elif idx-lst_idx>=2:
                return True
        return False
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

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

#### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        # find mid
        fast, slow = head.next, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # cut
        mid, slow.next = slow.next, None 
        left, right = self.sortList(head), self.sortList(mid)

        # merge
        new_head = ListNode(-1)
        p, p1, p2 = new_head, left, right
        while p1 and p2:
            if p1.val<p2.val:
                p.next, p1 = p1, p1.next
            else:
                p.next, p2 = p2, p2.next
            p = p.next
        p.next = p1 if p1 else p2
        return new_head.next
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(logn)$，递归的空间开销为$O(logn)$

#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p1, p2 = headA, headB
        while p1!=p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return p1
```

时间复杂度：$O(m+n)$

空间复杂度：$O(1)$

#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur = head
        last = None
        while cur:
            tmp = cur.next
            cur.next = last
            last = cur
            cur = tmp
        return last
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$



## 二叉树

### 完全二叉树

完全二叉树常被用来实现堆。用数组实现完全二叉树时有以下性质，idx节点对应的左节点索引为2\*idx + 1，右子树索引为2\*idx + 2。

###  二叉搜索树

二叉搜索树的性质是左节点值小于父节点，右节点值大于父节点。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/) | Easy   | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/ |
| [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | Medium | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/ |
| [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/) | Medium | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/ |
|                                                              |        |                                                              |

#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```python
class Solution(object):
    def kthLargest(self, root, k):
        def dfs(root):
            if not root:
                return
            dfs(root.right) # 注意这里是先右再左，因为是返回第k大而不是第k小
            self.k-=1
            if self.k==0: self.res = root.val
            dfs(root.left)
        
        self.res = 0
        self.k = k
        dfs(root)
        return self.res
```

时间复杂度：$O(n)$。

空间复杂度：$O(n)$。

#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        n = len(postorder)
        if n<3:
            return True
        root = postorder[-1]
        # 找到第一个比root大的值，此值左边为左子树，右边为右子树，递归判断即可
        idx = 0
        right_idx = -1
        while idx<n-1:
            if right_idx!=-1: # 此时已经找到第一个比root大的值
                if postorder[idx]<root: # 它右边还是有比root小的，必定不满足二叉搜索树
                    return False
            if postorder[idx]>root:
                right_idx = idx
                break
            idx+=1
        return self.verifyPostorder(postorder[:idx]) and self.verifyPostorder(postorder[idx:-1])
```

时间复杂度：$O(n^2)$。每次都要遍历整个数组，最坏的情况下要遍历n次。

空间复杂度：$O(n)$。压栈也会占用空间。

#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

首先需要知道一点：二叉搜索树中序遍历后可以得到一个排序数组。

这题有两种解法，一是用一个全局指针保存上一个递归的尾节点，二是dfs时同时返回头节点和尾节点。

解法1：

用全局指针`self.pre`保存上一次递归的尾节点。

```python
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(cur):
            if not cur: return
            dfs(cur.left) # 递归左子树
            if self.pre: # 修改节点引用
                self.pre.right, cur.left = cur, self.pre
            else: # 记录头节点
                self.head = cur
            self.pre = cur # 保存 cur
            dfs(cur.right) # 递归右子树
        
        if not root: return
        self.pre = None
        dfs(root)
        self.head.left, self.pre.right = self.pre, self.head
        return self.head

作者：jyd
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/solution/mian-shi-ti-36-er-cha-sou-suo-shu-yu-shuang-xian-5/
```

时间复杂度：$O(n)$，每个节点都需要访问一次。

空间复杂度：$O(h)$，其中h为树的高度，也就是递归时栈的开销。

解法2：

dfs时同时返回头节点和尾节点。

```python
class Solution(object):
    def treeToDoublyList(self, root):
        if not root:
            return None
        
        def dfs(root):
            # 返回值, (head, tail)
            if not root:
                return None, None
            if not root.left and not root.right: # 叶节点
                return root, root

            left_head, left_tail = dfs(root.left) # 递归得到左子树的头节点和尾节点
            right_head, right_tail = dfs(root.right) # 递归得到右子树的头节点和尾节点
            
            # 构建双向链表
            if right_head:
                right_head.left = root
            if left_tail:
                left_tail.right = root
            root.left = left_tail
            root.right = right_head
            
            head = left_head if left_head else root
            tail = right_tail if right_tail else root
            return head, tail
        
        head, tail = dfs(root)
        head.left = tail
        tail.right = head
        return head
```

时间复杂度：$O(n)$，每个节点都需要访问一次。

空间复杂度：$O(h)$，其中h为树的高度，也就是递归时栈的开销。

#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

一棵$n$个节点的二叉搜索树，在确定了根节点$i$之后，其左侧有$i-1$个节点，右侧有$n-i$个节点，则其排列方式共有

$G(i) = G(i-1)*G(n-i)$种，每个节点都可以作为根节点，对其进行求和即可得到答案。

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0 for i in range(max(n+1, 3))]
        dp[0], dp[1], dp[2] = 1, 1, 2
        for i in range(3, n+1):
            for j in range(1, i+1):
                dp[i] += dp[j-1]*dp[i-j]
        return dp[n]
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:     
        def dfs(root, cur_min, cur_max):
            if not root:
                return True
            
            if root.val<=cur_min or root.val>=cur_max:
                return False

            left_val = root.left.val if root.left else float('-inf')
            right_val = root.right.val if root.right else float('inf')

            return root.val>left_val and root.val<right_val and dfs(root.left, cur_min, root.val) and dfs(root.right, root.val, cur_max)
        
        return dfs(root, float('-inf'), float('inf'))
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:     
        def dfs(root):
            if not root:
                return
            
            dfs(root.left)
            if root.val<=self.pre:
                self.res = False
                return
            else:
                self.pre = root.val
            dfs(root.right)

        self.pre = float('-inf')
        self.res = True
        dfs(root)
        return self.res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$



### 遍历二叉树

需要熟练掌握二叉树的前序，中序，后序，层次遍历

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | Medium | https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ |
| [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/) | Easy   | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/ |
| [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/) | Easy   | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/ |
| [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/) | Medium | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/ |
| [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/) |        |                                                              |

#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

思路：前序遍历root节点必定是第一个，以此找到中序遍历数组中的root节点，并划分左右子树，递归调用即可。

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        if len(preorder) == 0:
            return None
        root = preorder[0]
        left_length = inorder.index(root) # 左子树长度，这里可以先用哈希表存储每个节点对应的索引，用空间换时间。
        root_node = TreeNode(root)
        root_node.left = self.buildTree(preorder[1:1+left_length], inorder[:left_length])
        root_node.right = self.buildTree(preorder[1+left_length:], inorder[left_length+1:])
        return root_node
```

时间复杂度：$O(n^2)$。`n`为树中节点的个数。如果使用哈希表，则时间复杂度为$O(n)$。

空间复杂度：$O(h)$。`h`为树的高度。如果使用哈希表，空间复杂度为$O(n)$。

#### [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        queue = deque([root]) # 双向队列deque popleft的时间复杂度为O(1)，数组pop(0)的时间复杂度为O(n)
        res = []
        while queue:
            cur = queue.popleft()
            res.append(cur.val)
            if cur.left:
                queue.append(cur.left)
            if cur.right:
                queue.append(cur.right)
        return res
```

时间复杂度：$O(n)$。`n`为树中节点的个数。每个节点都需要出队入队一次。

空间复杂度：$O(n)$。若二叉树为满二叉树，最后一层的节点个数为`n/2`。

#### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            tmp = []
            queue_len = len(queue)
            for i in range(queue_len):
                cur = queue.popleft()
                tmp.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(tmp)
        return res
```

时间复杂度：$O(n)$。`n`为树中节点的个数。每个节点都需要出队入队一次。

空间复杂度：$O(n)$。若二叉树为满二叉树，最后一层的节点个数为`n/2`。

#### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            tmp = []
            queue_len = len(queue)
            for i in range(queue_len):
                cur = queue.popleft()
                tmp.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            res.append(tmp[::-1])
        return res
```

时间复杂度：$O(n)$。`n`为树中节点的个数。每个节点都需要出队入队一次。倒序操作的时间复杂度也为$O(n)$，因为每个节点都只需要倒一次。

空间复杂度：$O(n)$。若二叉树为满二叉树，最后一层的节点个数为`n/2`。

#### [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

```python
class Codec:
    def serialize(self, root):
        # 层次化遍历二叉树
        if not root:
            return []
        queue = deque([root]) 
        res = []
        while queue:
            cur = queue.popleft()
            if cur:
                res.append(str(cur.val))
            else:
                res.append('None')
                continue

            queue.append(cur.left)
            queue.append(cur.right)

        return ','.join(res).rstrip(',None')

    def deserialize(self, data):
        if not data:
            return None
        nodes = data.split(',')
        root = TreeNode(int(nodes[0]))
        queue = deque([root])
        i = 1
        # 每次popleft出一个，i走两步，保证第一步为左节点，第二步为右节点
        while queue and i<len(nodes):
            cur = queue.popleft()
            if nodes[i]!='None':
                cur.left = TreeNode(int(nodes[i]))
                queue.append(cur.left)
            i+=1
            if i>=len(nodes): break
            if nodes[i]!='None':
                cur.right = TreeNode(int(nodes[i]))
                queue.append(cur.right)
            i+=1
        return root
```

时间复杂度：$O(n)$。`n`为树中节点的个数。每个节点都需要出队入队一次。

空间复杂度：$O(n)$。



### 路径总和题

同样都是找出路径和等于`target`，但是当路径的定义发生变化，解法也随之发生变化。

考虑以下四种路径定义：

1. 路径只能由根节点出发，叶节点结束。（[112. 路径总和](https://leetcode-cn.com/problems/path-sum/)，[113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)）
2. 路径由根节点出发，任意节点都能结束，但只能由父节点到子节点。
3. 路径不需要从根节点出发，也不需要在叶子节点结束，但只能由父节点到子节点。（[437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)）
4. 路径不需要从根节点出发，也不需要在叶子节点结束，可以从子节点到父节点，但是同一节点只能出现一次。（[剑指 Offer II 051. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/))

| 题目                                                         | 难度   | 链接                                           |
| ------------------------------------------------------------ | ------ | ---------------------------------------------- |
| [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)  | Easy   | https://leetcode-cn.com/problems/path-sum/     |
| [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/) | Medium | https://leetcode-cn.com/problems/path-sum-ii/  |
| [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/) | Medium | https://leetcode-cn.com/problems/path-sum-iii/ |
| [剑指 Offer II 051. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/) | Hard   | https://leetcode-cn.com/problems/jC7MId/       |

#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False

        def dfs(root, target):         
            if not root.left and not root.right:
                return target==root.val
            if root.left:
                if dfs(root.left, target-root.val):
                    return True
            if root.right:
                if dfs(root.right, target-root.val):
                    return True 
            return False
        
        return dfs(root, targetSum)
```

时间复杂度：$O(n)$，每个节点都需要访问一次。

空间复杂度：$O(h)$，其中`h`为树的高度，也就是递归时栈的开销。

#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if not root:
            return []
        
        self.res = []
        def dfs(root, path, cur_sum):
            if not root:
                return
            if not root.left and not root.right:
                if cur_sum + root.val == targetSum:
                    path.append(root.val)
                    self.res.append(path)
                return
            
            if root.left:
                dfs(root.left, path+[root.val], cur_sum+root.val)
            if root.right:
                dfs(root.right, path+[root.val], cur_sum+root.val)
        
        dfs(root, [], 0)
        return self.res
```

时间复杂度：$O(n^2)$，其中n是树的节点数。在最坏情况下，树的上半部分为链状，下半部分为完全二叉树，此时，路径的数目为 $O(n)$，并且每一条路径的节点个数也为 $O(n)$，因此要将这些路径全部添加进答案中，时间复杂度为 $O(n^2)$。

空间复杂度：$O(h)$，`h`为二叉树的高度。

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

思路：任意路径之和都可以用前缀和之差表示。

```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix = {0:1}

        def dfs(root, cur):
            if not root:
                return 0
            
            cur+=root.val # 当前前缀和
            res = prefix.get(cur-targetSum, 0) # 查询当前是否有满足题意的路径
            prefix[cur] = prefix.get(cur, 0) + 1 # 更新哈希表

            # dfs搜索
            res += dfs(root.left, cur)
            res += dfs(root.right, cur)
            prefix[cur] -= 1

            return res
        
        return dfs(root, 0)
```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(n)$

#### [剑指 Offer II 051. 节点之和最大的路径](https://leetcode-cn.com/problems/jC7MId/)

注意这里的dfs，返回值是以root节点为顶点的单边路径最大和，它和题目要求的路径和是不一样的。我们之所以要返回前者，是因为对于父节点来说，它只需要前者，所以我们在返回前者的同时，更新全局变量`res`。

做完这题可以去做做[剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:    
        self.res = float('-inf')
        # dfs: 搜索以root节点为顶点的最大路径和
        def dfs(root):
            if not root:
                return 0
            
            if not root.left and not root.right:
                self.res = max(self.res, root.val)
                return root.val
            
            # 如果为负数的话还不如不要
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)

            cur_max = left + right + root.val # 题目要求的路径和为 左边单边最大和+根节点+右边单边最大和
            self.res = max(self.res, cur_max) # 更新全局变量res

            return max(left+root.val, right+root.val) # 返回值：以root节点为顶点的单边最大路径和   
            
        dfs(root)
        return self.res
```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(h)$，`h`为二叉树的高度。

### 其它题

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/) | Easy   | https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/ |
| [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/) | Easy   | https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/  |
| [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/) | Medium | https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/     |
| [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/) | Easy   | https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/ |
| [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/) | Easy   | https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/ |
| [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/) | Easy   | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/submissions/ |
| [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/) | Medium | https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/ |

#### [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

二叉树的深度等于左子树和右子树深度最大值+1。

```python
class Solution(object):
    def maxDepth(self, root):
        def dfs(root):
            if not root:
                return 0
            return max(dfs(root.left), dfs(root.right)) + 1
        
        return dfs(root)
```

#### [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

分析题不难知道该二叉树平衡的充分必要条件是，左子树是平衡二叉树，右子树是平衡二叉树，且左子树右子树深度之差小于2。

可以写出以下代码：

```python
def isBalanced(root):
    def depth(root):
        if not root:
            return 0
        return max(dfs(root.left), dfs(root.right)) + 1
    left = depth(root.left)
    right = depth(root.right)
    if abs(left-right)<2 and isBalanced(root.left) and isBalanced(root.right):
        return True
   	return False
```

这时我们会发现一个问题，我们在计算根节点的左右子树深度时，其实就已经知道了左子树的子树的深度，但是我们并没有利用到这一信息，换言之，上面代码进行了重复搜索。

优化1：

用一个全局变量res存放子树是否为平衡二叉树，这样就不用重复搜索了

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True
        def depth(root):
            if not root:
                return 0
            left = depth(root.left)
            right = depth(root.right)
            if abs(left-right)>1:
                self.res = False
            return max(left, right) + 1

        self.res = True
        left_depth = depth(root.left)
        right_depth = depth(root.right)
        return True if self.res and abs(left_depth-right_depth)<=1 else False
```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(h)$，`h`为二叉树的高度。

优化2：

用-1返回值标识不平衡的情况。

```python
class Solution(object):
    def isBalanced(self, root):
        if not root:
            return True
        
        def depth(root):
            if not root:
                return 0
            
            left = depth(root.left)
            right = depth(root.right)
            if left==-1 or right==-1 or abs(left-right)>1:
                return -1
            else:
                return max(left, right)+1
        
        return depth(root)>=0

```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(h)$，`h`为二叉树的高度。

#### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

 注意这题，B是A的子树和B是A的子结构是不一样的，具体区别是：子树要求一定要到达叶节点，而子结构不一定。

用递归的时候一定要先弄清楚，函数的具体含义和返回值，比如这里的`isSubStructure`是判断B是否为A的子结构（B可以出现在A的子树中），`isSub`则一定要“以B节点为root节点”的树 是 “以A节点为root节点”的树的子结构。

```python
class Solution(object):
    def isSubStructure(self, A, B):
       
        if not B or not A:
            return False
        
        def isSub(A, B):
            if not B:
                return True
            if not A:
                return False
            if A.val != B.val:
                return False
            else:
                return isSub(A.left, B.left) and isSub(A.right, B.right)
        
        if isSub(A, B):
            return True
        
        return self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B) 
```

时间复杂度：$O(m*n)$，其中`m`为A的节点数量，`n`为B的节点数量，最坏情况下，对于A的每个节点，都要进行`n`次比较。

空间复杂度：$O(m)$，栈的深度，最坏情况下为`m`。

#### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        if not root.left and not root.right:
            return root

        left = root.left
        right = root.right
        root.left = self.mirrorTree(right)
        root.right = self.mirrorTree(left)

        return root
```

时间复杂度：$O(n)$。每个节点都需要访问一次。

空间复杂度：$O(h)$，`h`为二叉树的高度。

[剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

一棵树对称的充分必要条件是，它的左子树和右子树互为镜像。

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        def isMirror(A, B):
            if not A and not B:
                return True
            if not A or not B:
                return False
            
            if A.val == B.val:
                return isMirror(A.left, B.right) and isMirror(A.right, B.left)
            return False
        
        return isMirror(root.left, root.right)

```

时间复杂度：$O(n)$。最坏情况下，左子树的每个节点都需要和右子树的每个节点比较一次，也即比较`n/2`次。

空间复杂度：$O(h)$，`h`为二叉树的高度。

#### [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

注意这题是二叉搜索树，可以利用二叉搜索树的性质求解，如果p,q在root的左右两边，那么直接返回root即可。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(root, p, q):
            if root.val==p.val or root.val==q.val or (root.val>p.val and root.val<q.val):
                return root
            
            # p<q
            if q.val<root.val:
                return dfs(root.left, p, q)
            if root.val<p.val:
                return dfs(root.right, p, q)

        return dfs(root, p, q) if p.val<q.val else dfs(root, q, p)
```

#### [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

这题的dfs其实挺巧妙的，值得回顾。

这个dfs，如果是还没有递归判断（也就是第一行），出现了`root==p or root==q`，那么root也就是最近公共祖先。
但是如果p,q各自在左右子树时，这时的dfs返回值其实不是最近公共祖先，而是p或q，并不是严格意义上的dfs，只是可以巧妙的判断，如果left 和right同时存在的话，即p,q分居root左右，那么此时应该返回root。

如果p, q在同一边，这时候才是严格意义上的dfs，也即返回了子树的最近公共祖先。

```python
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root==p or root==q:
            return root
        left =self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return None
        if left and not right:
            return left
        if right and not left:
            return right
        return root 

```

#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        def dfs(root):
            if not root:
                return None
            
            root.left, root.right = dfs(root.right), dfs(root.left)
            return root
        
        return dfs(root)
```

时间复杂度：$O(n)$

空间复杂度：$O(h)$

## 队列
## 栈
### 单调栈
## 队列
# 算法



## 二分搜索

特点：在已排序数组中搜索值target

二分搜索模板：

```python
def binarySearch(arr, target):
    i, j = 0, len(arr)
    while i<j:
        m = i + (j-i)//2
        if arr[m]>=target: # 找大于等于target的下界
            j = m            
        else:
            i = m+1
    return i
```

有几点需要注意：

1. 上面的二分搜索是`左闭右开`，即搜索`[i, j)`，i,j分别初始化为0和数组长度。
2. 计算middle时，python虽然不会整数溢出，但也要保证长度为1时，left和right的middle落在[left, right)区间。
3. 若数组有target，上面的二分搜索返回的是`大于等于target的下界`，比如`[1,2,2,3], 2`返回的是`1`（即第一个2对应的位置）。如果不存在target，返回的是`大于target的下界`，比如`[1,2,2,4], 3`返回的是`3`（即第一个4对应的位置）。
4. 如果需要找`等于target的上界`，即最后一个target对应的位置，只需要`binarySearch(arr, target+1)-1`即可。比如`[1,2,2,4], 2`，我们想找到最后一个2，应该这么调用 `ans = binarySearch([1,2,2,4], 2+1) - 1`。最后返回的是`2`。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/) | Easy   | https://leetcode-cn.com/problems/search-insert-position/     |
| [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | Easy   | https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/ |
| [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/) | Easy   | https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/     |
| [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/) | Medium | https://leetcode-cn.com/problems/search-in-rotated-sorted-array/ |
| [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/) | Medium | https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/ |
| [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/) | Medium | https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/ |
| [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/) | Hard   | https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/ |
| [69. x 的平方根 ](https://leetcode-cn.com/problems/sqrtx/)   | Easy   | https://leetcode-cn.com/problems/sqrtx/                      |
| [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/) | Medium | https://leetcode-cn.com/problems/search-a-2d-matrix/         |

#### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

套模板直接秒

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        i, j = 0, len(nums)
        while i<j:
            m = i + (j-i)//2
            if nums[m]<target:
                i = m+1
            else:
                j = m
        return i
```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

直接套模板即可。

```python
class Solution(object):
    def searchRange(self, nums, target):
        def binarySearch(arr, target):
            i, j = 0, len(arr)
            while i<j:
                m = i + (j-i)//2
                if arr[m]<target:
                    i = m+1
                else:
                    j = m
            return i
        
        start = binarySearch(nums, target)
        end = binarySearch(nums, target+1)-1
        if start>end: # 不存在target
            return [-1, -1]
        else:
            return [start, end]
```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

思路：若`nums[i]>i`,说明该数字小于`i`,若`nums[i]==i`,说明该数字在`i`右边，仍然保持左闭右开

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:   
        n = len(nums)
        i, j = 0, len(nums)
        while i<j:
            m = i + (j-i)//2
            if nums[m]>m:
                j=m
            elif nums[m]==m:
                i=m+1
        
        return i
```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

对数组二分，直到target在有序数组中，再进行二分查找。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:     
        
        def binarySearch(nums, l, r, target):
            if l>=r:
                return l
            while l<r:
                m = l+(r-l)//2
                if nums[m]<target:
                    l = m+1                  
                else:
                    r = m # 左开右闭
            return -1 if nums[l]!=target else l
        
        n = len(nums)
        i, j = 0, n
        while i<j:
            m = i + (j-i)//2
            if nums[m]>nums[i]: #[i, m]为有序数组
                if nums[i]<=target<=nums[m]: # target在有序数组中，直接二分
                    return binarySearch(nums, i, m+1, target)
                else:
                    i=m+1
            else: #[m, j]为有序数组
                if nums[m]<=target<=nums[j-1]:
                    return binarySearch(nums, m, j, target)
                else:
                    j=m # 左开右闭
                
        return -1 #没找到
```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

#### [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

这题和前一题的区别在于，前者保证无重复元素，这就导致了当`nums[m]>=nums[i]`的时候，仍然无法判断`[i,m]`或`[m,j]`的有序性，遇到`nums[i]=nums[j]=nums[m]`的时候直接`i+=1, j-=1`即可。如果不满足三者同时相等，是可以判断出左右的有序性的。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
                        
        def binarySearch(nums, l, r, target):
            if l>=r:
                return l
            while l<r:
                m = l+(r-l)//2
                if nums[m]<target:
                    l = m+1                  
                else:
                    r = m
            return False if nums[l]!=target else True

        n = len(nums)
        i, j = 0, n
        while i<j:
            m = i + (j-i)//2
            if nums[m]==target:
                return True
            if nums[m]==nums[i] and nums[m]==nums[j-1]:
                i+=1
                j-=1
            elif nums[m]>=nums[i]: #[i, m]为有序数组
                if nums[i]<=target<=nums[m]: # target在有序数组中，直接二分
                    return binarySearch(nums, i, m+1, target)
                else:
                    i=m+1
            else: #[m, j]为有序数组
                if nums[m]<=target<=nums[j-1]:
                    return binarySearch(nums, m, j, target)
                else:
                    j=m
        return False
```

时间复杂度：$O(n)$，最坏情况下数组全为相等的数字，这时候会遍历整个数组。

空间复杂度：$O(1)$

#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

两种解法：

第一种直接维护一个全局变量，然后二分数组，比较无脑，

第二种通过判断`nums[m]`和`nums[j]`来判断最小值所在的位置。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        i, j = 0, len(nums)
        res = min(nums[i], nums[j-1])
        while i<j:
            m = i + (j-i)//2
            if nums[m]>nums[i]: #[i, m]为有序数组，所以比较res和nums[i]即可
                res = min(res, nums[i])
                i=m+1
            else: #[m, j]为有序数组
                res = min(res, nums[m])
                j=m
        return res
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        i, j = 0, len(nums)-1
        while i<j:
            m = i + (j-i)//2
            if nums[m]>nums[j]: # 必定在右边，如[3,4,1,2]中的4或3
                i=m+1
            else: # 必定在左边，如[3,4,1,2]中的1或2
                j=m             
        return nums[i]
```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

还是两种解法，和上面一样

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        i, j = 0, n
        res = min(nums[0], nums[n-1])
        while i<j:
            m = i + (j-i)//2
            if nums[m]==nums[i] and nums[m]==nums[j-1]:
                res = min(res, nums[m])
                i+=1
                j-=1
            elif nums[m]>=nums[i]: #[i, m]为有序数组
                res = min(res, nums[i])
                i=m+1
            else: #[m, j]为有序数组
                res = min(res, nums[m])
                j=m
        return res
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]: left = mid + 1
            elif nums[mid] < nums[right]: right = mid
            else: right = right - 1 # 后退一步再进行判断
        return nums[left]
```

时间复杂度：$O(n)$，最坏情况下数组全为相等的数字，这时候会遍历整个数组。

空间复杂度：$O(1)$

#### 69. x 的平方根 

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x==1 or x==0:
            return x
        i, j = 1, x
        # 找m*m>x的下界，然后减一就是所求值
        while i<j:
            m = i + (j-i)//2
            if m*m>x:
                j = m             
            else:
                i = m+1
        return i-1

```

时间复杂度：$O(logn)$

空间复杂度：$O(1)$

[74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

先找第一列中小于等于target的上界,即找到大于target的下界再减一，注意，用上面的二分模板都是找下界。

再找这一行中大于等于target的下界。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m,n = len(matrix), len(matrix[0])
        i, j = 0, m
        # 先找小于等于target的上界,即大于target的下界-1
        while i<j:
            mi = i+(j-i)//2
            if matrix[mi][0]>target:
                j = mi
            else:
                i = mi+1
             
        # 这时的i是大于target的下界，减一即为小于等于target的上界
        idx = i-1
        
        l, r = 0, n
        # 再找大于等于target的下界
        while l<r:
            mi = l+(r-l)//2
            if matrix[idx][mi]>=target:
                r = mi
            else:
                l = mi+1
        
        return True if l<n and matrix[idx][l]==target else False

```

时间复杂度：$O(logm + logn)$

空间复杂度：$O(1)$

#### [436. 寻找右区间](https://leetcode.cn/problems/find-right-interval/)

```python
class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        mapping = {interval[0]:idx for idx, interval in enumerate(intervals)}
        starts = sorted(list(mapping.keys()))

        def find(arr, start, end, target):
            i, j = start, end
            while i<j:
                m = i+(j-i)//2
                if arr[m]>=target:
                    j = m
                else:
                    i = m + 1
            return i
        
        n = len(intervals)
        res = [-1 for _ in range(n)]
        for idx, interval in enumerate(intervals):
            right = find(starts, 0, n, interval[1])
            res[idx] = -1 if right>=n else mapping[starts[right]]
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

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

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

求最长子串，参考模板二

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        n = len(s)
        i, j = 0, 0
        mapping = {}
        res = 0
        while j<n:
            while s[j] in mapping:
                mapping.pop(s[i])
                i += 1
            mapping[s[j]] = True
            res = max(res, j-i+1)
            j+=1
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [219. 存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii/)

解法一：哈希表查上一个出现的idx

```
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        mapping = {}
        n = len(nums)
        for idx, num in enumerate(nums):
            if num in mapping:
                lst_idx = mapping[num]
                if idx-lst_idx<=k:
                    return True
            mapping[num] = idx
        return False
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

解法二：滑动窗口+哈希表

```python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        mapping = {}
        n = len(nums)
        i, j = 0, 0
        while j<n:
            if j-i+1>k+1: # 注意这里的滑动窗口大小应该是k+1
                mapping.pop(nums[i])
                i+=1
            if nums[j] in mapping:
                return True
            mapping[nums[j]] = True
            j+=1
        return False
```

时间复杂度：$O(n)$

空间复杂度：$O(k)$

#### [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)

思路：利用sortedList(红黑树实现)和二分查找滑动窗口中最接近nums[j]的值，并判断他们之差是否小于t。

```python
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        from sortedcontainers import SortedList

        n = len(nums)
        i, j= 0, 0
        window = SortedList()
        while j<n:       
            if j-i+1>k+1:
                window.remove(nums[i])
                i+=1
            window.add(nums[j])
            idx = bisect.bisect_left(window, nums[j])
            if idx-1>=0 and window[idx]-window[idx-1]<=t:
                return True
            if idx+1<len(window) and window[idx+1]-window[idx]<=t:
                return True       
            j+=1
        return False
```

时间复杂度：$O(nlogk)$

空间复杂度：$O(k)$

#### [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

```python
class Solution(object):
    def characterReplacement(self, s, k):
        n = len(s)

        i, j = 0, 0
        mapping = {} # 记录窗口中各个字符的出现次数
        max_cnt = 0 # 记录窗口中出现次数最多的字符（的出现次数）
        res = 0
        while j<n:
            mapping[s[j]] = mapping.get(s[j], 0) + 1
            max_cnt = max(max_cnt, mapping[s[j]])
            while max_cnt + k < j-i+1: # 不满足题意
                mapping[s[i]] -= 1
                for key in mapping.keys():
                    max_cnt = max(max_cnt, mapping[key])
                i+=1
            
            res = max(res, j-i+1)
            j+=1
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$，字符只有26个，哈希表可看作常数大小

#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

通过移动左界来寻找最小值

```python
class Solution(object):
    def minSubArrayLen(self, target, nums):
        n = len(nums)
        i, j = 0, 0
        total, res = 0, float('inf')
        while j<n:
            total += nums[j]
            while total>=target:
                res = min(res, j-i+1)
                total -= nums[i]
                i+=1
            j+=1
        return res if res!=float('inf') else 0
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

```python
class Solution(object):
    def findAnagrams(self, s, p):
        m, n = len(s), len(p)
        if m<n:
            return []
        target_map = defaultdict(int)
        window_map = defaultdict(int)

        for char in p:
            target_map[char] += 1
        
        for char in s[:n]:
            window_map[char] += 1

        res = []
        for i in range(m-n+1):
            if target_map == window_map:
                res.append(i)
            if i<m-n:
                window_map[s[i]]-=1
                if window_map[s[i]]==0:
                    window_map.pop(s[i])
                window_map[s[i+n]]+=1 
        return res
```

时间复杂度：$O(n + 26*(m-n))$

空间复杂度：$O(1)$ or $O(26)$

#### [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)

```python
class Solution(object):
    def medianSlidingWindow(self, nums, k):
        from sortedcontainers import SortedList

        window = SortedList()
        n = len(nums)
        m = k//2

        for i in range(k):
            window.add(nums[i])
        
        res = []
        for i in range(n-k+1):
            if k%2==0:     
                res.append((window[m]+window[m-1])/2)
            else:
                res.append(window[m])
            
            if i+k<n:
                window.remove(nums[i])
                window.add(nums[i+k])
        
        return res
```

时间复杂度：$O(nlogk)$，红黑树的删除和插入操作是$O(logk)$

空间复杂度：$O(k)$

#### [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        m, n = len(s1), len(s2)
        if m>n:
            return False 
        target_map = defaultdict(int)
        window_map = defaultdict(int)

        for char in s1:
            target_map[char] += 1
        
        for char in s2[:m]:
            window_map[char] += 1

        res = []
        for i in range(n-m+1):
            if target_map == window_map:
                return True
            
            if i<n-m:
                window_map[s2[i]]-=1
                if window_map[s2[i]]==0:
                    window_map.pop(s2[i])
                window_map[s2[i+m]]+=1 

        return False
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [594. 最长和谐子序列](https://leetcode-cn.com/problems/longest-harmonious-subsequence/)

根据和谐子序列的定义，该序列必定只可能有两个值a和b，且b-a=1(假定b>a)。

所以可以排序之后滑窗。

```python
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        n = len(nums)
        nums = sorted(nums)

        i, j = 0, 0
        res = 0
        while j<n:
            while nums[j]-nums[i]>1:
                i+=1
            
            if nums[j]-nums[i]==1:
                res = max(res, j-i+1)
            
            j+=1
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(1)$

#### [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

直接滑窗即可

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)
        sumn = sum(nums[:k])
        res = float('-inf')
        for i in range(n-k+1):
            res = max(res, sumn)
            
            if i<n-k:
                sumn-=nums[i]
                sumn+=nums[i+k]
        
        return res/k
```

时间复杂度：$O(n)$

空间复杂度：O(1)

#### [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

```python
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        return self.find(nums, k) - self.find(nums, k-1)
    
    def find(self, nums, k):
        # 返回值为最多存在k个不同整数的子区间个数
        # 原问题不能直接滑窗，这个问题可以直接滑窗
        n = len(nums)
        mapping = defaultdict(int)
        res = 0
        i, j = 0, 0
        while j<n:
            mapping[nums[j]] += 1
            while len(mapping)>k:
                mapping[nums[i]] -= 1
                if mapping[nums[i]]==0:
                    mapping.pop(nums[i])
                i+=1
            res += (j-i)
            j+=1
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

将原问题转换成窗口大小为$n-k$的最小和问题

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        window_size = n-k
        total = sum(cardPoints[:n-k])
        res = float('inf')
        for i in range(k+1):
            res = min(res, total)

            if i<k:
                total-=cardPoints[i]
                total+=cardPoints[i+n-k]
        
        return sum(cardPoints) - res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [1610. 可见点的最大数目](https://leetcode-cn.com/problems/maximum-number-of-visible-points/)

将points转换为极角之后滑动窗口

```python
class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        sameCnt = 0
        polarDegrees = [] # 存放points对location的极角
        for p in points:
            if p == location:
                sameCnt += 1
            else:
                polarDegrees.append(atan2(p[1] - location[1], p[0] - location[0]))
        polarDegrees.sort()

        n = len(polarDegrees)
        polarDegrees += [deg + 2 * pi for deg in polarDegrees] # 避免漏掉同时看到第一和第四象限的点
        return sameCnt + self.sw(polarDegrees, angle*pi/180) # 主要要对angle进行转换

    def sw(self, nums, k):
        n = len(nums)
        i, j = 0, 0
        res = 0
        while j<n:
            while nums[j]-nums[i]>k:
                i+=1
            if nums[j]-nums[i]<=k:
                res = max(res, j-i+1)
            j+=1
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

#### [1984. 学生分数的最小差值](https://leetcode-cn.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/)

```python
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        nums = sorted(nums)
        n = len(nums)
        if n==1:
            return 0
        res = float('inf')
        for i in range(n-k+1):
            res = min(res, nums[i+k-1]-nums[i])
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(1)$

#### [2024. 考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

```python
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        n = len(answerKey)

        i, j = 0, 0
        m = defaultdict(int)
        res = 0
        while j<n:
            m[answerKey[j]] += 1
            while max(m['T'], m['F'])+k<j-i+1:
                m[answerKey[i]] -= 1
                i+=1
            res = max(res, j-i+1)
            j+=1
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

解法一：使用sortedList

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from sortedcontainers import SortedList

        queue = SortedList()
        n = len(nums)
        i, j = 0, 0
        res = float('-inf')
        while j<n:
            queue.add(nums[j])
            while queue[-1]-queue[0]>limit:
                queue.remove(nums[i])
                i+=1
            res = max(res, j-i+1)
            j+=1
        return res if res!=float('-inf') else 0
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

解法二：维护两个单调队列

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        n = len(nums)
        maxn, minn = deque(), deque()
        i, j = 0, 0
        res = 0
        while j<n:
            while maxn and nums[j]>maxn[0]: # max为单调增队列
                maxn.popleft()
            maxn.appendleft(nums[j])
            while minn and nums[j]<minn[0]: # min为单调减队列
                minn.popleft()
            minn.appendleft(nums[j])
            while maxn and minn and maxn[-1]-minn[-1]>limit:
                if maxn[-1]==nums[i]:
                    maxn.pop()
                if minn[-1]==nums[i]:
                    minn.pop()
                i+=1
            res = max(res, j-i+1)
            j+=1
        return res 
```

时间复杂度：$O(n)$，单调队列最多需要增删$n$次。

空间复杂度：$O(n)$

#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        m, n = len(s), len(t)
        target = defaultdict(int)
        for char in t:
            target[char] += 1
        
        def valid(cur, target):
            for key in target.keys():
                if cur.get(key, 0)<target[key]:
                    return False
            return True

        i, j = 0, 0
        cur = defaultdict(int)
        min_cnt = float('inf')
        res = ""
        while j<m:
            cur[s[j]] += 1
            while valid(cur, target):
                if j-i+1<min_cnt:
                    res = s[i:j+1]
                    min_cnt = j-i+1
                cur[s[i]] -= 1
                i+=1
            j+=1
        return res
```

时间复杂度：$O(n*k)$，$k$为t中不同字符的个数

空间复杂度：$O(k)$

## dfs

### 博弈论

#### [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/)

状态压缩+博弈论

```python
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        if desiredTotal<=1:
            return True
            
        m = defaultdict(bool)
        def dfs(state, sumn):

            if state in m:
                return m[state]
            
            for i in range(maxChoosableInteger):
                num = i+1               
                cur = 1<<i
                if state & cur:
                    continue
                if sumn+num>=desiredTotal or not dfs(state|cur, sumn+num):
                    m[state] = True
                    return True

            m[state] = False
            return False
        return (1+maxChoosableInteger)*maxChoosableInteger//2>=desiredTotal and dfs(0, 0)
```

时间复杂度：$O(2^{n}*n)$，总共要搜索的状态是子集个数$2^{n}$，每次搜索要遍历n个数字

空间复杂度：$O(2^{n})$，记忆化搜索需要的空间记忆。

### 状态压缩

从数组里不放回的取出元素遍历，如何表示数组的当前状态state？

假设数组长度为n，并且n<32，可以用一个2**(n)大小的整数state来表示表示数组的状态。

state的第i位为0时代表数组的第i位元素未被使用，为1时代表已被使用。

这样的好处是，state是数字，很方便存储，而且可被哈希,可以用哈希表优化dfs速度。



思考：有的dfs可以用参数k来表示当前遍历到了数组的哪个位置，什么时候传参数k，什么时候用状态压缩？

如果是传参数k，是不能“回头”的，所以如果需要“回头”，要用状态压缩。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/) | Medium | https://leetcode-cn.com/problems/beautiful-arrangement/      |
| [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/) | Medium | https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/ |
|                                                              |        |                                                              |

#### [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/)

```python
class Solution:
    def countArrangement(self, n):
        m = {}
        def dfs(state, l):
            """
            # state: 用来记录n个整数的使用情况
            # l: 当前排列的长度
            """
            if l==n:
                return 1 
            if state in m:
                return m[state]
            
            res = 0
            for i in range(1, n+1):
                cur = 1<<(i-1) 
                if cur & state!=0: # 当前数字已被使用
                    continue
                cur_length = l + 1
                if cur_length%i ==0 or i%cur_length==0: # 题目要求                    
                    res+= dfs(state|cur, l+1) # state|cur是将对应位数置为1
            m[state] = res
            return res
        
        res = dfs(0, 0)
        return res 
```

#### [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)

```python
class Solution:
    def canPartitionKSubsets(self, nums, k):
        sum_nums = sum(nums)
        if sum_nums%k!=0 or len(nums)<k:
            return False
        
        sumn = int(sum_nums/k) # 每个子集的和
        if max(nums)>sumn:
            return False
        
        m = {}
        def dfs(state, cur_sum, num_left):
            '''
            # state: 2*n，state的第i位为1代表nums的第i个数被用过
            # cur_sumn: 当前的长度
            # num_left：还剩下多少个子集
            '''
            if cur_sum > sumn: # 超过了指定大小
                return False
            if cur_sum == sumn:
                return dfs(state, 0, num_left-1)
            if num_left==0 and state == (1<<len(nums)) - 1:
                return True
            
            if state in m:
                return m[state]
            for i in range(len(nums)):
                cur = 1<<i
                if cur & state != 0:
                    continue
                if dfs(cur|state, cur_sum+nums[i], num_left):
                    return True
            
            m[state] = False
            return False
        
        return dfs(0, 0, k)
```

## 动态规划

### 路径问题

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/) | Medium | https://leetcode-cn.com/problems/unique-paths/               |
| [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/) | Medium | https://leetcode-cn.com/problems/unique-paths-ii/            |
| [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/) | Medium | https://leetcode-cn.com/problems/minimum-path-sum/           |
| [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/) | Medium | https://leetcode-cn.com/problems/triangle/                   |
| [931. 下降路径最小和](https://leetcode-cn.com/problems/minimum-falling-path-sum/) | Medium | https://leetcode-cn.com/problems/minimum-falling-path-sum/   |
| [1289. 下降路径最小和  II](https://leetcode-cn.com/problems/minimum-falling-path-sum-ii/) | Hard   | https://leetcode-cn.com/problems/minimum-falling-path-sum-ii/ |
| [1575. 统计所有可行路径](https://leetcode-cn.com/problems/count-all-possible-routes/) | Hard   | https://leetcode-cn.com/problems/count-all-possible-routes/  |
| [576. 出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/) | Medium | https://leetcode-cn.com/problems/out-of-boundary-paths/      |

#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

观察到`dp[i][j]=dp[i+1][j]+dp[i][j+1]`，即当前状态依赖下一行和下一列，所以从后往前迭代即可。

```python
class Solution(object):
    def uniquePaths(self, m, n):
        dp = [[0 for i in range(n)] for i in range(m)]
        # dp init
        dp[m-1][n-1] = 1
        for i in range(m-2, -1, -1):
            dp[i][n-1] = dp[i+1][n-1]
        for j in range(n-2, -1, -1):
            dp[m-1][j] = dp[m-1][j+1]
		
        # 从后往前迭代
        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i][j] = dp[i+1][j] + dp[i][j+1]
        return dp[0][0]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

增加一个判断，即当前节点有障碍物时不进行更新（路径总数为0）。

```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m,n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for i in range(n)] for i in range(m)]
        dp[m-1][n-1] = 1 if obstacleGrid[m-1][n-1]==0 else 0
        for i in range(m-2, -1, -1):
            if obstacleGrid[i][n-1]!=1:
                dp[i][n-1] = dp[i+1][n-1]
        for j in range(n-2, -1, -1):
            if obstacleGrid[m-1][j]!=1:
                dp[m-1][j] = dp[m-1][j+1]

        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                if obstacleGrid[i][j]!=1:
                    dp[i][j] = dp[i+1][j] + dp[i][j+1]
        return dp[0][0]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

```python
class Solution(object):
    def minPathSum(self, grid):
        m, n = len(grid), len(grid[0])
        dp = [[0 for i in range(n)] for i in range(m)]
        # dp init
        dp[m-1][n-1] = grid[m-1][n-1]
        for i in range(m-2, -1, -1):
            dp[i][n-1] = dp[i+1][n-1] + grid[i][n-1]
        for j in range(n-2, -1, -1):
            dp[m-1][j] = dp[m-1][j+1] + grid[m-1][j]
		
        # 从后往前迭代
        for i in range(m-2, -1, -1):
            for j in range(n-2, -1, -1):
                dp[i][j] = min(dp[i+1][j], dp[i][j+1]) + grid[i][j]
        return dp[0][0]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

这题可以用滚动数组降低空间复杂度，因为当前行的更新只依赖于下一行。

```python
class Solution(object):
    def minimumTotal(self, triangle):
        m, n = len(triangle), len(triangle[-1])
        dp = [triangle[m-1][i] for i in range(n)]

        for i in range(m-2, -1, -1):
            for j in range(i+1):
                dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
        return dp[0]
```

时间复杂度：$O(n^2)$，其中n是三角形的行数

空间复杂度：$O(n)$

#### [931. 下降路径最小和](https://leetcode-cn.com/problems/minimum-falling-path-sum/)

```python
class Solution(object):
    def minFallingPathSum(self, matrix):
        n = len(matrix)
        dp = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            dp[n-1][i] = matrix[n-1][i]

        for i in range(n-2, -1, -1):
            for j in range(n):
                minn = dp[i+1][j]
                minn = min(minn, dp[i+1][j-1]) if j-1>=0 else minn
                minn = min(minn, dp[i+1][j+1]) if j+1<n else minn
                dp[i][j] = matrix[i][j] + minn 
        return min(dp[0])
            
```

时间复杂度：$O(n^2)$

空间复杂度：$O(n^2)$

#### [1289. 下降路径最小和  II](https://leetcode-cn.com/problems/minimum-falling-path-sum-ii/)

用数组minn 存放 $min(dp[:i] + dp[i+1:])$,这样的好处是每次计算dp时非常方便。

```python
class Solution(object):
    def minFallingPathSum(self, grid):
        n = len(grid)
        if n==1:
            return grid[0][0]
        dp = [grid[-1][i] for i in range(n)]

        def getMin(dp):
            # minn[i]的含义是除dp[i]外的最小值
            minn = [0 for i in range(n)]
            # 遍历一遍dp，得到第一小和第二小
            fir, sec = float('inf'), float('inf')
            for i in range(n):
                if dp[i]<fir:
                    sec = fir
                    fir = dp[i]
                elif dp[i]<sec:
                    sec = dp[i]
            for i in range(n):
                if dp[i]==fir: # 第一小之外的最小值为第二小值
                    minn[i] = sec
                else: # 其它值之外的最小值为第一小值
                    minn[i] = fir
            return minn
        
        minn = getMin(dp)
        for i in range(n-2, -1, -1):
            for j in range(n):
                dp[j] = grid[i][j] + minn[j]
            minn = getMin(dp)
        return min(dp)
        
```

时间复杂度：$O(n^2)$，`getMin`函数复杂度为$O(n)$

空间复杂度：$O(n)$

#### [1575. 统计所有可行路径](https://leetcode-cn.com/problems/count-all-possible-routes/)

解法一：dfs

```python
class Solution(object):
    def countRoutes(self, locations, start, finish, fuel):
        n = len(locations)
        m = {}
        def dfs(cur_city, cur_fuel, path):
            cur_state = str((cur_city, cur_fuel))
            if cur_state in m:
                return m[cur_state]     
            paths = 0
            if cur_city==finish: # 在finish的城市停下不走了
                paths+=1
            
            if cur_fuel == 0:
                return paths
            for i in range(n):
                cost = abs(locations[cur_city]-locations[i])
                if cur_city==i or cost>cur_fuel:
                    continue
                paths += dfs(i, cur_fuel-cost, path+[i])

            m[cur_state] = paths
            return paths
        res = dfs(start, fuel, [start])%((10**9 )+7)
        return res
```

解法二：

```python
class Solution(object):
    def countRoutes(self, locations, start, finish, fuel):
        n = len(locations)
        # dp[i][j]表示在城市j还剩i燃料时的路径数量
        dp = [[0 for i in range(n)] for i in range(fuel+1)]
        # dis[i][j]表示城市i到城市j的距离
        dis = [[0 for i in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                dis[i][j] = abs(locations[i]-locations[j])
        
        # dp init
        for i in range(fuel+1):
            for j in range(n):
                if j==finish:
                    dp[i][j] = 1            
                for k in range(n):
                    if j==k:
                        continue
                    if dis[j][k]<=i: # 燃料足够从j走到k
                        dp[i][j] += dp[i-dis[j][k]][k] # 在消耗了dis[j][k]燃料后走到了城市k
        return dp[fuel][start]%((10**9) + 7)
```

时间复杂度：$O(f*n*n)$,其中`f`为燃料总量，`n`为城市总数

空间复杂度：$O(n*max(n, f))$

#### [576. 出界的路径数](https://leetcode-cn.com/problems/out-of-boundary-paths/)

```python
class Solution(object):
    def findPaths(self, m, n, maxMove, startRow, startColumn):
        if maxMove==0:
            return 0
        # dp[k][i][j]表示在剩下k次移动机会时，球在(i, j)位置能踢出界的数量
        dp = [[[0 for i in range(n)] for i in range(m)] for i in range(maxMove+1)]

        # 初始化边界状态
        for i in range(m):
            dp[1][i][0] += 1
            dp[1][i][n-1] += 1
        for j in range(n):
            dp[1][0][j] += 1
            dp[1][m-1][j] += 1
        
        for k in range(2, maxMove+1):
            for i in range(m):
                for j in range(n):
                    # 对应上下左右
                    dp[k][i][j] = dp[k][i][j] + dp[k-1][i+1][j] if i+1<m else dp[k][i][j]
                    dp[k][i][j] = dp[k][i][j] + dp[k-1][i][j+1] if j+1<n else dp[k][i][j]
                    dp[k][i][j] = dp[k][i][j] + dp[k-1][i-1][j] if i-1>=0 else dp[k][i][j]
                    dp[k][i][j] = dp[k][i][j] + dp[k-1][i][j-1] if j-1>=0 else dp[k][i][j]
                    
                    # 也可以直接踢出界
                    dp[k][i][j] += dp[1][i][j]
        return dp[maxMove][startRow][startColumn]%((10**9)+7)
```

时间复杂度：$O(f*m*n)$,其中`f`为可移动的次数

空间复杂度：$O(f*m*n)$

#### [1301. 最大得分的路径数目](https://leetcode-cn.com/problems/number-of-paths-with-max-score/)

解法一：dfs

```python
class Solution(object):
    def pathsWithMaxScore(self, board):
        m = {}
        def dfs(i, j):
            # 返回(i, j)的最大得分和此得分的方案数
            if i==0 and j==0:
                return 0, 1           
            
            if board[i][j] == 'X':
                return 0, 0

            pos = str((i, j))
            if pos in m:
                return m[pos][0], m[pos][1]
		
            if j-1>=0:
                left_max, left_num = dfs(i, j-1)
            else:
                left_max, left_num = 0, 0
                
            if i-1>=0:
                up_max, up_num = dfs(i-1, j)
            else:
                up_max, up_num = 0, 0
                
            if i-1>=0 and j-1>=0:
                ul_max, ul_num = dfs(i-1, j-1) 
            else:
                ul_max, ul_num = 0, 0
                
            maxn = max(left_max, max(up_max, ul_max))
            cur_num = 0
            if maxn==left_max:
                cur_num+=left_num
            if maxn==up_max:
                cur_num+=up_num
            if maxn==ul_max:
                cur_num+=ul_num

            maxn = maxn + int(board[i][j]) if board[i][j]!='S' else maxn
            m[pos] = (maxn, cur_num)
            return maxn, cur_num

        maxn, num = dfs(len(board)-1, len(board[0])-1)
        mod = (10**9)+7
        maxn = maxn%mod if num!=0 else 0
        num = num%mod
        return maxn, num
```

解法二：dp

```python
class Solution(object):
    def pathsWithMaxScore(self, board):
        m, n = len(board), len(board[0])

        # maxn[i][j]表示得分最大值，num[i][j]表示得到此最大得分的路径数
        num = [[0 for i in range(n)] for i in range(m)]
        maxn = [[0 for i in range(n)] for i in range(m)]
        
        for i in range(m):
            for j in range(n):
                if board[i][j]=='E':
                    maxn[i][j] = 0
                    num[i][j] = 1
                    continue
                if board[i][j] == 'X':
                    continue

                if j-1>=0:
                    left_max, left_num = maxn[i][j-1], num[i][j-1]
                else:
                    left_max, left_num = 0, 0

                if i-1>=0:
                    up_max, up_num = maxn[i-1][j], num[i-1][j]
                else:
                    up_max, up_num = 0, 0
                
                if i-1>=0 and j-1>=0:
                    ul_max, ul_num = maxn[i-1][j-1], num[i-1][j-1]
                else:
                    ul_max, ul_num = 0, 0
                
                cur_max = max(left_max, max(up_max, ul_max))
                cur_num = 0
                if cur_max==left_max:
                    cur_num+=left_num
                if cur_max==up_max:
                    cur_num+=up_num
                if cur_max==ul_max:
                    cur_num+=ul_num

                cur_max = cur_max + int(board[i][j]) if board[i][j]!='S' else cur_max
                maxn[i][j] = cur_max
                num[i][j] = cur_num
        
        if num[m-1][n-1]==0:
            return 0, 0
        mod = (10**9)+7
        return maxn[m-1][n-1]%mod, num[m-1][n-1]%mod
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        memory = [[0 for i in range(n)] for i in range(m)]
        def dfs(i, j):
            if memory[i][j]!=0:
                return memory[i][j]
            
            res = 1
            if i+1<m and matrix[i+1][j]>matrix[i][j]:
                res = max(res, 1 + dfs(i+1, j))
            if j+1<n and matrix[i][j+1]>matrix[i][j]:
                res = max(res, 1 + dfs(i, j+1))
            if i-1>=0 and matrix[i-1][j]>matrix[i][j]:
                res = max(res, 1 + dfs(i-1, j))
            if j-1>=0 and matrix[i][j-1]>matrix[i][j]:
                res = max(res, 1 + dfs(i, j-1))
            
            memory[i][j] = res
            return res
        
        res = 1
        for i in range(m):
            for j in range(n):
               res = max(res, dfs(i, j))
        # print(memory)
        return res 
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])

        v = [[False for i in range(n)] for i in range(m)]
        res = 0

        def dfs(i, j):
            v[i][j] = True
            if i+1<m and grid[i+1][j]!='0' and not v[i+1][j]:
                dfs(i+1, j)
            if j+1<n and grid[i][j+1]!='0' and not v[i][j+1]:
                dfs(i, j+1)
            if i-1>=0 and grid[i-1][j]!='0' and not v[i-1][j]:
                dfs(i-1, j)
            if j-1>=0 and grid[i][j-1]!='0' and not v[i][j-1]:
                dfs(i, j-1)

        for i in range(m):
            for j in range(n):
                if v[i][j] or grid[i][j]=='0':
                    continue     
                res+=1         
                dfs(i, j)
        return res
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

### 0-1背包问题模板

0-1背包问题的基本模板，根据问题的不同要修改递推公式

问题定义：有`N`件物品和一个容量为`V`的背包，第`i`件物品的体积为`v[i]`，价值为`w[i]`，求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。

#### dp[N][V+1\]模板

```python
def solution():
    # dp[i][j]的含义是前i件物品，背包容量为j的最大价值
    dp = [[0 for i in range(V+1)] for i in range(N)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = w[0] if v[0]<=i else 0
    
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0
            if j>=v[i]: # 第i件物品可以被装进背包
                cur = max(cur, dp[i-1][j-v[i]] + w[i]) # 装进背包（可能会扔掉背包的一些东西来腾出空间）
            cur = max(cur, dp[i-1][j]) # 不装进背包
            dp[i][j] = cur
    return dp[N-1][V]
```

时间复杂度：$O(N*V)$

空间复杂度：$O(N*V)$

#### dp[2][V+1\]模板

```python
def solution():
    # 观察dp表达式，只需要一个2*(V+1)的数组即可
    # dp[i][j]的含义是前i&1件物品，背包容量为j的最大价值
    dp = [[0 for i in range(V+1)] for i in range(2)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = w[0] if v[0]<=i else 0
    
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0
            if j>=v[i]: # 第i件物品可以被装进背包
                cur = max(cur, dp[(i-1)&1][j-v[i]]) + w[i]
            cur = max(cur, dp[(i-1)&1][j]) # 不装进背包
            dp[i&1][j] = cur
    return dp[(N-1)&1][V]
```

时间复杂度：$O(N*V)$

空间复杂度：$O(V)$

#### ⭐dp[V+1\]模板

```python
def solution():
    # 观察dp表达式，dp[i][j]依赖的是dp[i-1][j]和dp[i-1][j-v[i]]，如果从后往前遍历，维护一个一维数组即可
    # dp[i][j]的含义是前i件物品，背包容量为j的最大价值
    dp = [0 for i in range(V+1)]
   	# dp init
    for i in range(V+1):
        dp[i] = w[0] if v[0]<=i else 0

    for i in range(0, N):
        for j in range(V, v[i]-1, -1): # 当j<v[i]时，第i件物品闭不可能被装进背包，直接剪枝
            cur = 0
            if j>=v[i]: # 第i件物品可以被装进背包
                cur = max(cur, dp[j-v[i]]) + w[i] # 装进背包（可能会扔掉背包的一些东西来腾出空间）
            cur = max(cur, dp[j]) # 不装进背包
            dp[j] = cur
    return dp[V]
```

时间复杂度：$O(N*V)$

空间复杂度：$O(V)$

### 0-1背包问题

在熟悉了背包问题的模板后，我们可以将一些其他问题转化为背包问题求解。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/) | medium | https://leetcode-cn.com/problems/partition-equal-subset-sum/ |
|                                                              |        |                                                              |
|                                                              |        |                                                              |

#### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

假设数组总和为`sum`, `target=sum/2`，问题等价为`V=target`时，最大价值能否等于`target`的0-1背包问题。

也就是说在某个数组中选若干个数，使得其总和为某个特定值，都可以转换为0-1背包问题。

解法一：

不修改原背包dp的含义

```python
class Solution(object):
    def canPartition(self, nums):
        
        sumn = sum(nums)
        if sumn%2:
            return False
        target = sumn//2
        n = len(nums)
		
        ## dp init
        dp = [0 for i in range(target+1)]
        for i in range(target+1):
            dp[i] = nums[0] if nums[0]<=i else 0

        for i in range(1, n):
            for j in range(target, nums[i]-1, -1):
                dp[j] = max(dp[j-nums[i]]+nums[i], dp[j]) 

        return dp[target]==target
```

时间复杂度：$O(n*target)$,其中`n`为数组长度，`target`为数组之和的一半。

空间复杂度：$O(target+1)$

解法二：

修改原背包dp的含义

`dp[i][j]`定义为从前`i`个数能否选出恰好为和为`j`的子数组

```python
class Solution(object):
    def canPartition(self, nums):
        sumn = sum(nums)
        if sumn%2:
            return False
        target = sumn//2
        n = len(nums)
        
        dp = [False for i in range(target+1)]
        for i in range(target+1):
            dp[i] = True if nums[0]==i else False

        for i in range(1, n):
            for j in range(target, nums[i]-1, -1):
                dp[j] = dp[j] or dp[j-nums[i]] # 选择nums[i]或不选
        return dp[target]
```

时间复杂度：$O(n*target)$,其中`n`为数组长度，`target`为数组之和的一半。

空间复杂度：$O(target+1)$

### 完全背包问题模板

问题定义：有`N`件物品和一个容量为`V`的背包，第`i`件物品的体积为`v[i]`，价值为`w[i]`，求解将哪些物品装入背包(每件物品的个数是无限的），可使这些物品的总体积不超过背包容量，且总价值最大。

#### dp[N][V+1\]模板

```python
def solution():
    # dp[i][j]的含义是前i件物品，背包容量为j的最大价值
    dp = [[0 for i in range(V+1)] for i in range(N)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = (i//v[0])*w[0]
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0         
            # 可以放k件i物品
            num = j//v[i]
            for k in range(1, num+1):
                cur = max(cur, dp[i-1][j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[i-1][j]) # 不装进背包
            dp[i][j] = cur
    return dp[N-1][V]
```

时间复杂度：$O(N*V*V)$

空间复杂度：$O(N*V)$

#### dp[2][V+1\]模板

```python
def solution():
    dp = [[0 for i in range(V+1)] for i in range(2)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = (i//v[0])*w[0]
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0         
            # 可以放k件i物品
            num = j//v[i]
            for k in range(1, num+1):
                cur = max(cur, dp[(i-1)&1][j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[(i-1)&1][j]) # 不装进背包
            dp[i&1][j] = cur
    return dp[(N-1)&1][V]
```

时间复杂度：$O(N*V*V)$

空间复杂度：$O(V)$

#### dp[V+1\]模板

```python
def solution():
    dp = [0 for i in range(V+1)]
   	# dp init
    for i in range(V+1):
        dp[i] = (i//v[0])*w[0]
    for i in range(1, N):
        for j in range(V+1):
            cur = 0         
            # 可以放k件i物品
            num = j//v[i]
            for k in range(1, num+1):
                cur = max(cur, dp[j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[j]) # 不装进背包
            dp[j] = cur
    return dp[V]
```

时间复杂度：$O(N*V*V)$

空间复杂度：$O(V)$

#### ⭐dp[V+1\]模板(优化)

完全背包问题的递推公式为：`dp[i][j] = max(dp[i-1][j], dp[i][j-v[i]]+w[i])$

```python
def solution():
    dp = [0 for i in range(V+1)]
   	# dp init
    for i in range(N):
        for j in range(V+1):
            cur = 0
            if j>=v[i]:
                cur = max(cur, dp[j-v[i]] + w[i]) 
            cur = max(cur, dp[j]) # 不装进背包
            dp[j] = cur
    return dp[V]
```

时间复杂度：$O(N*V)$

空间复杂度：$O(V)$

### 完全背包问题

| 题目                                                         | 难度   | 链接                                              |
| ------------------------------------------------------------ | ------ | ------------------------------------------------- |
| [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/) | Medium | https://leetcode-cn.com/problems/perfect-squares/ |
| [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/) | Medium | https://leetcode-cn.com/problems/coin-change/     |
| [518. 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/) | Medium | https://leetcode-cn.com/problems/coin-change-2/   |
|                                                              |        |                                                   |

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

解法一：（n=6665时超时）

$dp[i][j] = min(dp[i-1][j-1], dp[i-1][j-nums[i]]+1, dp[i-1][j-2*nums[i]]+2, ..., dp[i-1][j-k*nums[i]]+k)$

```python
class Solution:
    def numSquares(self, n: int) -> int:
        num = int(n**(1/2))
        nums = [i*i for i in range(1, num+1)]

        # dp[i][j]表示前i个数和为j的最少数量
        dp = [[0 for i in range(n+1)] for i in range(num)]
        for i in range(n+1):
            dp[0][i] = i
        for i in range(1, num):
            for j in range(1, n+1):
                cur = dp[i-1][j]
                x = j//nums[i]

                for k in range(x+1):
                    cur = min(cur, dp[i-1][j-k*nums[i]]+k)
                
                dp[i][j] = cur
        # print(dp)
        return dp[num-1][n]

```

时间复杂度：$O(n^2*\sqrt n)$

空间复杂度:$O(\sqrt n *n)$

解法二：

观察到$dp[i][j-nums[i]] = min(dp[i-1][j-nums[i]], dp[i-1][j-2*nums[i]]+1, ..., dp[i-1][j-k*nums[i]]+k-1)$

原本的dp公式可化简为$dp[i][j] = min(dp[i-1][j], dp[i][j-nums[i]]+1)$

即$dp[j] = min(dp[j], dp[j-nums[i]]+1)$

```python
class Solution:
    def numSquares(self, n: int) -> int:
        num = int(n**(1/2))
        nums = [i*i for i in range(1, num+1)]

        # dp[j]表示和为j的最少数量
        dp = [0 for i in range(n+1)]
        # dp init
        for i in range(n+1):
            dp[i] = i # 只有1能用的时候

        for i in range(1, num):
            for j in range(1, n+1):
                if j>=nums[i]:
                    dp[j] = min(dp[j], dp[j-nums[i]]+1)
        return dp[n]
```

时间复杂度：$O(\sqrt n*n)$

空间复杂度：$O(n)$

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

这题的动态转移方程和上一题是一模一样的，直接算即可，注意初始状态的初始化。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        n = len(coins)
        # dp[j]表示和为j的最少数量
        dp = [-1 for i in range(amount+1)]

        # dp init
        for i in range(amount+1):
            dp[i] = i//coins[0] if i%coins[0]==0 else float('inf') # 只有coins[0]能用的时候

        for i in range(1, n):
            for j in range(1, amount+1):
                if j>=coins[i]:                   
                    dp[j] = min(dp[j-coins[i]]+1, dp[j])
        return dp[amount] if dp[amount]!=float('inf') else -1
```

时间复杂度：$O(n*amount)$

空间复杂度：$O(amount)$

#### 518. 零钱兑换 II

先得到一般的状态转移方程$dp[i][j] = dp[i-1][j] + dp[i][j-coins[i]] + ... + dp[i][j-k*coins[i]]$

再化简得到$dp[i][j] = dp[i-1][j] + dp[i][j-coins[i]]$

最后化为一维的，得到$dp[j] = dp[j] + dp[j-coins[i]]$

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [0 for i in range(amount+1)]

        for i in range(amount+1):
            dp[i] = 1 if i%coins[0]==0 else 0
        
        for i in range(1, n):
            for j in range(1, amount+1):
                if j>=coins[i]:
                    dp[j]+=dp[j-coins[i]]
        return dp[amount]
```

时间复杂度：$O(n*amount)$

空间复杂度：$O(amount)$

### 多重背包模板

问题定义：有`N`件物品和一个容量为`V`的背包，第`i`件物品的体积为`v[i]`，价值为`w[i]`，数量为`s[i]`，求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。

#### dp[N][V+1\]模板

```python
def solution():
    # dp[i][j]的含义是前i件物品，背包容量为j的最大价值
    dp = [[0 for i in range(V+1)] for i in range(N)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = min(i//v[0], s[0])*w[0]
    print(dp)
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0         
            # 可以放k件i物品
            num = min(j//v[i], s[i]) # 第i件物品的数量是s[i]
            for k in range(1, num+1):
                cur = max(cur, dp[i-1][j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[i-1][j]) # 不装进背包
            dp[i][j] = cur
    return dp[N-1][V]
```

#### dp[2][V+1\]模板

```python
def solution():
    dp = [[0 for i in range(V+1)] for i in range(2)]
   	# dp init
    for i in range(V+1):
        dp[0][i] = min(i//v[0], s[0])*w[0]
    print(dp)
    for i in range(1, N):
        for j in range(1, V+1):
            cur = 0                   
            num = min(j//v[i], s[i]) # 第i件物品的数量是s[i]
            for k in range(1, num+1): # 可以放k件i物品
                cur = max(cur, dp[(i-1)&1][j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[(i-1)&1][j]) # 不装进背包
            dp[i&1][j] = cur
    return dp[(N-1)&1][V]
```

#### dp[V+1\]模板

```python
def solution():
    # dp[j]的含义是背包容量为j的最大价值
    dp = [0 for i in range(V+1)]
   	# dp init
    for i in range(V+1):
        dp[i] = min(i//v[0], s[0])*w[0]
    for i in range(1, N):
        for j in range(V, v[i]-1, -1):
            cur = 0         
            # 可以放k件i物品
            num = min(j//v[i], s[i]) # 第i件物品的数量是s[i]
            for k in range(1, num+1):
                cur = max(cur, dp[j-k*v[i]] + k*w[i]) 
            cur = max(cur, dp[j]) # 不装进背包
            dp[j] = cur
    return dp[V]
```

### 背包问题

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [1155. 掷骰子的N种方法](https://leetcode-cn.com/problems/number-of-dice-rolls-with-target-sum/) | Medium | https://leetcode-cn.com/problems/number-of-dice-rolls-with-target-sum/ |
| [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/) | Medium | https://leetcode-cn.com/problems/ones-and-zeroes/            |
| [879. 盈利计划](https://leetcode-cn.com/problems/profitable-schemes/) | Hard   | https://leetcode-cn.com/problems/profitable-schemes/         |
| [494. 目标和](https://leetcode-cn.com/problems/target-sum/)  | Medium | https://leetcode-cn.com/problems/target-sum/                 |
| [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/) | Medium | https://leetcode-cn.com/problems/last-stone-weight-ii/       |

#### [1155. 掷骰子的N种方法](https://leetcode-cn.com/problems/number-of-dice-rolls-with-target-sum/)

朴素转移方程：$dp[i][j] = dp[i-1][j-1] + ...+ dp[i-1][j-k]$

```python
class Solution(object):
    def numRollsToTarget(self, n, k, target):
        # dp[i][j]表示前i个骰子，和为target的方式
        dp = [[0 for i in range(target+1)] for i in range(n)]

        for i in range(1, target+1):
            # 只能用1个骰子
            dp[0][i] = 1 if k>=i else 0 
        for i in range(1, n):
            for j in range(1, target+1):
                cur = 0
                # cur = dp[i-1][j] # 必须要用第i个骰子
                for x in range(1, k+1):
                    if j-x<0:
                        break
                    cur += dp[i-1][j-x]
                dp[i][j] = cur
        mod = (10**9)+7
        return dp[n-1][target]%mod

```

时间复杂度：$O(n*target*k)$

空间复杂度：$O(n*target)$

化简转移方程：$dp[i][j] = dp[i-1][j-1] + dp[i][j-1] - dp[i-1][j-k-1]$

```python
class Solution(object):
    def numRollsToTarget(self, n, k, target):
        # dp[i][j]表示前i个骰子，和为target的方式
        dp = [[0 for i in range(target+1)] for i in range(n)]

        for i in range(1, target+1):
            # 只能用1个骰子
            dp[0][i] = 1 if k>=i else 0 
        for i in range(1, n):
            for j in range(1, target+1):             
                dp[i][j] += dp[i][j-1]+dp[i-1][j-1]
                if j-k-1>=0:
                    dp[i][j]-=dp[i-1][j-k-1]
        mod = (10**9)+7
        return dp[n-1][target]%mod
```

时间复杂度：$O(n*target)$

空间复杂度：$O(n*target)$

#### [474. 一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

```python
class Solution(object):
    def findMaxForm(self, strs, m, n):
        num = len(strs)

        # dp[i][j][k]的意思是前i个字符串，最多有i个0j个1的最大子集长度
        dp = [[[0 for i in range(n+1)] for i in range(m+1)] for i in range(num)]
        
        # 提前计算得到各个字符串的01个数
        count_z = [string.count('0') for string in strs]
        count_o = [string.count('1') for string in strs]
        for i in range(m+1):
            for j in range(n+1):
                dp[0][i][j] = 1 if count_z[0]<=i and count_o[0]<=j else 0
        
        for i in range(1, num):
            for j in range(m+1):
                for k in range(n+1):                    
                    dp[i][j][k] =  dp[i-1][j][k] # 不选                
                    if j>=count_z[i] and k>=count_o[i]: # 选
                        dp[i][j][k] = max(dp[i-1][j-count_z[i]][k-count_o[i]] + 1, dp[i][j][k]) 
        return dp[num-1][m][n]

```

时间复杂度：$O(num*m*n)$，num为strs的长度

空间复杂度：$O(num*m*n)$

滚动数组优化：

```python
class Solution(object):
    def findMaxForm(self, strs, m, n):
        num = len(strs)

        # dp[i][j][k]的意思是前i个字符串，最多有i个0j个1的最大子集长度
        dp = [[0 for i in range(n+1)] for i in range(m+1)]
        
        count_z = [string.count('0') for string in strs]
        count_o = [string.count('1') for string in strs]
        for i in range(m+1):
            for j in range(n+1):
                dp[i][j] = 1 if count_z[0]<=i and count_o[0]<=j else 0
        
        for i in range(1, num):
            zero = count_z[i]
            one = count_o[i]
            for j in range(m, -1, -1):
                for k in range(n, -1, -1):                                   
                    if j>=count_z[i] and k>=count_o[i]: # 选
                        dp[j][k] = max(dp[j-zero][k-one] + 1, dp[j][k]) 
        return dp[m][n]

```

时间复杂度：$O(num*m*n)$，num为strs的长度

空间复杂度：$O(m*n)$

#### [879. 盈利计划](https://leetcode-cn.com/problems/profitable-schemes/)

```python
class Solution(object):
    def profitableSchemes(self, n, minProfit, group, profit):
        max_profit = self.cal(group, profit, n) # 先求最大利润
        
        m = len(group) # 总共的工作数
        # dp[i][j][k]表示有前i个工作可以选，有j名员工,产生利润k的计划数
        dp = [[[0 for i in range(max_profit+1)] for i in range(n+1)] for i in range(m)]

        for i in range(n+1):
            for j in range(max_profit+1):
                dp[0][i][j] = 1 if (group[0]<=i and profit[0]==j) or j==0 else 0
        for i in range(1, m):
            p = profit[i]
            g = group[i]
            for j in range(n+1):
                for k in range(max_profit+1):                   
                    dp[i][j][k] += dp[i-1][j][k]
                    if j>=g and k>=p:
                        dp[i][j][k] += dp[i-1][j-g][k-p]
        res = 0
        for i in range(minProfit, max_profit+1): # 求满足题意利润的所有方案数
            res += dp[m-1][n][i]
        return res%((10**9)+7)
    
    def cal(self, group, profit, n):
        # 求n名员工，m个工作，每个工作i需要的员工是group[i]，利润profit[i]的最大利润
        # 很显然，这是个0-1背包问题，时间复杂度：O(m*n)
        m = len(group)
        # dp[i][j]的含义是前i件物品，背包容量为j的最大价值
        dp = [0 for i in range(n+1)]
        # dp init
        for i in range(n+1):
            dp[i] = profit[0] if group[0]<=i else 0

        for i in range(0, m):
            for j in range(n, group[i]-1, -1): # 当j<v[i]时，第i件物品闭不可能被装进背包，直接剪枝
                cur = 0
                if j>=group[i]: # 第i件物品可以被装进背包
                    cur = max(cur, dp[j-group[i]]) + profit[i] # 装进背包（可能会扔掉背包的一些东西来腾出空间）
                cur = max(cur, dp[j]) # 不装进背包
                dp[j] = cur
        return dp[n]
        
```

时间复杂度：$O(maxProfit*m*n)$，$maxProfit$为最大利润数

空间复杂度：$O(maxProfit*m*n)$

#### [494. 目标和](https://leetcode-cn.com/problems/target-sum/)

dfs（使用全局变量）：超时

```python
class Solution(object):
    def findTargetSumWays(self, nums, target):
        n = len(nums)
        self.res = 0
        def dfs(cur, k):
            if cur==target and k==n:
                self.res += 1
                return
            if k>=n:
                return
            dfs(cur+nums[k], k+1)
            dfs(cur-nums[k], k+1)
        
        dfs(0, 0)
        return self.res
```

dfs（使用返回值）：超时

```python
class Solution(object):
    def findTargetSumWays(self, nums, target):
        n = len(nums)
        def dfs(cur, k):
            if cur==target and k==n:
                return 1
            if k>=n:
                return 0
            left = dfs(cur+nums[k], k+1)
            right = dfs(cur-nums[k], k+1)
            res = left+right
            return res
        
        return dfs(0, 0)
```

时间复杂度：$O(2^n)$

空间复杂度：$O(1)$，忽略递归消耗

dfs+备忘录：

```python
class Solution(object):
    def findTargetSumWays(self, nums, target):
        n = len(nums)
        m = {}
        def dfs(cur, k):
            state = str((cur, k))
            if state in m:
                return m[state]
            if cur==target and k==n:
                return 1
            if k>=n:
                return 0
            left = dfs(cur+nums[k], k+1)
            right = dfs(cur-nums[k], k+1)
            res = left+right
            m[state] = res
            return res
        
        return dfs(0, 0)
```

时间复杂度：$O(n*\sum nums[i])$

空间复杂度：$O(n*\sum nums[i])$

动态规划

```python
class Solution(object):
    def findTargetSumWays(self, nums, target):
        n = len(nums)
        sumn = sum(nums)
        # dp[i][j]的含义是前i个数，构成j的方案数
        dp = [[0 for i in range(2*sumn+1)] for j in range(n)]

        for i in range(2*sumn+1):
            if abs(i-sumn)==nums[0]:
                if nums[0]!=0:
                    dp[0][i] = 1
                else:
                    dp[0][i] = 2
        for i in range(1, n):
            for j in range(-1*sumn, sumn+1):
                # dp[i][j+sumn] += dp[i-1][j+sumn] # 第i个数必须要选
                if j-nums[i]>=-1*sumn:
                    dp[i][j+sumn] += dp[i-1][j-nums[i]+sumn]
                if j+nums[i]<sumn+1:
                    dp[i][j+sumn] += dp[i-1][j+nums[i]+sumn]
        return dp[n-1][target+sumn] if target+sumn<2*sumn+1 else 0
```

时间复杂度：$O(n*\sum nums[i])$

空间复杂度：$O(n*\sum nums[i])$

#### [1049. 最后一块石头的重量 II](https://leetcode-cn.com/problems/last-stone-weight-ii/)

将问题转换为，把石头分成两堆，使其重量尽可能相等，所求为这两堆石头的差，可以参考[416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```python
class Solution(object):
    def lastStoneWeightII(self, stones):
        n = len(stones)
        sumn = sum(stones)
        target = (sumn+1)//2

        # dp[i][j]代表前i块石头小于等于j重量的最大重量
        dp = [0 for i in range(target+1)]

        for i in range(target+1):
            dp[i] = stones[0] if i>=stones[0] else 0
        
        for i in range(1, n):
            s = stones[i]
            for j in range(target, s, -1):
                dp[j] = max(dp[j], dp[j-s]+s)

        return abs(sumn-dp[target]-dp[target])
```

时间复杂度：$O(n*\sum stones[i]/2)$

空间复杂度: $O(\sum stones[i]/2)$

### LCS和LIS问题

[宫水三叶:LCS 问题与 LIS 问题的相互关系，以及 LIS 问题的最优解证明](https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247487814&idx=1&sn=e33023c2d474ff75af83eda1c4d01892&chksm=fd9cba59caeb334f1fbfa1aefd3d9b2ab6abfccfcab8cb1dbff93191ae9b787e1b4681bbbde3&token=252055586&lang=zh_CN#rd)

#### LIS问题

问题定义：求给定数组的最长上升子序列长度

解法一：朴素dp

```python
def solution(nums):
    n = len(nums)
    # dp[i]表示以nums[i]为底的最长递增子序列长度
    # dp[i] = max(dp[j])+1 if dp[j]<dp[i] and j<i
    dp = [1 for i in range(n)]
    res = 0
    for i in range(n):
        maxn = 0
        for j in range(i):
            if nums[i]>nums[j]:
            	maxn = max(maxn, dp[j])
        dp[i] = maxn + 1
        res = max(res, dp[i])
    return res
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

解法二：贪心+二分+dp

思路：通过维护一个贪心数组$g$，$g[len]$表示上升子序列长度为$len$的最小结尾元素。更新$dp[i]$的时候去找小于$nums[i]$的上界，可以通过二分解决，整体复杂度是$O(nlogn)$

```python
def solution(nums):
    n = len(nums)
    # dp[i]表示以nums[i]结尾的最长递增子序列长度
    dp = [0 for i in range(n)]
    dp[1] = 1

    # g[i]表示上升子序列长度为i的最小结尾元素
    g = [float('inf') for i in range(n+1)]
    g[0], g[1] = float('-inf'), nums[0]

    def find(arr, target, end):
        # 寻找大于等于target的下界
        i, j = 0, end
        while i<j:
            m = i + (j-i)//2
            if target<=arr[m]:
                j = m
            else:
                i = m+1
        return i

        res = 0
        for i in range(1, n):
            lst_len = find(g, nums[i], i+1) # O(logn)
            dp[i] = lst_len
            g[lst_len] = min(nums[i], g[lst_len]) # 更新g数组
            res = max(res, dp[i])
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

#### LCS问题

问题定义：最长公共子序列问题，求两个数组的最长公共子序列

朴素dp:

```python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        m = len(text1)
        n = len(text2)

        # dp[i][j]表示text1[:i]和text2[:j]的最长公共子序列长度
        dp = [[0 for i in range(n+1)] for i in range(m+1)]
        dp[0][0] = 0

        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1]==text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j])
        return dp[m][n]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

### 序列dp

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/) | Medium | https://leetcode-cn.com/problems/longest-increasing-subsequence/ |
| [334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/) | Medium | https://leetcode-cn.com/problems/increasing-triplet-subsequence/ |
| [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/) | Hard   | https://leetcode-cn.com/problems/russian-doll-envelopes/     |
| [368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/) | Medium | https://leetcode-cn.com/problems/largest-divisible-subset/   |
| [413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/) | Medium | https://leetcode-cn.com/problems/arithmetic-slices/          |
| [446. 等差数列划分 II - 子序列](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/) | Hard   | https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/ |
| [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/) | Medium | https://leetcode-cn.com/problems/house-robber/               |
| [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/) | Medium | https://leetcode-cn.com/problems/delete-and-earn/            |
| [978. 最长湍流子数组](https://leetcode-cn.com/problems/longest-turbulent-subarray/) | Medium | https://leetcode-cn.com/problems/longest-turbulent-subarray/ |
| [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/) | Medium | https://leetcode-cn.com/problems/uncrossed-lines/            |
| [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/) | Medium | https://leetcode-cn.com/problems/delete-operation-for-two-strings/ |
| [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/) | Medium | https://leetcode-cn.com/problems/edit-distance/              |
| [629. K个逆序对数组](https://leetcode-cn.com/problems/k-inverse-pairs-array/) | Hard   | https://leetcode-cn.com/problems/k-inverse-pairs-array/      |
| [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/) | Medium | https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/ |
| [1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/) | Medium | https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/ |
| [1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/) | Hard   | https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/ |

#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

经典LIS问题，直接秒

```python
class Solution(object):
    def lengthOfLIS(self, nums):
        n = len(nums)
        if n==1:
            return 1
        dp = [0 for i in range(n)]
        g = [float('inf') for i in range(n+1)]

        dp[0] = 1
        g[0], g[1] = float('-inf'), nums[0]

        res = 1
        for i in range(1, n):
            target = nums[i]
            l, r = 0, i+1
            while l<r:
                m = l + (r-l)//2
                if g[m]>=target:
                    r = m
                else:
                    l = m + 1
            g[l] = min(g[l], nums[i])
            dp[i] = l
            res = max(res, dp[i])
        return res
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

#### [334. 递增的三元子序列](https://leetcode-cn.com/problems/increasing-triplet-subsequence/)

解法一：暴力dp(超时,75/76)

```python
class Solution(object):
    def increasingTriplet(self, nums):
        # dp[i]表示以nums[i]为底的最长递增子序列长度
        # dp[i] = max(dp[j])+1 if dp[j]<dp[i] and j<i
        n = len(nums)
        dp = [1 for i in range(n)]
        for i in range(n):
            maxn = 0
            for j in range(i):
                if nums[i]>nums[j]:
                    maxn = max(maxn, dp[j])
            dp[i] = maxn + 1
            if dp[i]>=3:
                return True
        return False
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

解法二：LIS式dp

```python
class Solution(object):
    def increasingTriplet(self, nums):
        n = len(nums)
        if n<3:
            return False
        # dp[i]表示以nums[i]结尾的最长递增子序列长度
        dp = [0 for i in range(n)]
        dp[1] = 1

        # g[i]表示上升子序列长度为i的最小结尾元素
        g = [float('inf') for i in range(n+1)]
        g[0], g[1] = float('-inf'), nums[0]
        
        def find(arr, target, end):
            # 寻找大于等于target的下界
            i, j = 0, end
            while i<j:
                m = i + (j-i)//2
                if target<=arr[m]:
                    j = m
                else:
                    i = m+1
            return i
        
        for i in range(1, n):
            lst_len = find(g, nums[i], i+1) # O(logn)
            dp[i] = lst_len
            g[lst_len] = min(nums[i], g[lst_len])
            if dp[i]>=3:
                return True

        return False
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

解法三：

维护两个变量first和second，使first和second尽可能小

```python
class Solution(object):
    def increasingTriplet(self, nums):
        n = len(nums)
        if n<3:
            return False
        first, second = nums[0], float('inf')
        for i in range(1, n):
            num = nums[i]
            if num>second:
                return True
            elif num>first:
                second = num
            elif num<first:
                first = num

        return False
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

思路：按照第一维升序，第二维降序排序，直接转换成LIS问题

这里有一个坑点，就是第二维需要降序，因为如果升序的话，会出现`[[1, 2], [1, 3]]`，`[1, 3]`套`[1, 2]`的情况，而降序的话完美的解决了这一问题。

```python
class Solution(object):
    def maxEnvelopes(self, envelopes):
        n = len(envelopes)
        if n==1:
            return 1
        envelopes = sorted(envelopes, key=lambda x:(x[0], -x[1]))
        dp = [0 for i in range(n)]
        g = [float('inf') for i in range(n+1)]

        dp[0] = 1
        g[0] = float('-inf')

        res = 0
        for i in range(n):
            clen = self.find(g, envelopes[i][1], i+1)
            dp[i] = clen
            g[clen] = min(envelopes[i][1], g[clen])
            res = max(res, clen)
        return res
    
    def find(self, arr, target, end):
        i, j = 0, end
        while i<j:
            m = i + (j-i)//2
            if arr[m]>=target:
                j = m
            else:
                i = m + 1
        return i
```

时间复杂度：$O(nlogn)$

空间复杂度：$O(n)$

#### [368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/)

排序后转换成最长整除序列，其本质是因为整除和递增一样具有传递性。

```python
class Solution(object):
    def largestDivisibleSubset(self, nums):
        n = len(nums)
        nums = sorted(nums)

        # dp[i]为以nums[i]为最大整数的子集大小
        dp = [1 for i in range(n)]
        # g[i]记录dp的转移情况
        g = [-1 for i in range(n)]

        for i in range(1, n):
            maxn = 0
            max_idx = -1
            for j in range(i-1, -1, -1):              
                if nums[i]%nums[j]==0:
                    if dp[j]>maxn:
                        maxn = dp[j]
                        max_idx = j                  
            dp[i] = maxn + 1
            g[i] = max_idx

        res = []
        max_len = max(dp)
        cur = dp.index(max_len)
        for i in range(max_len):
            res.append(nums[cur])
            cur = g[cur]
        return res
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

#### [413. 等差数列划分](https://leetcode-cn.com/problems/arithmetic-slices/)

解法一：dp

```python
class Solution(object):
    def numberOfArithmeticSlices(self, nums):
        n = len(nums)
        if n<3:
            return 0
        # dp[i]表示以nums[i]结尾的等差数列个数
        dp = [0 for i in range(n)]

        res = 0
        for i in range(2, n):
            if nums[i-1]*2 == nums[i-2]+nums[i]:
                dp[i] = dp[i-1] + 1
            res += dp[i]

        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

解法二：

长为3的等差数列贡献为1，长为4的贡献为1+2，长为5的贡献为1+2+3，以此类推

```python
class Solution(object):
    def numberOfArithmeticSlices(self, nums):
        n = len(nums)
        if n<3:
            return 0
        
        res, tmp = 0, 1
        i = 2
        while i<n:
            if nums[i]+nums[i-2]==2*nums[i-1]:
                res += tmp
                tmp += 1
            else:
                tmp = 1
            i+=1

        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [446. 等差数列划分 II - 子序列](https://leetcode-cn.com/problems/arithmetic-slices-ii-subsequence/)

思路：动态规划，$dp[i][j]$表示以$nums[i]$结尾的，公差为$d$的数列个数，分析可得到递推方程 $dp[i] = \sum dp[j][diff]$，

同时，我们需要更新$dp[i][diff] += dp[j][diff] + 1$(算上nums[j], nums[i]这个等差数列)，

这里有个巧妙的地方，当$dp[j][diff]]$不为0时，表示其等差数列长度已经大于等于2，这时再加上$nums[i]$，长度必定大于等于3。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [defaultdict(int) for i in range(n)]

        res = 0
        for i in range(n):
            for j in range(i):
                diff = nums[i]-nums[j]
                res += dp[j][diff]
                dp[i][diff] += dp[j][diff]+1
        return res
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n^{2})$

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][0]表示不偷i号的最大金额，dp[i][1]表示偷i号的最大金额（只考虑nums[:i+1]）
        dp =[[0 for i in range(2)] for i in range(n)]

        dp[0][1] = nums[0]
        for i in range(1, n):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1])
            dp[i][1] = dp[i-1][0] + nums[i]
        return max(dp[n-1][0], dp[n-1][1]) 
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        x, y = 0, nums[0]
        for i in range(1, n):
            x, y = max(x, y), x+nums[i] 
        return max(x, y) 
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [740. 删除并获得点数](https://leetcode-cn.com/problems/delete-and-earn/)

可将问题转换成[198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        m = {}
        for num in nums:
            m[num] = m.get(num, 0) + num
        
        # m[key]为key对应的和，即cnt*key
        key_list = sorted(list(m.keys()))
        n = len(key_list)

        x, y = 0, m[key_list[0]]       
        for i in range(1, n):
            if key_list[i]-key_list[i-1]==1: # 间隔等于1， 对应198题的情况
                x, y = max(x, y), x+m[key_list[i]]
            else: # 间隔大于1，所以前一个数字无所谓选或不选
                x, y = max(x, y), max(x, y) + m[key_list[i]]
            
        return max(x, y)
```

时间复杂度：$O(m+nlogn)$，$m$为$nums$的长度，$n$为$nums$中不同元素的个数

空间复杂度：$O(n)$，哈希表所需大小为$O(n)$

#### [978. 最长湍流子数组](https://leetcode-cn.com/problems/longest-turbulent-subarray/)

和198. 打家劫舍思想差不多

```python
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        n = len(arr)
        # dp[i]表示以arr[i]为结尾的最长湍流子数组长度 dp[i][0]代表结尾升序，dp[i][1]代表结尾降序
        x, y = 1, 1
        res = 1
        for i in range(1, n):
            if arr[i]>arr[i-1]:
                x, y = y + 1, 1
            elif arr[i]<arr[i-1]:
                x, y = 1, x + 1
            else: # 相等
                x, y = 1, 1
            res = max(res, max(x, y))
        return res
```

时间复杂度： $O(n)$

空间复杂度：$O(1)$

#### [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

LCS的变形题

```python
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)
        n = len(nums2)

        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        res = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if nums1[i-1]==nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i-1][j])

                res = max(res, dp[i][j])
        return res
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)

```python
class Solution(object):
    def minDistance(self, word1, word2):
        m = len(word1)
        n = len(word2)

        # dp[i][j]表示word1[:i]和word2[:j]的最小步数
        dp = [[0 for i in range(n+1)] for i in range(m+1)]

        # dp init
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j

        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1
        
        return dp[m][n]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```python
class Solution(object):
    def minDistance(self, word1, word2):
        m = len(word1)
        n = len(word2)

        # dp[i][j]表示word1[:i]和word2[:j]的编辑距离
        dp = [[0 for i in range(n+1)] for i in range(m+1)]

        # dp init
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j

        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # 分别对应删除，插入，替换
                    dp[i][j] = min(dp[i-1][j], min(dp[i][j-1], dp[i-1][j-1])) + 1
        
        return dp[m][n]
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$

#### [629. K个逆序对数组](https://leetcode-cn.com/problems/k-inverse-pairs-array/)

观察得知$dp[i][j] = \sum^{j}_{j-i+1}dp[i-1][x]$

且有$dp[i][j-1] = \sum^{j-1}_{j-i}dp[i-1][x]$

有$dp[i][j] = dp[i-1][j] + dp[i-1][j-1] - dp[i-1][j-i]$

（可参考完全背包递推公式的化简）

```python
class Solution(object):
    def kInversePairs(self, n, k):
        sumn = n*(n-1)//2
        if k>sumn:
            return 0
        if k==0:
            return 1
        dp = [[0 for i in range(k+1)] for i in range(2)]
        dp[1][0] = 1

        for i in range(2, n+1):
            for j in range(k+1):
                if j==0:
                    dp[i&1][j] = 1
                else:
                    dp[i&1][j] = dp[(i-1)&1][j] + dp[i&1][j-1]
                    if j-i>=0:
                        dp[i&1][j] -= dp[(i-1)&1][j-i]

        mod = 1000000007
        return dp[n&1][k]%mod
```

时间复杂度：$O(n*k)$

空间复杂度：$O(k)$，可使用二维数组优化时间复杂度

#### [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

```python
class Solution(object):
    def findNumberOfLIS(self, nums):
        n = len(nums)
        
        # dp[i]是以nums[i]为结尾的最长递增子序列的个数
        dp = [0 for i in range(n)]
        # g[i]是以nums[i]为结尾的最长递增子序列的长度
        g = [0 for i in range(n)]

        dp[0], g[0] = 1, 1
        for i in range(1, n):
            dp[i] = g[i] = 1
            for j in range(i):
                if nums[j]<nums[i]:
                    if g[i]<g[j]+1: #找到了新的最长长度，更新dp和g
                        dp[i] = dp[j]
                        g[i] = g[j] + 1
                    elif g[i]==g[j]+1: # 找到了重复的最长长度 
                        dp[i] += dp[j]
        maxn = max(g)
        res = 0
        for i in range(n):
            if g[i]==maxn:
                res+=dp[i]      
        return res
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

#### [1218. 最长定差子序列](https://leetcode-cn.com/problems/longest-arithmetic-subsequence-of-given-difference/)

解法一：朴素dp

```python
class Solution(object):
    def longestSubsequence(self, arr, difference):
        n = len(arr)

        # dp[i]表示以arr[i]结尾的最长等差子序列长度
        dp = [0 for i in range(n)]
        for i in range(n):
            dp[i] = 1
            for j in range(i):
                diff = arr[i] - arr[j]
                if diff == difference:
                    dp[i] = max(dp[j] + 1, dp[i])
        return max(dp)
```

时间复杂度：$O(n^{2})$

空间复杂度：$O(n)$

解法二：dp+哈希表

为什么这题能用哈希表，因为每次我们要找的值是确定的（$arr[i]-diff$)，所以直接查表即可。

```python
class Solution(object):
    def longestSubsequence(self, arr, difference):
        n = len(arr)

        # dp[i]表示以arr[i]结尾的最长等差子序列长度
        dp = [0 for i in range(n)]
        m = {}
        res = 0
        for i in range(n):
            dp[i] = 1
            target = arr[i] - difference
            if target in m:
                j = m[target]
                dp[i] = dp[j]+1
            m[arr[i]] = i
            res = max(res, dp[i])
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [1713. 得到子序列的最少操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)

这题一眼LCS，但是LCS会超时，要利用题目的`target`元素各不相同，将其转换成LIS问题。

```
例：
target = [6,4,8,1,3,2]
arr = [4,7,6,2,3,8,6,1]

原问题 ==》求：target和arr的最长公共子序列长度LCS
最少操作次数 = target.length- lcs
时间复杂度：O(n*m)，target.length, arr.length <= 10^5，无法通过

由于target中的元素不重复，可将两个数组转成对应target的下标数组：
target': 0,1,2,3,4,5
arr': 1,0,5,4,2,0,3
问题 ==》求：target'和arr'的最长公共子序列长度LCS

由于target'是严格单调递增的
问题 ==》求：arr'的最长递增子序列的长度LIS
```

所以这里有一个重要结论：如果LCS的序列之一不存在重复元素，可以直接转换成LIS。

```python
class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        m = len(target)
        n = len(arr)

        mapping = {}
        for idx, num in enumerate(target):
            mapping[num] = idx
        
        arr_ = []
        for idx, num in enumerate(arr):
            if num in mapping:
                arr_.append(mapping[num])

        if len(arr_)<=1:
            return m-len(arr_)
        lcs = self.LIS(arr_)
        return m-lcs
    
    def LIS(self, arr):
        n = len(arr)

        # dp[i]表示以arr[i]结尾的最长递增子序列
        dp = [0 for i in range(n)]
        g = [float('inf') for i in range(n+1)]

        dp[0] = 1
        g[0], g[1] = float('-inf'), arr[0]

        res = 0
        for i in range(n):
            clen = self.find(g, arr[i], i+1)
            g[clen] = min(g[clen], arr[i])
            dp[i] = clen
            res = max(res, clen)
        return res
    
    def find(self, arr, target, end):
        l, r = 0, end
        while l<r:
            m = l+(r-l)//2
            if arr[m]>=target:
                r = m
            else:
                l = m+1
        return l
```

时间复杂度：$O(m+nlogn)$

空间复杂度：$O(m+n)$

评论区还有一个简化版本：

```python
class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        m, n = len(target), len(arr)
        num2idx = dict()
        for i, num in enumerate(target):
            num2idx[num] = i
        sequence = []
        
        # 直接得到LIS
        for num in arr:
            if num in num2idx:
                if not sequence or sequence[-1] < num2idx[num]:
                    sequence.append(num2idx[num])
                else:
                    idx = bisect.bisect_left(sequence, num2idx[num]) # 直接求二分
                    sequence[idx] = num2idx[num]
        return m - len(sequence)
```

时间复杂度：$O(m+nlogn)$

空间复杂度：$O(m+n)$

### 其它

#### [396. 旋转函数](https://leetcode-cn.com/problems/rotate-function/)

观察到$dp[0] = 0*nums[0] +...+(n-1)*nums[n-1]$

$dp[1] = 1*nums[0]+...+(n-1)*nums[n-2]+0*nums[n-1]$

$dp[n] = dp[n-1] + sum - n*nums[n-k]$

```python
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        n = len(nums)
        numSum = sum(nums)
        cur = 0
        for idx, num in enumerate(nums):
            cur+=idx*num
        res = cur # dp[0]
        for i in range(n-1, 0, -1): # 由dp[0]到dp[n]
            cur += numSum-n*nums[i]
            res = max(res, cur)
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

#### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])

        dp = [[0 for i in range(n)] for i in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                if i==0 or j==0:
                    dp[i][j] = 1 if matrix[i][j]=='1' else 0
                elif matrix[i][j]=='1':
                    dp[i][j] = min(dp[i-1][j], min(dp[i-1][j-1], dp[i][j-1])) + 1
                res = max(res, dp[i][j])
        
        return res*res
```

时间复杂度：$O(m*n)$

空间复杂度：$O(m*n)$



### 数学

#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/) （约瑟夫环）

f(n,m)在首轮删除了一个数后后变成了f(n-1, m)问题，只是初始位置发生了变化。

```python
class Solution(object):
    def lastRemaining(self, n, m):
        x = 0 # f(1)=0
        for i in range(2, n+1):
            x = (x+m)%i # f(n) = (f(n-1) + m) % n
        return x
```

#### [390. 消除游戏](https://leetcode-cn.com/problems/elimination-game/)

定义$f(n)$为从左到右删除后，从右到左的轮流删除，$f'(n)$为从右到左删除后，从左到右的轮流删除，分析可知两者对称，有$f(n)+f'(n) = n + 1$

在$f(n)$进行一次从左到右删除后，数组变成$2, 4, ..., x$，再从右到左删除，等价于$f'(\frac n{2})*2$，除号均取下界。

考虑以上两者，消除$f'(n)$后得到递推方程式$f(n) = 2(\frac n{2}+1-f(\frac n{2}))$

```python
class Solution(object):
    def lastRemaining(self, n):
        return 1 if n==1 else 2*(n//2+1-self.lastRemaining(n//2))
```

#### [357. 统计各位数字都不同的数字个数](https://leetcode-cn.com/problems/count-numbers-with-unique-digits/)

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n==0:
            return 1
        if n==1:
            return 10
        res, cur = 10, 9

        for i in range(n-1):
            cur*=(9-i)
            res+=cur
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(1)$

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

a ^ b 可以看作 a 和 b 的无进位加法

(a & b)<<1 可以看作a+b的进位
a^a = 0
a|(1<<i) ，将第i为置为1

a & (1<<i) ==0，a的第i位是否为1

a & b == 0 ，a和b是否正交

a & ((1<<b) -1)，只取a的低b位，即a & (2^b -1 ) 

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/) | Easy   | https://leetcode-cn.com/problems/number-of-1-bits/           |
| [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/) | Medium | https://leetcode-cn.com/problems/single-number-iii/          |
| [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/) | Medium | https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/ |
| [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/) | Medium | https://leetcode-cn.com/problems/divide-two-integers/        |

#### [191. 位1的个数](https://leetcode-cn.com/problems/number-of-1-bits/)

```python
class Solution(object):
    def hammingWeight(self, n):
        ret = sum(1 for i in range(32) if n & (1 << i))
        return ret      
```

#### [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

```python
class Solution(object):
    def singleNumber(self, nums):
        # 两个相同的数做异或必定为0
        xorsum = 0
        for num in nums:
            xorsum ^= num
            
        # l为xorsum最右边为1的位
        # 比如xorsum = 1100，l为0100
        l = xorsum & (-1*xorsum)

        type1, type2 = 0, 0
        for num in nums:
            # 两个只出现一次的数在l位上必定一个为0一个为1，所以可以借此把这两个数区分开
            # 而那些出现两次的数在做完异或后必定为0，所以不用管
            if num&l: 
                type1 ^= num
            else:
                type2 ^= num
        return [type1, type2]
```

#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

```python
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a&x, b&x
        while b:
            # 分别做无进位加法以及进位，每次的计算之后，a1+b1 = a2+b2，但是a越来越大，b越来越小，直到b为0，此时的a = a + b      
            a, b = a^b, ((a&b)<<1)&x 
        return a if a<=0x7fffffff else ~(a^x)
```

#### [29. 两数相除](https://leetcode-cn.com/problems/divide-two-integers/)

```python
class Solution(object):
    def divide(self, a, b):
        if a==-2**31 and b==-1:
            return (1<<31) - 1
        # 思路
        ## 找到最大的一个i，满足 (b<<i) < a
        ## 递归计算 divide(a-(b<<i), b)
        flag = True if (a<0 and b<0) or (a>0 and b>0) else False

        def dfs(a, b):
            # 默认a,b>0
            if a<b:
                return 0
            i=0
            while a>(b<<(i+1)):
                i+=1

            return (1<<i) + dfs(a-(b<<i), b)
        
        res = dfs(abs(a), abs(b))
        return res if flag else -1*res
```

## 图论

### 拓扑排序

#### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

解法一：dfs

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def dfs(i, adj, flags):
            if flags[i]==1: # 成环
                return False
            if flags[i]==-1: # 已访问过，避免重复搜索
                return True
            
            flags[i] = 1
            for j in adj[i]: 
                if not dfs(j, adj, flags):
                    return False
            flags[i] = -1
            return True 
        
        adj = [[] for _ in range(numCourses)]
        flags = [0 for _ in range(numCourses)]

        for cur, pre in prerequisites: # 构建邻接表
            adj[pre].append(cur)
        
        for i in range(numCourses): # 每个节点都不成环才为True
            if not dfs(i, adj, flags):
                return False
        return True
```

时间复杂度：$O(m+n)$，分别为建表和搜索所需时间

空间复杂度：$O(m+n)$

解法二：利用入度的拓扑排序

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        in_edges = [0 for _ in range(numCourses)]

        for cur, pre in prerequisites:
            adj[pre].append(cur)
            in_edges[cur] += 1
		
        # 将入度为0的结点放进队列
        queue = deque([edge for edge in range(numCourses) if in_edges[edge]==0])

        cnt = 0
        while queue:
            cnt+=1
            cur = queue.popleft()
            for node in adj[cur]:
                in_edges[node] -= 1
                if in_edges[node] == 0:
                    queue.append(node)
        
        return cnt==numCourses
```

时间复杂度：$O(m+n)$，分别为建表和搜索所需时间

空间复杂度：$O(m+n)$

# 特定类型题

## 数据结构题

#### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

```python
import heapq
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = [float('inf')]


    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(self.min_stack[-1], val))


    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()


    def top(self) -> int:
        return self.stack[-1]


    def getMin(self) -> int:
        return self.min_stack[-1]
```

时间复杂度：$O(1)$

空间复杂度：$O(n)$

## 组合/排列

## 基本计算器

#### [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

- 维护两个栈，nums和ops，分别用来保存数字和运算符
- 遇到左括号时直接加进ops
- 遇到右括号时，开始计算括号内的值（每次取出nums中的两个值和ops的一个运算符）并将结果加进nums
- 遇到+-直接加进ops
- 遇到数字，将连续数字加进nums

```python
class Solution:
    def calculate(self, s: str) -> int:
        s = s.replace(' ', '')
        n = len(s)
        nums = [0] # 处理第一个数为负数的情况
        ops = []

        def isdigit(char):
            return ord(char)>=ord('0') and ord(char)<=ord('9')

        idx = 0
        while idx<n:
            char = s[idx]
            if char=='(':
                # 左括号直接加
                ops.append(char)
            elif char==')':
                while ops[-1]!='(':
                    # 遇到右括号，开始计算括号内的值
                    self.cal(nums, ops)
                ops.pop() # pop出左括号
            elif isdigit(char): # 数字
                num = 0
                while idx<n and isdigit(s[idx]):
                    num = num*10 + int(s[idx])
                    idx+=1
                nums.append(num)
                continue
            else: # 运算符
                if s[idx-1]=='(': # 处理(-1+2)这种情况，也就是第一个数字为负数
                    nums.append(0)
                while len(ops)>0 and ops[-1]!='(' and len(nums)>1:
                    self.cal(nums, ops)
                ops.append(char)
            
            idx+=1

        while len(ops)>0 and len(nums)>1:
            self.cal(nums, ops)
        return nums[-1]
    

    def cal(self, nums, ops) -> None :
        if len(nums)<2 or len(ops)<1 or ops[-1] in '()':
            return
        
        num1 = nums.pop()
        num2 = nums.pop()
        op = ops.pop()

        res = num2-num1 if op=='-' else num1+num2
        nums.append(res)
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

#### [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

只有当ops尾的运算符优先级大于当前时才进行运算

```python
class Solution:
    def calculate(self, s: str) -> int:
        op_map = {
            '-': 1,
            '+': 1,
            '*': 2,
            '/': 2,
            '%': 3,
            '^': 3
        }
    
        def isdigit(char):
            return ord(char)>=ord('0') and ord(char)<=ord('9')

        s = s.replace(' ','')
        s = s.replace('(+','(0+')
        s = s.replace('(-', '(0-')

        ops, nums = [], [0]
        idx, n = 0, len(s)
        while idx<n:
            char = s[idx]
            if char=='(':
                ops.append(char)
            elif char==')':
                while len(ops)>0 and ops[-1]!='(':
                    self.cal(nums, ops)
                ops.pop()
            elif isdigit(char):
                num = 0
                while idx<n and isdigit(s[idx]):
                    num = num*10 + int(s[idx])
                    idx+=1
                nums.append(num)
                continue
            else:
                while ops and op_map[ops[-1]]>=op_map[char]:
                    self.cal(nums, ops)
                ops.append(char)
            
            idx+=1
        
        while len(ops)>0 and len(nums)>1:
            self.cal(nums, ops)
        res = nums[-1]
        if res<-1*(2**31):
            res = -1*(2**31)
        elif res>=1<<31:
            res = (1<<31)-1
        return res

    def cal(self, nums, ops):
        if len(nums)<2 or len(ops)<1 or ops[-1] in '()':
            return
        
        num2 = nums.pop()
        num1 = nums.pop()
        op = ops.pop()
        res = 0
        if op=='+':
            res = num1 + num2
        elif op=='-':
            res = num1 - num2
        elif op=='*':
            res = num1 * num2
        elif op=='/':
            res = num1//num2
        elif op=='%':
            res = num1%num2
        elif op=='^':
            res = num1**num2
        nums.append(res)
        return
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$



## 字符串匹配

### 字符串哈希

字符串哈希模板：

```python
s = "AAAABBBBAAAA"
n = len(s)

P = 131313
h = [0 for i in range(n+1)] # 保存字符串s的前缀哈希
b = [0 for i in range(n+1)] # 保存次方值

# 预处理，O(n)
b[0] = 1
for i in range(1, n+1):
    h[i] = h[i-1]*P + ord(s[i-1])
    b[i] = b[i-1]*P

# 得到子串s[i:j]的哈希值
def getHash(i, j):
    x, y = i+1, j
    return h[y] - h[x-1]*b[y-x+1]
```

| 题目                                                         | 难度   | 链接                                                     |
| ------------------------------------------------------------ | ------ | -------------------------------------------------------- |
| [187. 重复的DNA序列](https://leetcode-cn.com/problems/repeated-dna-sequences/) | Medium | https://leetcode-cn.com/problems/repeated-dna-sequences/ |
|                                                              |        |                                                          |
|                                                              |        |                                                          |

#### [187. 重复的DNA序列](https://leetcode-cn.com/problems/repeated-dna-sequences/)

解法一：滑动窗口+哈希表

```python
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        n = len(s)
        if n<10:
            return []        
        m = {}
        res = []
        for i in range(n-9):
            cur = s[i:i+10]
            if cur not in m:
                m[cur] = 1
            elif m[cur]==1:
                res.append(cur)
                m[cur]+=1             
        return res
```

时间复杂度：$O(n*C)$,$C$为DNA序列长度。

空间复杂度：$O(n*C)$，最差情况下每个序列都不相同，都要加入哈希表。

解法二：位运算+哈希表+滑动窗口

序列最长为10，并且只有ATCG四个符号，所以能用一个$2^{20}$的数字唯一的表示一个序列。

```python
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        n = len(s)
        if n<10:
            return []

        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        num = 0
        for i in range(10):
            num = (num<<2) | mapping[s[i]]

        m = {num: 1}
        res = []
        for i in range(10, n):
            num = (num<<2 | mapping[s[i]]) & ((1<<20)-1)
            if num not in m:
                m[num] = 1
            elif m[num] == 1:
                res.append(s[i-9:i+1])
                m[num] += 1  
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

解法三：字符串哈希

卡在了30/31

```python
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        n = len(s)
        if n<=10:
            return [] 
        
        P = 131313
        h = [0 for i in range(n+1)] # 保存字符串s的前缀哈希
        b = [0 for i in range(n+1)] # 保存次方值

        # 预处理，O(n)
        b[0] = 1
        for i in range(1, n+1):
            h[i] = h[i-1]*P + ord(s[i-1])
            b[i] = b[i-1]*P
        
        m = {}
        res = []
        for i in range(n-9):
            x, y = i+1, i+10
            h_v = h[y] - h[x-1]*b[y-x+1] # 计算s[i:j]的哈希值
            if h_v not in m:
                m[h_v] = 1
            elif m[h_v] == 1:
                res.append(s[i:i+10])
                m[h_v] += 1
        return res
```

时间复杂度：$O(n)$

空间复杂度：$O(n)$

## 其它

#### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        cnt = 0
        for idx, num in enumerate(nums):
            if cnt==0:
                candidate = num
                cnt = 1           
            elif num==candidate:
                cnt+=1
            else:
                cnt-=1
        
        return candidate
```



## 二叉树路径和

## 模拟题




# 脑筋急转弯题
