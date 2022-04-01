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
            else: # 必定在左边，# 必定在右边，如[3,4,1,2]中的1或2
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



## dfs
### 博弈论

### 状态压缩

从数组里不放回的取出元素遍历，如何表示数组的当前状态state？

假设数组长度为n，并且n<32，可以用一个2**(n)大小的整数state来表示表示数组的状态。

state的第i位为0时代表数组的第i位元素未被使用，为1时代表已被使用。

这样的好处是，state是数字，很方便存储，而且可被哈希,可以用哈希表优化dfs速度。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/) | Medium | https://leetcode-cn.com/problems/beautiful-arrangement/      |
| [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/) | Medium | https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/ |
|                                                              |        |                                                              |

#### [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/)

```python
class Solution:
    def countArrangement(self, n: int) -> int:
        m = {}
        def dfs(state, path):
            """
            # state: 用来记录n个整数的使用情况
            # path: 当前排列
            """
            if len(path)==n:
                return 1 
            if state in m:
                return m[state]

            res = 0
            for i in range(1, n+1):
                cur = 1<<(i-1) 
                if cur & state!=0: # 当前数字已被使用
                    continue

                cur_length = len(path) + 1
                if cur_length%i ==0 or i%cur_length==0: # 题目要求                    
                    res+= dfs(state|cur, path+[i], m) # state|cur是将对应位数置为1
            m[state] = res
            return res
        
        res = dfs(0, [], m)
        return res 
```

#### [698. 划分为k个相等的子集](https://leetcode-cn.com/problems/partition-to-k-equal-sum-subsets/)

```python
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        sum_nums = sum(nums)
        if sum_nums%k!=0 or len(nums)<k:
            return False
        
        length = int(sum_nums/k)
        if max(nums)>length:
            return False
        
        m = {}
        def dfs(state, cur_length, length, num_left, m):
            '''
            # state: 2*n，state的第i位为1代表nums的第i个数被用过
            # cur_length: 当前的长度
            # length: 每个子集的总和
            # num_left：还剩下多少个子集
            '''
            if cur_length > length:
                return False
            if cur_length == length:
                return dfs(state, 0, length, num_left-1, m)
            if num_left==0 and state == (1<<len(nums)) - 1:
                return True
            
            if state in m:
                return m[state]
            for i in range(len(nums)):
                cur = 1<<i
                if cur & state != 0:
                    continue
                if dfs(cur|state, cur_length+nums[i], length, num_left, m):
                    return True
            
            m[state] = False
            return False
        
        return dfs(0, 0, length, k, m)
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

用数组minn 存放 `min(dp[:i] + dp[i+1:])$,这样的好处是每次计算dp时非常方便。

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
|                                                              |        |                                                              |

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



# 特定类型题
## 组合/排列
## 子序列
## 字符串匹配

## 二叉树路径和

## 模拟题




# 脑筋急转弯题
