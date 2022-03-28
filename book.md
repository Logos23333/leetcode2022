按照题的类型，可以分成数据结构题，算法题，脑筋急转弯题。

对于一道新题，我们要做的是快速的分析出其类型，找到适用算法或数据结构，为此，我们需要知道每个算法的适用范围和数据结构的特点。

# 数据结构
## 哈希表
哈希表的思想在于用空间换时间，它访问Key的时间复杂度为O(1)。

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

### 前缀和

前缀和通常被用于“连续子序列之和/积”类型的题目中，它计算序列的前k个数之和并用哈希表存储。
它的思想是，任意连续子数组nums[i:j]之和都可以用total[j]-total[i]表示。
假设数组为nums，长度为n，我们想知道该数组存不存在和为target的“连续子数组”，用前缀和的模板如下：

```python
m = {0:-1} # 哈希表初始化
total = 0 # 当前前缀和
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

## 二叉树

### 完全二叉树

完全二叉树常被用来实现堆。用数组实现完全二叉树时有以下性质，idx节点对应的左节点索引为2\*idx + 1，右子树索引为2\*idx + 2。

###  二叉搜索树

二叉搜索树的性质是左节点值小于根节点，右节点值大于根节点。

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/) | Medium | https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/ |
|                                                              |        |                                                              |
|                                                              |        |                                                              |

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



### 遍历二叉树

需要熟练掌握二叉树的前序，中序，后序，层次遍历

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | Medium | https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/ |
| [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/) | Easy   | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/ |
| [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/) | Easy   | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/ |
| [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/) | Medium | https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/ |

#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        if len(preorder) == 0:
            return None
        root = preorder[0]
        left_length = inorder.index(root)
        root_node = TreeNode(root)
        root_node.left = self.buildTree(preorder[1:1+left_length], inorder[:left_length])
        root_node.right = self.buildTree(preorder[1+left_length:], inorder[left_length+1:])
        return root_node
```

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

### 路径总和题

| 题目 | 难度 | 链接 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



### 其它题

| 题目                                                         | 难度   | 链接                                                         |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------ |
| [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/) | Medium | https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/     |
| [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/) | Easy   | https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/ |
| [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/) | Easy   | https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/ |
|                                                              |        |                                                              |

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



## dfs
### 备忘录
### 剪枝
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
