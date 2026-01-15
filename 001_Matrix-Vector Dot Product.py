'''
编写一个Python函数，计算矩阵和向量的点积。
如果操作有效，函数应返回一个表示结果向量的列表；如果矩阵和向量维度不兼容，则返回-1。
只有当矩阵的列数等于向量的长度时，矩阵（列表的列表）才能与向量（列表）进行点积运算。
例如，一个n×m的矩阵需要一个长度为m的向量。

示例：
输入：
a = [[1, 2], [2, 4]], b = [1, 2]
输出:
[5, 10]
推理：
第1行：(1 × 1) + (2 × 2) = 1 + 4 = 5；第2行：(2 × 1) + (4 × 2) = 2 + 8 = 10

这是一个编程题目的说明，要求实现一个矩阵向量乘法函数。
'''
def matrix_dot_vector_numpy(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# 返回一个列表，其中每个元素是'a'的一行与'b'的点积。
    # 如果'a'的列数与'b'的长度不匹配，则返回-1。
	# 如果a和b的维数不兼容
	if (len(a[0]) != len(b)):
		return -1
	else:
		# 如果维数相同
		result = [0 for i in range(len(a))]
		for i in range(len(a)):
			for j in range(len(b)):
				result[i] += a[i][j] * b[j]
		return result

import torch

def matrix_dot_vector_pytorch(a, b) -> torch.Tensor:
    """
    使用PyTorch计算矩阵a和向量b的乘积。
    输入可以是Python列表、NumPy数组或torch张量。
    返回长度为m的一维张量，如果维度不匹配则返回tensor(-1)。
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    # 检查维数
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    # Your implementation here
    result = torch.matmul(a_t, b_t)
    return result

# 测试代码
if __name__ == "__main__":
    # 测试用例1
    a = [[1, 2], [2, 4]]
    b = [1, 2]
    print("NumPy实现:", matrix_dot_vector_numpy(a, b))
    print("PyTorch实现:", matrix_dot_vector_pytorch(a, b))

    # 测试用例2 - 维度不匹配
    a = [[1, 2, 3], [4, 5, 6]]
    b = [1, 2]
    print("NumPy实现(维度不匹配):", matrix_dot_vector_numpy(a, b))
    print("PyTorch实现(维度不匹配):", matrix_dot_vector_pytorch(a, b))
