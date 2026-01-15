'''
编写一个Python函数，计算给定二维矩阵的转置。
矩阵的转置是通过将其行变为列，列变为行来形成的。
对于一个m×n矩阵，其转置将是一个n×m矩阵。

示例：
输入：
a = [[1, 2, 3], [4, 5, 6]]
输出：
[[1, 4], [2, 5], [3, 6]]
'''
def transpose_matrix_numpy(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    通过交换行和列来转置二维矩阵。
    参数：
        a: 形状为(m, n)的二维矩阵
    返回：
        形状为(n, m)的转置矩阵
    """
    result = [[0 for j in range(len(a))] for i in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[j][i] = a[i][j]

    return result


import torch

def transpose_matrix_pytorch(a) -> torch.Tensor:
    """
    使用PyTorch转置二维矩阵。
    
    参数：
        a: 二维矩阵（可以是列表、NumPy数组或torch.Tensor）
    
    返回：
        转置后的torch.Tensor
    """
    a_t = torch.as_tensor(a)
    return a_t.T


if __name__ == "__main__":
    # 测试用例1：基本示例
    a = [[1, 2, 3], [4, 5, 6]]
    print("输入:", a)
    print("输出:", transpose_matrix_numpy(a))
    # 预期输出: [[1, 4], [2, 5], [3, 6]]

    # 测试用例2：方阵
    b = [[1, 2], [3, 4]]
    print("\n输入:", b)
    print("输出:", transpose_matrix_numpy(b))
    # 预期输出: [[1, 3], [2, 4]]

    # 测试用例3：单行矩阵
    c = [[1, 2, 3, 4]]
    print("\n输入:", c)
    print("输出:", transpose_matrix_numpy(c))
    # 预期输出: [[1], [2], [3], [4]]

    # 测试用例4：单列矩阵
    d = [[1], [2], [3]]
    print("\n输入:", d)
    print("输出:", transpose_matrix_numpy(d))
    # 预期输出: [[1, 2, 3]]

    # 测试用例5：浮点数矩阵
    e = [[1.5, 2.5], [3.5, 4.5]]
    print("\n输入:", e)
    print("输出:", transpose_matrix_numpy(e))
    # 预期输出: [[1.5, 3.5], [2.5, 4.5]]

    # 测试用例6：空矩阵
    f = []
    print("\n输入:", f)
    try:
        print("输出:", transpose_matrix_numpy(f))
    except IndexError as e:
        print("错误:", e)
    # 预期输出: IndexError

    # 测试用例7：不规则矩阵（每行列数不同）
    g = [[1, 2, 3], [4, 5]]
    print("\n输入:", g)
    try :
        print("输出:", transpose_matrix_numpy(g))
    except Exception as e:
        print("错误:", e)
    
    # 测试PyTorch实现
    print("\n--- PyTorch实现测试 ---")
    print("输入:", a)
    print("输出:", transpose_matrix_pytorch(a))
    # 预期输出: tensor([[1, 4], [2, 5], [3, 6]])