import random
import numpy as np
import math
import jittor as jt
from jittor.nn import MSELoss,L1Loss
import matplotlib.pyplot as plt
# 使用GPU进行计算
jt.flags.use_cuda = 1

def calculators(omega, Tc, Pc, T):
    """
    定义计算器，用于计算多项式函数的值
    """
    # P = RT/(V-b)- a(T)/V(V+b)
    m = 0.480 + 1.574*omega-0.176*(omega**2)
    a = 0.42748*(R**2)*((Tc**2)/Pc)*(1+m*(1-(T/Tc)**0.5))**2
    b = 0.08664*R*Tc/Pc
    dict = {'a':a, 'b':b, 'm':m}
    return dict

def compressibility_factor(P,V,T):
    """
    定义计算器，用于计算压缩因子Z
    """
    P = np.array(P)
    V = np.array(V)
    T = np.array(T)
    Z = P * V / (R * T)
    return Z.tolist()

def calculate_A(T, P, a):
    """
    定义计算器，用于计算A
    """
    T = np.array(T)
    P = np.array(P)
    a = np.array(a)
    return (a * P) / ((R * T) ** 2).tolist()

def calculate_B(A,Z):
    """
    定义计算器，用于计算B
    """
    A = np.array(A)
    Z = np.array(Z) 
    a = Z.tolist()
    b = (Z + A).tolist()
    c = (-(Z**3) + Z**2 + A*Z).tolist()
    roots = []
    for i in range(len(Z)):
        coefficients = [a[i], b[i], c[i]]
        root = np.roots(coefficients)
        roots.append(root)
    return roots


class DynamicNet(jt.nn.Module):
    def __init__(self): 
        """
        模型初始化，定义5个参数位随机数
        """
        super().__init__()
        self.a = jt.randn(())
        # self.b = jt.randn(())
        # self.c = jt.randn(())
        # self.d = jt.randn(())
        #TODO1：添加一个新的参数e
        # self.e = jt.randn(())

    def execute(self, x):
        """
        模型的前向传播，定义了一个多项式函数，其中包含了5个参数
        y = a + b * x + c * x^2 + d * x^3 + e * x^4 ? + e * x^5 ? (?表示可能存在)
        """
        # todo 拟合曲线表达式
        y = self.a * x
        # y = y + self.b * x 
        # y = y + self.c * x ** 2
        # y = y + self.d * x ** 3
        # for exp in range(4, random.randint(4, 6)):
        #     y = y + self.e * x ** exp
        return y

    def string(self):
        """
        返回多项式模型的字符串表示
        """
        return f'y = {self.a.item()} P/RT'
        # return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'
    
def main():
    omega = input("请输入偏心因子:")
    global R
    R = 8.314
    Tc = input("请输入Tc:")
    Pc = input("请输入Pc:")
    T = []
    P = []
    V = []
    ab = calculators(omega, R, Tc, Pc, T)
    Z = compressibility_factor(P,V,T)
    A = calculate_A(T, P, ab['a'])
    B = calculate_B(A,Z)
    model = DynamicNet()
    #定义损失函数和优化器
    loss_func = MSELoss()
    # loss_func = jt.nn.L1Loss()
    # learning_rate
    learning_rate = 1e-5
    #定义优化器，这里使用了SGD
    optimizer = jt.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # 准备x和y
    P = np.array(P)
    T = np.array(T)
    x = (P/T).tolist()
    y = B
    for t in range(6000): # 训练6000次
        # 模型的前向传播，计算预测值
        y_pred = model(x)
        # 计算损失
        loss = loss_func(y_pred, y)
        if t % 2000 == 1999:
            print(t, loss.item())
        # jittor的优化器可以直接传入loss，自动计算清空旧的梯度，反向传播得到新的梯度，更新参数
        optimizer.step(loss)
    print(f'Result: {model.string()}')
    b = model.a.item()

if __name__ == '__main__':
    main()