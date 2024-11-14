import random
import math
import jittor as jt
from jittor.nn import MSELoss,L1Loss
import matplotlib.pyplot as plt
# 使用GPU进行计算
jt.flags.use_cuda = 1

class DynamicNet(jt.nn.Module):
    def __init__(self): 
        """
        模型初始化，定义5个参数位随机数
        """
        super().__init__()
        self.a = jt.randn(())
        self.b = jt.randn(())
        self.c = jt.randn(())
        self.d = jt.randn(())
        #TODO1：添加一个新的参数e
        self.e = jt.randn(())

    def execute(self, x):
        """
        模型的前向传播，定义了一个多项式函数，其中包含了5个参数
        y = a + b * x + c * x^2 + d * x^3 + e * x^4 ? + e * x^5 ? (?表示可能存在)
        """
        # todo 拟合曲线表达式
        y = self.a 
        y = y + self.b * x 
        y = y + self.c * x ** 2
        y = y + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        返回多项式模型的字符串表示
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'
    


# 生成数据集或者是获取数据集
x = 
y = 

#定义模型
model = DynamicNet()

#定义损失函数和优化器
loss_func = MSELoss()
# loss_func = jt.nn.L1Loss()
# learning_rate
learning_rate = 1e-5
#定义优化器，这里使用了SGD
optimizer = jt.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
for t in range(60000): # 训练60000次
    # 模型的前向传播，计算预测值
    y_pred = model(x)
    # 计算损失
    loss = loss_func(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())
    # jittor的优化器可以直接传入loss，自动计算清空旧的梯度，反向传播得到新的梯度，更新参数
    optimizer.step(loss)
 
#打印模型的参数
print(f'Result: {model.string()}')


# 生成简单的测试数据
x_test = 
x_test = 
# 计算x_test对应的预测值
with jt.no_grad():#进行预测时不需要计算梯度，所以使用no_grad，这样可以加快计算速度
    y_test_pred = model.execute(x_test)
plt.plot(x_test.numpy(), y_test_pred.numpy(), 'b', label='model')
plt.legend()
plt.show()