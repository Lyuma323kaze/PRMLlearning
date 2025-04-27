from graphviz import Digraph

# 创建一个有向图对象
dot = Digraph()

# 添加节点
dot.node('x', 'x')              # 输入 x
dot.node('W0', 'W0')              # 权重 W
dot.node('b0', 'b0')              # 偏置 b
dot.node('W1', 'W1')
dot.node('b1', 'b1')
dot.node('W2', 'W2')
dot.node('b2', 'b2')
dot.node('mul0', '×', shape='circle')  # 乘法节点
dot.node('mul1', r'×', shape='circle')
dot.node('mul2', r'×', shape='circle')
dot.node('add0', '+', shape='circle')  # 加法节点
dot.node('add1', '+', shape='circle')
dot.node('add2', '+', shape='circle')
dot.node('z1', 'z1')              # 中间变量 z
dot.node('z2', 'z2')
dot.node('fx', 'f(x)')
dot.node('relu1', 'ReLU1', shape='circle')# ReLU 激活函数
dot.node('relu2', 'ReLU2', shape='circle')
dot.node('a1', 'a1')              # 输出 y
dot.node('a2', 'a2')
# 添加边（箭头）
dot.edge('x', 'mul0')            # x 到乘法节点
dot.edge('W0', 'mul0')            # W 到乘法节点
dot.edge('mul0', 'add0')          # 乘法节点到加法节点
dot.edge('b0', 'add0')            # b 到加法节点
dot.edge('add0', 'z1')            # 加法节点到 z
dot.edge('z1', 'relu1')           # z 到 ReLU
dot.edge('relu1', 'a1')           # ReLU 到 y

dot.edge('a1', 'mul1')
dot.edge('W1', 'mul1')
dot.edge('mul1', 'add1')
dot.edge('b1', 'add1')
dot.edge('add1', 'z2')
dot.edge('z2', 'relu2')
dot.edge('relu2', 'a2')

dot.edge('a2', 'mul2')
dot.edge('W2', 'mul2')
dot.edge('mul2', 'add2')
dot.edge('b2', 'add2')
dot.edge('add2', 'fx')

# 渲染并保存图表
dot.render('MLP_plot', format='png', view=True)