from graphviz import Digraph


dot = Digraph()


dot.node('x', 'x')
dot.node('y', 'y')
dot.node('W0', 'W0')
dot.node('b0', 'b0')
dot.node('W1', 'W1')
dot.node('b1', 'b1')
dot.node('W2', 'W2')
dot.node('b2', 'b2')
dot.node('mul0', '×', shape='circle')
dot.node('mul1', r'×', shape='circle')
dot.node('mul2', r'×', shape='circle')
dot.node('add0', '+', shape='circle')
dot.node('add1', '+', shape='circle')
dot.node('add2', '+', shape='circle')
dot.node('z1', 'z1')
dot.node('z2', 'z2')
dot.node('relu1', 'ReLU1', shape='circle')
dot.node('relu2', 'ReLU2', shape='circle')
dot.node('a1', 'a1')
dot.node('a2', 'a2')
dot.node('L2', 'L2', shape = 'circle')
dot.node('J', 'J')
# 添加边（箭头）
dot.edge('x', 'mul0')
dot.edge('W0', 'mul0')
dot.edge('mul0', 'add0')
dot.edge('b0', 'add0')
dot.edge('add0', 'z1')
dot.edge('z1', 'relu1')
dot.edge('relu1', 'a1')

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

dot.edge('fx', 'L2')
dot.edge('y', 'L2')
dot.edge('L2', 'J')


dot.render('MLP_plot', format='png', view=True)