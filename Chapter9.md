# 9.现代循环神经网络

前一章我们介绍了RNN的基础知识，这种网络可以更好地处理序列数据。我们在文本数据上实现了基于RNN的语言模型，但是对于当今各种各样的序列问题，这些技术并不够用。

例如，RNN在实践中的一个常见问题是数值不稳定性。尽管我们已经用梯度裁剪等技巧缓解这个问题，但仍需要设计更复杂的序列模型来进一步处理它。

具体来说，我们将引入两个广泛使用的网络，即门控循环单元（gated recurrent units，GRU）和长短期记忆网络（long short-term memory，LSTM）。然后，我们将基于一个单项隐藏层来扩展循环神经网络架构。我们将描述具有多个隐藏层的深层架构，并讨论基于前向和后向循环计算的双向设计。现代循环网络经常采用这种扩展。在解释这些RNN的变体时，我们将继续考虑第8章中的语言建模问题。

事实上，语言建模只揭示了序列学习能力的冰山一角。在各种序列学习问题中，如自动语音识别、文本到语音转换和机器翻译，输入和输出都是任意长度的序列。为了阐述如何拟合这种类型的数据，我们将以机器翻译为例介绍基于RNN的“编码器-解码器”架构和束搜索，并用它们来生成序列。

## 9.1.门控循环单元（GRU）

在8.7.节中，我们讨论了如何在RNN中计算梯度，以及矩阵连续乘积可以导致梯度消失或梯度爆炸的问题。下面我们简单思考一下这种梯度异常在实践中的意义：

- 在某些情况下，早期观测值对预测所有未来观测值具有非常重要的意义。考虑一个极端情况：第一个观测值包含一个校验和，目标是在序列的末尾辨别校验和是否正确。在这种情况下，第一个词元的影响至关重要。我们希望有某些机制能够在一个记忆元里存储重要的早期信息。如果没有这样的机制，我们将不得不给这个观测值指定一个非常大的梯度，因为它会影响所有后续的观测值。
- 我们可能还会遇到这样的情况：一些词元没有相关的观测值。例如，在对网页内容进行情感分析时，一些辅助HTML代码与网页传达的情绪无关。我们希望有一些机制能够跳过隐状态表示中的此类词元。
- 我们可能会遇到这样的情况：序列的各个部分存在逻辑中断。例如，书的章节之间可能有过渡存在，或者证券的熊市和牛市之间可能会有过渡存在。在这种情况下，最好有一种方法来重置我们的内部状态。

学术界提出了许多方法解决这类问题，其中最早的方法是“长短期记忆”（long-short-term memory，LSTM）。门控循环单元（gated recurrent unit，GRU）是一个稍微简化的变体，通常能够提供同等的效果，并且计算速度明显更快。由于GRU相对简单，先从GRU讲起。

### 9.1.1.门控隐状态

门控循环单元与普通循环神经网络之间的关键区别在于：前者支持隐状态的门控。这意味着模型有专门的机制来确定该何时更新隐状态，以及该何时重置隐状态。这些机制是可学习的，并且能够解决上述问题。

例如，如果第一个词元非常重要，模型将学会在第一次观测后不更新隐状态。同时，模型也可以学会跳过不相关的临时观测。最后模型还将学会在需要的时候重置隐状态。

#### 9.1.1.1.重置门和更新门

首先介绍重置门（reset gate）和更新门（update gate）。我们把它们设计成（0，1）区间中的向量，这样我们就可以进行凸组合。

重置门允许我们控制“可能还想记住”的过去状态的数量；更新门允许我们控制新状态中有多少个是旧状态的副本。

我们从构造这些门控开始。下图描述了门控循环单元中的重置门和更新门的输入，输入是由当前时间步的输入和前一时间步的隐状态给出。两个门的输出是由使用sigmoid激活函数的两个全连接层给出。

![image-20250810102205551](./Chapter9.assets/image-20250810102205551.png)

我们来看一下门控循环单元的数学表达。对于给定时间步$t$，假设输入是一个小批量$\mathbf{X}_t \in \mathbb{R}^{n\times d}$（样本数量$n$，输入个数$d$），上一个时间步的隐状态是$\mathbf{H}_{t-1}\in \mathbb{R}^{n\times h}$（隐藏单元个数$h$）。那么重置门$\mathbf{R}_t\in \mathbb{R}^{n\times h}$和更新门$\mathbf{Z}_t\in \mathbb{R}^{n\times h}$的计算如下所示：
$$
\mathbf{R}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xr}+\mathbf{H}_{t-1}\mathbf{W}_{hr}+\mathbf{b}_r)\\
\mathbf{Z}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xz}+\mathbf{H}_{t-1}\mathbf{W}_{hz}+\mathbf{b}_z)
$$
其中$\mathbf{W}_{xr},\mathbf{W}_{xz}\in \mathbb{R}^{d\times h}$和$\mathbf{W}_{hr},\mathbf{W}_{hz}\in \mathbb{R}^{h\times h}$是权重参数，$\mathbf{b}_r,\mathbf{b}_z \in \mathbb{R}^{1\times h}$是偏置参数。

> 在这个式子中，$\mathbf{X}_t$是给定时间步的输入，代表有n个行，每行都是同一时间步的输入（或许是one-hot后的token）。
>
> $R_t/Z_t$的形状是$n\times h$，有n个行，每行代表每个隐藏单元的重置/更新情况。

#### 9.1.1.2.候选隐状态

接下来，让我们将重置门$\mathbf{R}_t$与常规隐状态更新机制集成，得到在时间步$t$的候选隐状态（candidate hidden state）$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$。
$$
\tilde{\mathbf{H}}_t=tanh(\mathbf{X}_t\mathbf{W}_{xh}+(\mathbf{R}_t\odot \mathbf{H}_{t-1})\mathbf{W}_{hh}+\mathbf{b}_h)
$$
其中$\mathbf{W}_{xh}\in \mathbb{R}^{d\times h}$和$\mathbf{W}_{hh}\in \mathbb{R}^{h\times h}$是权重参数，$\mathbf{b}_h\in \mathbb{R}^{1\times h}$是偏置项。符号$\odot$是Hadamard积（按元素乘积）运算符。在这里，我们可以使用tanh非线性激活函数来确保候选隐状态中的值保持在区间$(-1,1)$中。

与常规RNN中的隐状态相比，$\mathbf{R}_t$和$\mathbf{H}_{t-1}$的元素相乘可以减少以往状态的影响。每当重置门$\mathbf{R}_t$中的项接近1时，我们会得到一个与普通RNN无异的网络。对于重置门$\mathbf{R}_t$中所有接近于0的项，候选隐状态是以$\mathbf{X}_t$作为输入的MLP的结果。因此，任何预先存在的隐状态会被重置为默认值。

如图，说明了应用重置门后的计算流程。

![image-20250810105404912](./Chapter9.assets/image-20250810105404912.png)

> 先用H和X计算得到重置门R，再用R、H、X计算得到候选隐状态$\tilde{H}$。

#### 9.1.1.3.隐状态

上述计算结果只是候选隐状态，我们仍需要结合更新门$\mathbf{Z}_t$的效果。这一步确定新的隐状态$\mathbf{H}_t\in \mathbb{R}^{n\times h}$在多大程度上来自旧状态$\mathbf{H}_{t-1}$和新的候选状态$\tilde{\mathbf{H}}_t$。更新门$\mathbf{Z}_t$仅需要在$\mathbf{H}_{t-1}$和$\tilde{\mathbf{H}}_t$之间进行按元素的凸组合就可以实现这个目标。

因此，门控循环单元的最终更新公式是：
$$
\mathbf{H}_t=\mathbf{Z}_t \odot\mathbf{H}_{t-1}+(1-\mathbf{Z}_t)\odot \tilde{\mathbf{H}}_t
$$
每当更新门接近于1时，模型就倾向于保留旧状态。此时，来自$\mathbf{X}_t$的信息基本被忽略，从而有效地跳过了依赖链条中的时间步$t$。相反，当$\mathbf{Z}_t$接近于0时，新的隐状态$\mathbf{H}_t$就会更接近于候选隐状态$\tilde{\mathbf{H}}_t$。

这些设计可以帮助我们处理RNN中的梯度消失问题，并更好地捕获时间步距离很长的序列的依赖关系。例如，如果整个子序列的每个时间步的更新门都接近于1，则无论序列长度如何，在序列起始时间步的旧隐状态将很容易保留并传递到序列结束。

下图说明了更新门起作用后完整的计算流：

![image-20250810110445060](./Chapter9.assets/image-20250810110445060.png)

总之，GRU具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系
- 更新门有助于捕获序列中的长期依赖关系

### 9.1.2.从零开始实现

首先读取time machine 数据集。

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size,num_steps = 32,35
train_iter,vocab = d2l.load_data_time_machine(batch_size,num_steps)
```

#### 9.1.2.1.初始化模型参数

下一步是初始化模型参数。我们从标准差为0.01的高斯分布中提取权重，并将偏置设为0，超参数num_hiddens定义隐藏单元的数量，实例化与更新门、重置门、候选隐状态和输出层相关的所有权重和偏置。

```python
def get_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device) * 0.01
    def three():
        return (normal((num_inputs,num_hiddens))
                ,normal(num_hiddens,num_outputs)
                ,torch.zeros(num_hiddens,device=device))
    W_xz , W_hz,b_z = three()
    W_xr , W_hr,b_r = three()
    W_xh , W_hh,b_h = three()
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeors(num_outputs,device=device)
    params = [W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

#### 9.1.2.2.定义模型

现在我们将定义隐状态的初始化函数init_gru_state。此函数返回一个形状为(batch_size,num_hiddens)的张量，张量值全为0。

```python
def init_gru_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),)
```

现在准备定义门控循环单元模型，模型的架构与基本的RNN基本一致，只是更新公式更复杂。

```python
def gru(inputs,state,params):
    W_xz,W_hz,b_z,W_xr,W_hr,b_r,W_xh,W_hh,b_h,W_hq,b_q=params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz)+(H@W_hz)+b_z)
        R = torch.sigmoid((X @ W_xr)+(H @ W_hr)+b_r)
        H_tilda = torch.tanh((X@W_xh)+(H@W_hh)+b_h)
        H = Z*H +(1-Z)*H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)
```

#### 9.1.2.3.训练与预测

略

### 9.1.3.简洁实现

高级API包含了前文介绍的所有配置细节，所以我们可以直接实例化门控循环单元模型。这段代码的运行速度快得多，因为它使用的是编译好的运算符而非python来处理之前阐述的许多细节。

```python
num_inputs = vocab.size
gru_layer = nn.GRU(num_inputs,num_hiddens)
model = d2l.RNNModel(gru_layer,len(vocab))
```

### 9.1.4.小结

- GRU可以更好地捕获时间步距离很长的序列上的依赖关系。
- 重置门有助于捕获序列中的短期依赖关系
- 更新门有助于捕获序列中的长期依赖关系
- 重置门打开时，门控循环单元包含基本RNN；更新门打开时，GRU可以跳过当前子序列带来的隐状态的更新。



## 9.2.长短期记忆网络（LSTM）

隐变量模型一直存在着长期信息保存和短期输入缺失的问题。解决这一问题的最早方法之一是长短期存储器（long-short-term memory,LSTM）。它有许多与GRU一样的属性。长短期记忆网络的设计比GRU稍微复杂一些，却比GRU早出现了近20年。

### 9.2.1.门控记忆元

LSTM的设计灵感来自于计算机的逻辑门。LSTM引入了记忆元（memory cell）或称为单元（cell）。有些文献认为记忆元是隐状态的一种特殊类型，***它们与隐状态有相同的形状***，其设计目的是用于记录附加的信息。

为了控制记忆元，我们需要许多门。其中一个门用来从单元输出条目，我们将其称为输出门（output gate）。另外一个门用于决定何时将数据读入单元，我们将其称为输入门（input gate）。我们还需要一种机制来重置单元的内容，由遗忘门（forget gate）来管理，这种设计的动机与GRU相同，能够通过专用机制决定什么时候记忆或忽略隐状态的输入。

#### 9.2.1.1.输入门、遗忘门和输出门

与GRU相同，***当前时间步的输入和前一个时间步的隐状态***将作为数据送入LSTM的门中，如图所示，它们由三个具有sigmoid激活函数的全连接层处理，以计算输入门、遗忘门和输出门的值。因此，这三个门的值都在$(0,1)$的范围内。

> sigmoid将输入映射到$(0,1)$
>
> tanh将输入映射到$(-1,1)$
>
> ReLU将输入映射到$(0,+\infin)$

![image-20250810114240883](./Chapter9.assets/image-20250810114240883.png)

假设有$h$个隐藏单元，批量大小为$n$，输入数为$d$。因此，输入为$\mathbf{X}_t\in \mathbb{R}^{n\times d}$，前一时间步的隐状态为$\mathbf{H}_{t-1}\in \mathbb{R}^{n\times h}$。相应的，时间步的$t$的门定义如下：输入门是$\mathbf{I}_t \in \mathbb{R}^{n\times h}$，遗忘门是$\mathbf{F}_t \in \mathbb{R}^{n\times h}$，输出门是$\mathbf{O}_t \in \mathbb{R}^{n\times h}$，它的计算方法如下：
$$
\mathbf{I}_t = \sigma(\mathbf{X}_t\mathbf{W}_{xi}+\mathbf{H}_{t-1}\mathbf{W}_{hi}+\mathbf{b}_t)\\
\mathbf{F}_t = \sigma(\mathbf{X}_t\mathbf{W}_{xf}+\mathbf{H}_{t-1}\mathbf{W}_{hf}+\mathbf{b}_f)\\
\mathbf{O}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xo}+\mathbf{H}_{t-1}\mathbf{W}_{ho}+\mathbf{b}_o)
$$
其中$\mathbf{W}_{xi}$，$\mathbf{W}_{xf}$，$\mathbf{W}_{xo}\in\mathbb{R}^{d\times h}$和$\mathbf{W}_{hi},\mathbf{W}_{hf},\mathbf{W}_{ho}\in \mathbb{R}^{h\times h}$是权重参数，$\mathbf{b}_i,\mathbf{b}_f,\mathbf{b}_o\in \mathbb{R}^{1\times h}$是偏置参数。

#### 9.2.1.2.候选记忆元

先介绍候选记忆元（candidate memory cell）$\tilde{\mathbf{C}}_t\in \mathbb{R}^{n\times h}$。它的计算与上面描述的三个门的计算类似，但使用tanh函数作为激活函数，函数的值范围为$(-1,1)$。其在时间步$t$处的公式为：
$$
\tilde{\mathbf{C}}_t=tanh(\mathbf{X}_t\mathbf{W}_{xc}+\mathbf{H}_{t-1}\mathbf{W}_{hc}+\mathbf{b}_c)
$$
其中$\mathbf{W}_{xc}\in \mathbb{R}^{d\times h}$和$\mathbf{W}_{hc}\in \mathbb{R}^{h\times h}$是权重参数，$\mathbf{b}_c\in \mathbb{R}^{1\times h}$是偏置参数。

整体计算如图所示：

![image-20250810135926501](./Chapter9.assets/image-20250810135926501.png)

#### 9.2.1.3.记忆元

在门控循环单元中，有一种机制来控制输入和遗忘（跳过）。类似地，在LSTM中，也有两个门用于相同的目的：输入门$\mathbf{I}_t$控制采用多少来自$\tilde{\mathbf{C}}_t$的新数据，而遗忘门$\mathbf{F}_t$控制保留多少过去的记忆元$\mathbf{C}_{t-1}\in \mathbb{R}^{n\times h}$的内容。

使用按元素乘法，得：
$$
\mathbf{C}_t=\mathbf{F}_t\odot\mathbf{C}_{t-1}+\mathbf{I}_t\odot\tilde{\mathbf{C}}_t
$$
如果遗忘门始终为1且输入门始终为0，则过去的记忆元$\mathbf{C}_{t-1}$将随时间被保存并传递到当前时间步。引入这种设计是为了缓解梯度消失问题，以便更好地捕获序列中的长距离依赖关系。

这样就得到了计算记忆元的流程图：

![image-20250810141049106](./Chapter9.assets/image-20250810141049106.png)

#### 9.2.1.4.隐状态

最后需要定义如何计算隐状态$\mathbf{H}_t\in \mathbb{R}^{n\times h}$，在这里输出门发挥作用。在LSTM中，它仅仅是记忆元的tanh门控版本。这就确保了$\mathbf{H}_t$的值始终在区间$(-1,1)$内：
$$
\mathbf{H}_t=\mathbf{O}_t \odot tanh(\mathbf{C}_t)
$$
只要输出门接近1，我们就能有效地将所有记忆信息传递给预测部分，而对于输出门接近0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

如图，提供了数据流的图形化演示：

![image-20250810141603958](./Chapter9.assets/image-20250810141603958.png)

### 9.2.2.从零开始实现

首先加载time machine数据集。

```python
import torch
from d2l import torch as d2l
from torch import nn
batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)
```

#### 9.2.2.1.初始化模型参数

接下来需要定义和初始化模型参数。如前所述，超参数num_hiddens定义了隐藏单元的数量。我们按照标准差0.01的高斯分布初始化，并将偏置项设为0。

```python
def get_lstm_params(vocab_size,num_hiddens,device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    def three():
        return(
            normal((num_inputs,num_hiddens)),
            normal((num_hiddens,num_outputs)),
            torch.zeros(num_hiddens,device=device)
        )
    W_xi,W_hi,b_i = three()
    W_xo,W_ho,b_o = three()
    W_xf,W_hf,b_f = three()
    W_xc,W_hc,b_c = three()
    W_hq = normal((num_hiddens,num_outputs),device = device)
    b_q = torch.zeros(num_outputs,device=device)
    params = [W_xi,W_hi,b_i,W_xo,W_ho,b_o,W_xf,W_hf,b_f,W_xc,W_hc,b_c,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

> torch.randn()是生成在均值为0，方差为1的正态分布。
>
> torch.rand()是生成一个均匀分布的随机数，数值范围在[0,1）之间。
>
> torch.randint()是生成一个整数类型的随机数，数值范围在[low,high）之间。
>
> ```python
> torch.randint(low, high, size, dtype=None, device=None, requires_grad=False)
> ```
>
> torch.normal()与torch.randn()都是生成正态分布，但torch.normal可以指定均值和方差。

#### 9.2.2.2.定义模型

在初始化函数时，长短期记忆网络的隐状态需要返回一个额外的记忆元，单元的值为0，形状为(batch_size,num_hiddens)。

```python
def init_lstm_state(batch_size,num_hiddens,device):
    return (torch.zeros((batch_size,num_hiddens),device=device),
            torch.zeros((batch_size,num_hiddens),device=device))
```

实际模型定义与前述相同：提供三个门和一个额外的记忆单元。值得注意的是，只有隐状态会传递到输出层，而记忆元$\mathbf{C}_t$不参与输出计算。

```python
def lstm(inputs,state,params):
    W_xi,W_hi,b_i,W_xo,W_ho,b_o,W_xf,W_hf,b_f,W_xc,W_hc,b_c,W_hq,b_q=params
    (H,C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi)+(H@W_hi)+b_i)
        F = torch.sigmoid((X @ W_xf)+(H @ W_hf)+b_f)
        O = torch.sigmoid((X @ W_xo)+ (H @ W_ho) + b_o)
        C_tilda = torch.tanh( X@W_xc+H@W_hc+b_c)
        C = F*C + I*C_tilda
        H = O* torch.tanh(C)
        Y = (H@W_hq)+b_q
        outputs.append(Y)
    return torch.cat(Y,dim=0),(H,C)
```

#### 9.2.2.3.训练与预测

略

#### 9.2.3.简洁实现

使用高级API，我们可以直接实例化LSTM模型。与GRU相同，高级API中的LSTM使用的是编译好的运算符而非python，因此运行速度快得多。

```python
num_inputs = len(vocab)
lstm_layer = nn.LSTM(num_inputs,num_hiddens)
```

LSTM是典型的具有重要状态控制的隐变量自回归模型。多年来已经提出了其许多变体，例如：多层、残差连接、不同类型的正则化。然而，由于序列的长距离依赖性，训练LSTM和其他序列模型（如GRU）的成本很高。在后续内容中我们将介绍更高级的替代模型，如Transformer。

### 9.2.4.小结

- LSTM有三种门：输入门、遗忘门、输出门
- LSTM的隐藏层输出包括隐状态和记忆元。只有隐状态会传递到输出层，而记忆元属于内部信息。
- LSTM可以缓解梯度消失和梯度爆炸。



## GRU与LSTM小结

GRU与LSTM在某些地方较为相似，在此进行一个简单的总结

GRU：门控循环单元（gated recurrent unit）

LSTM：长短期记忆网络（long-short-term memory）

### GRU

GRU的隐藏层内部有两个新变量：重置门R和更新门Z。

在每个时间步$t$，GRU会根据该时间步的输入$X_t\in \mathbb{R}^{n\times d}$与上一时间步的隐变量$h_{t-1}\in \mathbb{R}^{n\times h}$计算重置门$\mathbf{R}_t\in \mathbb{R}^{n\times h}$和更新门$\mathbf{Z}_t\in \mathbb{R}^{n\times h}$，并应用sigmoid激活函数使其值域为$(0,1)$：
$$
R_t = \mathbf{sigmoid}(\mathbf{X}_t \mathbf{W}_{xr}+\mathbf{H}_{t-1}\mathbf{W}_{hr}+\mathbf{b}_r)\\
Z_t = \mathbf{sigmoid}(\mathbf{X}_t \mathbf{W}_{xz}+\mathbf{H}_{t-1}\mathbf{W}_{hz}+\mathbf{b}_z)\\
$$
随后，GRU会计算候选隐状态$\tilde{H}_t\in \mathbb{R}^{n\times h}$，应用tanh激活函数，其值域为$(-1,1)$公式为：
$$
\tilde{\mathbf{H}}_t = \mathbf{tanh}(\mathbf{X}_t\mathbf{W}_{xh}+(\mathbf{H}_{t-1}\odot \mathbf{R}_t)\mathbf{W}_{hh}+\mathbf{b}_t)
$$
在这一过程中，上一时间步的隐状态$\mathbf{H}_{t-1}$与重置门$\mathbf{R}_t$进行逐元素乘法。重置门$\mathbf{R}_t$的值越接近0，则说明需要丢弃之前的隐状态，则候选隐状态越接近于$\mathbf{X}_t$的MLP输出结果。

随后，GRU会根据上一时间步的隐状态$\mathbf{H}_{t-1}$、当前时间步的候选隐状态$\tilde{\mathbf{H}}_t$和更新门$\mathbf{Z}_t$计算当前时间步的隐状态$\mathbf{H}_t$：
$$
\mathbf{H}_t = \mathbf{H}_{t-1} \odot \mathbf{Z}_t+(1-\mathbf{Z}_t)\odot \tilde{\mathbf{H}}_t
$$
其中，更新门$\mathbf{Z}_t$越接近于0，则$\mathbf{H}_t$受候选隐状态的影响越大。

### LSTM

在LSTM的隐藏层内部有四个新变量：输入门$\mathbf{I}$、遗忘门$\mathbf{F}$，输出门$\mathbf{O}$和记忆元$\mathbf{C}$。

在每个时间步，LSTM会首先计算$\mathbf{I}_t\in \R^{n\times h}$，$\mathbf{F}_t \in \R^{n \times h}$,$\mathbf{O}_t\in \R^{n\times h}$。
$$
\mathbf{I}_t=\sigma(\mathbf{X}_t\mathbf{W}_{xi}+\mathbf{H}_{t-1}\mathbf{W}_{hi}+\mathbf{b}_i)\\
\mathbf{F}_t = \sigma(\mathbf{X}_t\mathbf{W}_{xf}+\mathbf{H}_{t-1}\mathbf{W}_{hf}+\mathbf{b}_f)\\
\mathbf{O}_t =\sigma(\mathbf{X}_t\mathbf{W}_{xo}+\mathbf{H}_{t-1}\mathbf{W}_{ho}+\mathbf{b}_o)
$$
与此同时，LSTM会计算候选记忆元$\tilde{\mathbf{C}}_t\in \R^{n\times h}$：
$$
\tilde{\mathbf{C}}_t=tanh(\mathbf{X}_t\mathbf{W}_{xc}+\mathbf{H}_{t-1}W_{hc}+b_c)
$$
随后，LSTM会根据当前时间步的遗忘门$\mathbf{F}_t$和输入门$\mathbf{I}_t$计算当前时间步的记忆元$\mathbf{C}_t\in \R^{n\times h}$。
$$
\mathbf{C}_t=\mathbf{I}_t \odot \tilde{\mathbf{C}}_t+\mathbf{F}_t\odot C_{t-1}
$$
最后用当前时间步的记忆元与输出门计算$\mathbf{H}_t\in \R^{n\times h}$：
$$
\mathbf{H}_t=\mathbf{O}_t \odot tanh(\mathbf{C}_t)
$$
在LSTM中，$\mathbf{C}_t$相当于另一个隐状态，但这个隐状态不会传递到输出层，只会随着时间步在隐藏层进行传递。