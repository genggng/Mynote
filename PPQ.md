# PPQ

- 神经网络推理延迟和吞吐量并不是反比的关系。很强的显卡吞吐量很大，可能也有很高的延迟；很小的专用芯片，吞吐量不大，但是可能延迟很低。这个类似于网络通信中的延迟和带宽的关系。
- 使用torch.profiler统计网络运行过程中，各操作占用的时间比例，以及cpu和gpu的利用率，从而找到网络瓶颈。当gpu利用率很低时，同时copy操作占用很多，此时的瓶颈在IO，可以考虑增大batchsize，同时尽量调大num_work。

## 量化方案

量化方案是PPQ中的核心，针对不同的**后端**定制不同的量化方案，但是对模型本身并没有太多的特异的优化。

```Python
PLATFORM = TargetPlatform.PPL_CUDA_INT8 # 对不同的后端使用不同的量化方案。
```

量化方案主要是决定三件事：

- 如何量化（确定scale的方法，对称非对称量化等）
- 哪些量化（哪些算子能够量化，哪些算子不需要量化）
- 图调度和融合如何做（实际部署推理中，需要做调度将网络划分为子图送到不同的设备，并且将算子进行合并）

## 调度器

网络部署到终端，第一步先做调度，将网络切成子图，送到不同的设备上。比如说移动芯片中dsp不支持一些算子，需要移出到cpu，解决不能算的问题。同时解决能不能量化的问题。

```python
quant_setting.dispatcher  #定义调度器
```

## 实现细节

BaseQuantizer类掌控了PPQ量化的整体流程，其中step2是拆算子，会改变网络的结构；step3则是对各种算子进行量化。

```python
executor.tracing_operation_meta(inputs=inputs) #获得算子的输出输出尺寸，数据类型等
```

其中**BaseQuantizer**基类定义了**quantize_operations** 只是初始化了量化方案信息，其中调用的虚函数**init_quantize_config**必须由不同后端平台的子类实现。

```python
TensorRTQuantizer()  #继承BaseQuantizer类,根据后端平台（TRT）确定量化方案，实现父类的虚函数
TensorQuantizationConfig  #定义了量化config，最核心的东西，决定所有量化方案。在/ppq/core/quant.py
```

在**/ppq/api/seeting.py**中，有ppq的各种setting，用户设置什么就加什么东西pass，比如添加SSD权重均衡化,

 在**/ppq/quantization/optim/**中，有ppq的各种pass，即管道过滤器。相当于各种功能的模块化，可以以流水的方式插入组合。

其中最核心的有baking.py, base.py, calibration.py, morph.py, parameters.py, refine.py这个6个文件。在一般的量化过程中，一定会包含一下7个pass。

- **PPQ Quantization Config Refine Pass** # 决定算子能不能量化
- **PPQ Quantization Fusion Pass**  # 图融合的pass
- **PPQ Quantize Point Reduce Pass**  # 关闭中间量化，上一层输出和下一层输入量化一次
- **PPQ Parameter Quantization Pass**  #对参数进行量化，计算scale
- **PPQ Quantization Alignment**  #量化对齐：对于多个输入的算法，比如说concat操作，需要要拼接的多个数值的scale是一致的。
- **PPQ Passive Parameter Quantization Running**  #一些不该量化的，比如说reshape算子中的size信息，就不需要量化。
- **PPQ Parameter Baking Pass Running**  #对所有权重参数改为量化版本，从而加速模拟量化。

剩下的pass能够优化量化的精度，但是不可能提速。量化方案决定后，速度就确定了，pass优化只能降速。

ppq最关键的state，整个流程就是这些状态的变化

```python
INITIAL   = 1  # 量化参数刚刚被初始化，当前 config 不生效，数据不能被使用
BAKED    = 2  # 只针对参数量化，表示参数已经被静态量化，当前 config 不生效，数据可以直接使用
OVERLAPPED  = 3  # 只针对activation量化，表示数据流的量化由其他 config 管理，当前 config 不生效
SLAVE    = 4  # 只针对activation量化，表示当前数据流处于强制联合定点，当前 config 生效
DEACTIVATED = 5  # 表示当前 config 不生效
ACTIVATED  = 6  # 表示当前 config 生效
DEQUANTIZED = 7  # 表示当前 config 处于解量化状态，解量化是 PPQ 的一个系统操作
SOI     = 8  # 表示这一路输入与 Shape or index 相关，不量化
PASSIVE   = 9  # 表示这一路输入被动量化，如 bias, clip value 等
PASSIVE_INIT = 10  # 表示这一路输入被动量化，并且刚刚初始化不能被使用
PASSIVE_BAKED = 11 # 被动量化且静态量化，当前config不生效，数据可以直接使用
FP32     = 12  # 表示这一路输入直接为FP32浮点数
```

## 神经网络量化硬件实现

- 硬件上round取整函数一般有5种,主要是遇到.5时如何处理，小于.5向下取整，大于.5向上取整：
  - Round half to even: 向偶数取整 round(1.5) = 2  round(2.5) = 2  round(-0.5) = 0
  - Round half away from zero: 向零取整 round(2.5) = 2    round(-2.5)=-2
  - Round half toward zero: 反向零取整 round(2.5) = 3  round(-2.5) = -3
  - Round half down:  向下取整  round(2.5) = 2   round(-2.5) = -3
  - Round half up:  向上取整  round(2.5) = 3  round(-2.5) = -2

- 网络中部分算子可能不能量化，如果在中间插入**反量化解量化**进行单独的FP32推理，可能会比完全的FP32网络要更慢。尽量保证网络中的所有算子都是能够量化的。

####  对称量化

原始的FP32和量化后的INT8范围都是对称的，不需要零点，但是INT8范围中的-128会浪费。

```c++
float value = 1.0; // 待量化浮点值 
float scale = 0.1;  //尺度因子
int qt32 = round_fn(value/scale);  //截断前的量化值，round_fn为round函数
char qt8 = clip(qt32,Q_MIN,Q_MAX);  // 量化值，Q_MIN=-127 Q_MAX=127
```

#### 非对称量化

但是考虑到relu函数后的feature只可能是正数，这就浪费了很大的空间。可以对relu激活值可以采用非对称量化，INT8范围只采用正数。方法就是对应原来的对称量化，在尺度放缩后加上一个偏移量（零点），将整数值都便宜到正数区间。

```c++
float value = 1.0; // 待量化浮点值 
float scale = 0.1;  //尺度因子
int qt32 = round_fn(value/scale + zero_point);  //截断前的量化值，round_fn为round函数,zero_point为偏移零点
unsigned char qt8 = clip(qt32,Q_MIN,Q_MAX);  // 量化值 Q_MIN=0 Q_MAX=255
```

#### 整数量化

一些设备上不支持浮点除法，因此上面方式的qt32就无法计算。因此使用位移运算代替，但显然就只能进行2的倍数的乘除。

```c++
float value = 1.0; // 待量化浮点值 
int shift = 1;  //定点位
int qt32 = round_fn(value<<shfit);  //截断前的量化值，round_fn为round函数
char qt8 = clip(qt32,Q_MIN,Q_MAX);  // 量化值，Q_MIN=-127 Q_MAX=127
```

####  量化加速的原因

- 整数(int32)乘法和int8乘法本身并不比浮点（FP32）计算快，**关键是int8显著地降低了访存开销**。

- 而在处理器中，为了大规模计算，通常会设置专用的向量化运算指令（AVX512）或者处理器（TensorCore）。

  正常16个数字（FP32）相加需要至少16次add指令，但是使用向量化指令加速，转为int32加法就只用5次，使用int8加法就只用一次。这是因为AVX512指令集最大支持512bit并行运算，16个int8刚好就是能够一次算完。

#### 量化乘法

量化乘法的关键就是讲量化值反量化为浮点值，进行乘法后再转为量化值。但是注意这个过程是融合在乘法算子（公式）中的，并没有显式的量化反量化操作。这是rescale的操作，将两个不同scale的操作数，经过乘法计算结果统一到一个scale。

```c++
void Mul(char **input_a, char **input_b, char **output, const unsigned int num_elements,
        const float scale_a, const float scale_b, const float scale_c){
    /*
    input_a,input_b 为输入的int8矩阵，output为输出的乘积，也为int8矩阵。
    scale_a, scale_b, scale_c 为校准时确定的量化参数
    num_elements为矩阵的维数，假设为方阵
    这里展示的是对称量化，对于非对称量化只需要替换反量化公式即可
    output = (input_a - offest_a)*scale_a * (input_b - offest_b)*scale_b / scale_c + offest_c
    */
    for(unsigned int i=0; i<num_elements; i++){
        for(unsigned int j=0; j<num_elements; j++){
            output[i][j] = clip(round_fn(input_a[i][j]*scale_a * input_b[i][j]*scale_b / scale_c)) 
		}
    }
}
```

#### 量化加法

量化加法操作和乘法类似，但是存在一个问题，不能像乘法一样提前算好$\frac{S_a S_b}{S_c}$ ,只用计算$A*B$，只多引入了一次乘法。 但是对于add来说，计算从简单的浮点 $A+B$ 转为了复杂的$(A*S_a + B*S_B)/S_c$ ，不仅增加了两次乘法，而且还引入了非常慢的除法运算。

因此我们要求：**两个加法的scale_a, scale_b必须相同，如果不一致就强制转为一致**。这样加法运算就可以简化为$(A+B)*\frac{S_{ab}}{S_c}$, 也是只多引入了一次乘法运算。

```c++
void Add(char **input_a, char **input_b, char **output, const unsigned int num_elements,
        const float scale_a, const float scale_b, const float scale_c){
    /*
    input_a,input_b 为输入的int8矩阵，output为输出的和，也为int8矩阵。
    scale_a, scale_b, scale_c 为校准时确定的量化参数
    num_elements为矩阵的维数，假设为方阵
    */
    for(unsigned int i=0; i<num_elements; i++){
        for(unsigned int j=0; j<num_elements; j++){
            output[i][j] = clip(round_fn((input_a[i][j]*scale_a + input_b[i][j]*scale_b) / scale_c)) //无法提公因式
		}
    }
}
```

#### 被动量化算子

clip截断函数本身本质上并没有对输入进行计算，换句话说并没有对输入进行映射变换，而只是发生了数值的截断。因此，我们要求**clip函数的输入和输出scale应该一致**。这样clip函数就可以简化，将截断边界值（min,max）进行量化, 在量化区间上进行截断。

```c++
const char max = 10.0  //截断边界是常量，在浮点区间有意义
const char min = -10.0
output[i][j] = MAX(input[i][j],min/scale)   
output[i][j] = MIN(input[i][j],max/scale)
```

注意，这里我们对截断边界进行量化，使用的scale是来自输入，其本身不具备自己的量化参数。因此我们称min是**被动量化的**。

**像这种输入和输出共享scale，同时算子的运算不改变量化参数，我们称之为被动两户算子**。常见的被动量化算子有：

> Pad, Clip, Relu, Maxpooling, Reshape, Concat, Split, Transpose, Slice, Permute



#### 量化矩阵乘法

矩阵乘法是一个非常关键的运算，包括Conv，Transformer都是由矩阵乘法实现。

下图中，绿色的部分是int8计算，黄色的部分是int32计算，红色的部分可能就为int64计算。整个计算过程，使用高bit的int32整数保存中间结果，得到最终的结果时可能已经是int64，但最终都要rescale到int8。

我们要求$WX$，具体实现为：每次从$X$取4行，从$W$取4列，送到L2缓存上去。然后进行分块矩阵乘法，得到一个$4\times4$的小矩阵块。由于两个8位整数相乘是16位，并且矩阵乘法存在累加的过程，因此这里使用int32存储中间结果（accumulator）。最终将分块结果与bias相加，得到输出的结果。由于accumulator采用int32，bias也采用int32，为了防止溢出相加结果可能采用int64，但最终还是会rescale到int8作为最终结果。

![量化矩阵乘法实现](C:\Users\耿耿的笔记本\OneDrive - bupt.edu.cn\academic\code\Mynote\images\MatMul.png)

```c++
output[i+k][j+0] = Clip(round((packedA[0]*packedB[0] + bias[i+k])*S_a*S_b/S_c)));
/*
1. int8矩阵乘在累加器中，结果为int32
2. 处理bias add，结果为int32
3. 执行rescale，结果为fp32
4. 取整，截断，结果为int8
*/
```



#### 量化非线性运算

一些算子，比如指数函数，对数函数，内部包含非线性运算，无法靠整数计算，不可以直接量化，这些算子有：

> Exp, Tanh, Sigmoid, Softmax, Swish, Resize

在CPU和GPU这种具备浮点运算能力的硬件上，这类算子不进行量化，以全精度模型进行。

在FPGA，ASIC，DSP上，需要更改算子的计算逻辑，使用线性运算拟合（泰勒展开）或者直接查表（枚举输入x的256种可能与scale的十几种可能，最终也就几千种结果，输入与输出的映射可以直接用哈希表存储）。

#### 量化口诀

> 量化计算量化算，中间结果精度高。
>
> 中间算完转尺度，转完尺度取整数。
>
> 加法减法不能转，被动算子也一样。
>
> 非线性函数查表算，不然你就等死吧。



## 神经网络图优化与量化模拟

#### 计算图

一个计算图可以表示为一个由节点、边集、输入边、输出边组成的四元组：$ C = \{N,E,I,O\}$ 

- 计算图必须是一个有向联通无环图，其中节点也称为**算子**。
- **算子必定有边相连，输入边，输出边不为空**。
- 计算图可以有重边。

#### 算子

算子是神经网络的核心，神经网络的功能就通过一系列的算子实现。[onnx](https://github.com/onnx/onnx/blob/main/docs/Operators.md)中支持了一百多中算子，一般的神经网络中常用的算子也就十几种，比如Conv，Clip，Add等。

算子是神经网络的最小调度单位，虽然一个复杂的算子可以被更细粒度的算子所表示，但是推理框架总是以算子为单位去支持网络。在划分子图和调度的是否，不会将一个算子再进行拆分。**为了部署友好，网络中应该尽量避免使用特殊的算子**。

#### 算子（图）融合加速

将多个算子合并融合进行加速。比如在MatMul + Bias + Relu的子网中，如何不融合算子，output至少要被写入三次（内存），并且因为每个算子（函数）都需要指令发射-读操作-计算-写操作，启动三次算子的速度也不是很快。**图融合能够降低访存和算子开销**。

![算子融合](C:\Users\耿耿的笔记本\OneDrive - bupt.edu.cn\academic\code\Mynote\images\graph.png)

常见的图优化有：

- **激活函数融合**

  

- **移除BN层和Dropout**

- **常量折叠**

- **矩阵乘融合**

- **Conv-Add融合**