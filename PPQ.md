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

