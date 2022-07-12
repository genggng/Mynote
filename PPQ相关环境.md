# PPQ相关后端环境配置

这个文件主要介绍了ppq相关的环境安装教程，以及tensorRT、openvino、snpe和ncnn的安装。

## 安装配置PPQ

ninja-build是ppq的内核编译依赖，需要提前安装。

#### 安装ninja-build

- 如果有root（sudo）权限，直接使用apt下载即可。

```shell
sudo apt update
sudo apt install ninja-build
```

- 对于实验室没有root权限，可以通过[源码安装ninja-build](https://zhuanlan.zhihu.com/p/321882707)。这部分是记录我踩得坑，不感兴趣的同学请直接看下一步**直接安装ninja-build**。

源码安装**需要先安装re2c**，在我的机器上发现存在错误，原来是**没有安装libtoolize**。

![without libtoolize](C:\Users\耿耿的笔记本\OneDrive - bupt.edu.cn\academic\code\Mynote\images\no_libtoolize.png)

没办法，只能源码[编译安装libtool](https://www.cnblogs.com/dakewei/p/10682596.html)，没报错可以跳过这一步。

```shell
# 获取源码，可以更换更新的源码包
wget http://ftpmirror.gnu.org/libtool/libtool-2.4.6.tar.gz
tar xvf libtool-2.4.6.tar.gz -C ~/
cd ~/libtool-2.4.6
# 配置编译文件，确定编译位置，记得修改
./configure --prefix=/home/geng/bin/libtool
# 编译源码
make -j4
# 安装
make install
# 添加环境变量
export PATH=/home/geng/bin/libtool/bin:$PATH
```

安装完libtool，就愉悦地安装re2c啦

```shell
# 获取源代码
wget https://github.com/skvadrik/re2c/releases/download/3.0/re2c-3.0.tar.xz
tar -xf re2c-3.0.tar.xz
cd re2c-3.0
# 编译安装
autoreconf -i-W all  #生成configure脚本
./configure --prefix=/home/geng/bin/re2c  #选择安装位置
make  #编译  make -j4 可以4个命令并行，一般可以使核心数的2倍，cad02是8核的
make install  #安装
# 添加环境变量
export PATH=/home/geng/bin/lre2c/bin:$PATH
```

接下来就可以顺利地编译安装ninja-build啦，但**实际上面的都不需要，ninja-build的github仓库存在编译好的二进制文件，可以直接下载使用**。

- **直接安装ninja-build**，上面的步骤都不用做，直接从这里开始。

```shell
# 获取二级制文件
wget https://github.com/ninja-build/ninja/releases/download/v1.11.0/ninja-linux.zip
unzip ninja-linux.zip
# 直接将解压的文件（所在文件夹）添加到环境变量即可
export PATH=/home/geng/bin/:$PATH
```

 #### 安装PPQ

直接按照[PPQ官方仓库](https://github.com/openppl-public/ppq)安装即可。 



## 安装ONNXRuntime

使用pip直接安装onnxruntime只能使用cpu推理，速度太慢。这里安装能够使用CUDA的版本。

```shell
pip install onnxruntime   #只支持cpu
pip install onnxruntime-gpu  #支持gpu，两个版本都要下载
```





## 安装配置Openvino

openvino是英特尔面向inel处理器（x86 cpu）和显卡的神经网络推理引擎，并且只支持酷睿6代以上或者至强处理器。

openvino的安装依赖较少，其python接口尤其容易安装。由于我只用到openvino的python接口，因此只介绍python接口的安装。

```shell
pip install openvino
python -c "from openvino.inference_engine import IECore"  #可以使用这条命令测试是否安装成功
```

一条命令即可，就是这么简单~



## 安装配置TensorRT

#### Pip wheel install

这里为了方便，我只安装TensorRT的Python接口。使用 **.deb** 安装完整版TensorRT包含pip wheel file，但是注意完整版安装的python wheel文件不可以脱离.deb安装独立使用。**这里介绍的方法使用pip安装，不必安装完整TensorRT，不携带C/C++接口，而只提供Python接口而且可以独立使用**。

根据[TensorRT官方安装向导](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)，pip安装TensorRT存在一些限制：[2022年7月10日,可能会更新]

> Python == 3.6 - 3.10
>
> CUDA ==  11.X
>
> Only the **Linux OS**  and **x86_64 CPU** architecture is supported
>
> CentOs >=7 or Ubuntu >= 18.04

首先需要安装**nvidia-pyindex**来保证能够获取NGC PypI 之外的python包。

```shell
python3 -m pip install --upgrade setuptools pip  #更新pip
python3 -m pip install nvidia-pyindex  #安装nvidia-pyindex
```

然后安装tensorRT

```shell
python3 -m pip install --upgrade nvidia-tensorrt
```

测试是否安装成功

```python3
import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())
```



#### Tar install

如果没有root权限想要安装完整版TensorRT，可以采用tar压缩包安装。

~~我本机上安装的是cuda11.4+cudnn8.2，因此下载cuda11.1+cudnn8.1的版本，本地的高版本能够兼容低版本~~

**高版本不能兼容低版本，必须安装符合条件的cuda和cudnn。**

[下载地址](https://developer.nvidia.com/nvidia-tensorrt-8x-download)，下载前需要注册登录nvidia账号。[TensorRT官方快速教程](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/quick-start-guide/index.html#install) [TensorRT官方安装向导](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading)

我下载的是 **TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz**，安装步骤如下

```shell
tar -xzvf TensorRT-8.2.1.8.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz  #解压缩

# 添加环境变量，可以将其添加到 ~/.bashrc中  注意最后面是你解压的TensorRT路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/geng/bin/TensorRT-8.2.1.8/lib

#安装TensorRT python包，注意切换到你想要的虚拟环境，并且一定注意选择你符合当前环境python版本的wheel，我这里用的python3.8
cd TensorRT-8.2.1.8/python 
python3 -m pip install tensorrt-8.2.1.8-cp38-none-linux_x86_64.whl

## （可选）如果你使用的Tensorflow，想要配合TensorRT使用，请安装 UFF wheel file，反之则跳过这一步
cd TensorRT-8.2.1.8/uff
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
which convert-to-uff  #检查是否安装成功uff

# 安装graphsurgeon包
cd TensorRT-8.2.1.8/graphsurgeon
python3 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

# 安装onnx-graphsurgeon包
cd TensorRT-8.2.1.8/onnx_graphsurgeon
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

# 执行样例文件,测试是否安装成功
cd TensorRT-8.2.1.8/samples/python
python common.py
```



## 安装配置SNPE

## 安装配置NCNN

待续......