## 目录说明
- icoFoam icoFoam求解器
- elbow elbow案例
- icoFoamPy 复现和测试python代码


## 简易使用

### 设置环境变量
就是编译安装OpenFOAM的教程里那个
https://openfoam.org/download/source/setting-environment/
```shell
source $HOME/OpenFOAM/OpenFOAM-dev/etc/bashrc
source $HOME/.bashrc
```

### 编译运行案例
全部都在这个脚本里
```shell
./m-r.sh
```
如果有什么问题，就手动执行一下里面的命令吧

icoFoam的日志文件 `elbow/log.icoFoam.exe` 已经被改成了一个可执行的python脚本

如果编译和运行icoFoam有什么问题，也可以直接查看`./bak/log.py`和`./bak/log.diff`。

分别是前面提到的`elbow/log.icoFoam.exe`及其执行结果的备份

### 其他
旧的master分支在old-master分支上,新的master分支,补充完善了中间变量边界部分的数据

目前已将cavity和elbow两个case用到的scheme都已经复现了,主要是以下两个scheme
```
divSchemes
{
    default         none;
    // div(phi,U)      Gauss linear;
    div(phi,U)      Gauss limitedLinearV 1;
}

laplacianSchemes
{
    // default         Gauss linear orthogonal;
    default         Gauss linear corrected;
}
```

除此之外,在cavity和elbow之间进行切换,以及修改elbow的scheme,并不需要再修改base_elbow.py才能正常运行

在 constant/physicalProperties 可以修改hexfloat以确定是否使用十六进制浮点数打印

vscode 使用`alt+z`可是切换是否多行显示长行

想要把的log里的大量 `''' xxx '''`删除的话,可以使用正则表达式`^'''(.|\n)*?'''`查找

在 cavity/system/blockMeshDict 设置cavity网格的密度，目前已将`20*20*1`改成了`2*3*1`
