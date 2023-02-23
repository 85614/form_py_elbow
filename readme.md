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

### 选项
新增use-cavity分支，使用cavity case，可以自己调整网格数量，网格简单，方便查看数据

在 constant/physicalProperties 可以修改hexfloat以确定是否使用十六进制浮点数打印

vscode 使用`alt+z`可是切换是否多行显示长行

使用正则表达式`^'''(.|\n)*?'''`查找，可以把生成的log里的大量 `''' xxx '''` 给选中，然后替换成空

### 使用cavity案例查看数据
使用use-cavity分支

在 cavity/constant/physicalProperties 可以修改hexfloat以确定是否使用十六进制浮点数打印

在 cavity/system/blockMeshDict 设置网格的密度，目前已将`20*20*1`改成了`2*3*1`

在 icoFoamPy/base_elbow.py  修改 `boundary_out_begin`和` boundary_out_end`，有说明
