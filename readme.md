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
