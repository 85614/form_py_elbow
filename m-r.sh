#!/bin/sh

cd ${0%/*} || exit 1    # Run from this directory

# source ~/OpenFOAM/foam-source

cd icoFoam

wmake # 编译

cd ../cavity

./Allclean # 清理

./Allrun # 运行icoFoam 会生成日志文件 log.icoFoam.exe

python log.icoFoam.exe # 这个日志文件是一个python脚本，直接运行

# python log.icoFoam.exe > log.diff # 重定向记得输入'n'以取消手动控制