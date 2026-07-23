# trajectory-lab

[English](README.md) | [简体中文](README.zh-CN.md)

面向机器人与低速车辆的确定性二维路径规划、后处理、时间参数化和轨迹验证工具。项目仅使用 Python 标准库，可在没有 GPU、仿真平台或实体硬件的环境中完成完整实验。

## 核心能力

- 栅格地图与可读 ASCII 场景；
- 确定性最优 A*、Dijkstra 和航向感知状态格规划；
- 连续线段与占用单元的精确碰撞检测；
- 带碰撞保护的捷径化与曲率感知平滑；
- 速度、纵向加速度和横向加速度约束下的时间参数化；
- 结构化验证报告，以及 `plan`、`validate` 两个 CLI 入口。

## 快速开始

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python -m trajectory_lab plan `
  --map examples\maps\warehouse.txt `
  --algorithm astar --shortcut `
  --output path.csv --report report.json
```

规划结果包含位置、弧长、速度、时间和曲率。输入错误、无路径和轨迹验证失败分别使用不同退出码，便于集成到自动化测试流程。

## 工程边界

项目适合算法教学、CPU 仿真和回归验证，不是经过认证的机器人安全组件。几何与动力学约束严格对应离散输入模型，不能替代真实传感、定位和执行器测试。

## 协作

史浩轩负责总体设计与主要实现；刘泽康参与安全边界和集成接口核验。职责说明见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

采用 [MIT License](LICENSE)。
