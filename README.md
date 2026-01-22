# midrangeRL

二维中距空战 RL 训练平台 —— 兼具科普与强化学习研究价值

## 核心特性

- **时空图(Space Time Graph/STG)**：直观展示导弹真实射程，辅助战斗决断
- **大规模并行训练**：单次支持数万个环境并行训练，并在每个环境中实现1000倍甚至更高倍率的时间加速
- **冷启动**：通过规则式 agent 引导智能体学习，加快训练收敛

## 快速开始

```bash
pip install -r requirements.txt
python game_play.py   # 试玩游戏
```

**操作：** 红方 `A/D` 转向 `T` 开火；蓝方 方向键 转向 `=` 开火；`V` 切换视图

## 项目结构

```
midrangeRL/
├── env_warp/       # GPU 并行 RL 环境
├── env_numpy/     # CPU 单例环境，可以用来试验和开发仿真环境的新特性
├── agents/        # Agent 接口与实现
│   ├── rule_based/     # 规则式智能体（用于冷启动）
│   └── learned/        # 被训练智能体
├── spacetime/       # 时空图渲染模块
│   ├── spacetime_renderer.py  # 渲染时空图
│   └── spacetime_core.py  # CPU模式的时空图计算
├── train_warp.py       # 训练入口
├── game_play.py   # 游戏主程序
└── config.py      # 配置文件
```

## 训练脚本

有时空图：
```bash
python train.py --agent STG_ppo1 --opponent STG_rule_based --reward zero --num-envs 256 --time-scale 4.0
```

无时空图：
```bash
python train.py --agent ppo_agent --opponent crank --reward zero --num-envs 256 --time-scale 4.0
```

支持参数：`--agent`、`--opponent`、`--reward`、`--num-envs`、`--time-scale`、`--device`

## RoadMap

重大改变：改用Nvidia Warp进行高并行环境仿真，这将成为本项目的核心特色之一

近期feature：
- [x] 开发能在手机浏览器上运行的html版游戏，让广大军迷评价参数拟真性和时空图实用性
- [x] 修复GPU版时空图的全力脱离线不符合要求的问题
- [x] 设计合理的RL冷启动课程
- [ ] 将主要功能迁移到Nvidia Warp当中实现，并删除原本的低性能实现

长远：
- [ ] 基于扩散模型的时空图概率云模式
- [ ] 大规模空战模式，一大群战斗机在同一个环境当中对战
- [ ] 中心化指挥系统，设计指挥型agent与基层agent，指挥型没有实体，负责对基层进行调度，尝试训练出互相掩护的编队战术

Issue：
- [x] TODO:四阶段冷启动的第一阶段经过测试发现，原本应该直线飞行的对手居然也是智能体，这是一个bug，需要修复。
- [x] TODO:目前Warp训练脚本的可视化功能在初始化时会卡顿十几秒，后续需要优化

## 环境对比

| 模块 | 用途 | 设备 | 并行 |
|------|------|------|------|
| env_numpy | 可视化游戏 | CPU | 单环境 |
| env_warp | RL 训练 | GPU | 多环境 |

## 许可证

MIT License
