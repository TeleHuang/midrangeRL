# midrangeRL

二维中距空战 RL 训练平台 —— 兼具科普与强化学习研究价值

## 核心特性

- **时空图(Space Time Graph/STG)**：直观展示导弹真实射程，辅助战斗决断
- **GPU 多环境并行**：单次支持数千环境同步训练
- **Gymnasium 接口**：兼容主流 RL 算法
- **手动对抗**：红蓝双方键盘操控，体验人机对战

## 快速开始

```bash
pip install torch pygame numpy gymnasium
python game_play.py   # 试玩游戏
```

**操作：** 红方 `A/D` 转向 `T` 开火；蓝方 方向键 转向 `=` 开火；`V` 切换视图

## 项目结构

```
midrangeRL/
├── env_gym/       # GPU 并行 RL 环境（tensor_env, gym_wrapper）
├── env_numpy/     # CPU 单例环境（气动、制导模块）
├── agents/        # Agent 接口与实现
│   ├── base_agent.py   # 抽象基类：act(obs)→action, reset()
│   ├── rule_based/     # 规则 agent（冷启动）
│   └── learned/        # 训练模型参数
├── rewards/       # Reward 接口与预设
│   ├── base_reward.py  # 抽象基类：compute()→scalar
│   └── presets/        # 不同阶段的 reward 组合
├── spacetime/       # 时空图渲染模块
│   ├── tensor_spacetime.py  # RL训练使用的GPU时空图
│   ├── spacetime_renderer.py  # 渲染时空图
│   └── spacetime_core.py  # CPU模式的时空图计算
├── train.py       # 训练入口（命令行参数选择 agent/reward/并行数等）
├── game_play.py   # 游戏主程序
└── config.py      # 配置文件6*5
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

近期：
- [ ] 开发能在手机浏览器上运行的html版游戏，让广大军迷评价参数拟真性和时空图实用性
- [ ] 修复GPU版时空图的全力脱离线不符合要求的问题
- [ ] 整合GPU版时空图与奖励函数
- [ ] 设计知道躲避导弹的规则式agent：更改crank
- [ ] 实现基于时空图的agent：STG_ppo1、STG_rule_based
- [ ] 将带SGT与非SGT的agent训练至可用水平进行对比

长远：
- [ ] 基于扩散模型的时空图概率云模式
- [ ] 指挥型agent/互相掩护的编队战术

## 环境对比

| 模块 | 用途 | 设备 | 并行 |
|------|------|------|------|
| env_numpy | 可视化游戏 | CPU | 单环境 |
| env_gym | RL 训练 | GPU | 多环境 |

## 许可证

MIT License
