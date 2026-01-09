# PPO智能体使用说明

## 快速开始

### 基础训练命令

```bash
# 最小测试（快速验证）
python train.py --agent ppo --opponent rule_based --reward zero --num-envs 4 --n-steps 128 --max-steps 1000

# 标准训练（推荐配置）
python train.py --agent ppo --opponent rule_based --reward zero --num-envs 8 --n-steps 2048 --max-steps 50000 --time-scale 2.0

# 快速训练（调试用）
python train.py --agent ppo --opponent rule_based --num-envs 16 --n-steps 256 --max-steps 10000 --time-scale 4.0
```

## 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--agent` | - | 训练的智能体（ppo） |
| `--opponent` | - | 对手智能体（rule_based=CrankAgent） |
| `--num-envs` | 32 | 并行环境数量 |
| `--device` | cuda | 计算设备（cuda/cpu） |

### PPO超参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--learning-rate` | 3e-4 | 学习率 |
| `--n-steps` | 2048 | 每次更新收集的步数 |
| `--batch-size` | 64 | mini-batch大小 |
| `--n-epochs` | 10 | 每次更新的epoch数 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.95 | GAE参数 |
| `--clip-epsilon` | 0.2 | PPO裁剪系数 |
| `--value-loss-coef` | 0.5 | 价值损失权重 |
| `--entropy-coef` | 0.01 | 熵正则化权重 |

### 其他参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--max-steps` | 10000 | 最大训练步数 |
| `--time-scale` | 1.0 | 时间加速倍率 |
| `--save-interval` | 10000 | 模型保存间隔 |
| `--log-interval` | 100 | 日志输出间隔 |

## 模型保存与加载

### 自动保存

训练过程中，模型会自动保存到 `checkpoints/` 目录：

```
checkpoints/
├── ppo_agent_step_10000.pt
├── ppo_agent_step_20000.pt
└── ...
```

### 手动加载模型

```python
from agents.learned import PPOAgent

# 创建agent
agent = PPOAgent(device='cuda')
agent.num_envs = 8

# 加载模型
agent.load('checkpoints/ppo_agent_step_10000.pt')

# 评估模式
agent.eval()
```

## 训练监控

### 日志输出示例

```
[Step   1400] Episodes: 7, P1胜率: 14.3% | Loss: P=-0.003 V=0.038 E=2.233
```

说明：
- **Episodes**: 已完成的对局数
- **P1胜率**: PPO智能体（红方）的胜率
- **P**: 策略损失（越接近0越好）
- **V**: 价值损失（逐渐下降）
- **E**: 策略熵（保持在2左右表示良好探索）

### 性能指标

正常训练应观察到：
- ✅ 策略损失在 -0.01 到 0.01 之间
- ✅ 价值损失逐渐下降
- ✅ 熵值保持在 1.5 到 2.5 之间
- ✅ 胜率逐步提升（对抗CrankAgent初期胜率<10%是正常的）

## 常见问题

### Q: 训练过程中出现NaN？

A: 尝试：
- 降低学习率：`--learning-rate 1e-4`
- 减小裁剪系数：`--clip-epsilon 0.1`
- 检查CUDA内存

### Q: 胜率不提升？

A: 可能原因：
- 训练步数不够（CrankAgent较强，需要更多训练）
- 增大熵系数提高探索：`--entropy-coef 0.02`
- 调整奖励函数（当前使用zero奖励）

### Q: CUDA内存不足？

A: 减少：
- 并行环境数：`--num-envs 4`
- n-steps：`--n-steps 1024`
- batch-size：`--batch-size 32`

## 架构说明

### 文件结构

```
agents/learned/
├── __init__.py          # 模块导出
├── networks.py          # Actor和Critic网络
└── ppo_agent.py         # PPO智能体实现
```

### 网络架构

**Actor网络**（策略网络）：
- 输入：10维观察特征
- 隐藏层：128 → 64
- 输出：
  - 方向舵：高斯分布（均值+对数标准差）
  - 开火：伯努利分布（概率）
  - 油门：固定为1.0（满油门）

**Critic网络**（价值网络）：
- 输入：10维观察特征
- 隐藏层：128 → 64
- 输出：状态价值V(s)

### 观察空间

10维特征向量：
1. x - 己方归一化x坐标 [0,1]
2. y - 己方归一化y坐标 [0,1]
3. angle - 己方归一化角度 [0,1]
4. speed - 己方归一化速度 [0,2]
5. missiles - 己方归一化导弹数 [0,1]
6. alive - 己方存活状态 {0,1}
7. enemy_distance - 敌方归一化距离 [0,1]
8. enemy_relative_angle - 敌方相对角度 [-1,1]
9. enemy_speed - 敌方归一化速度 [0,2]
10. enemy_alive - 敌方存活状态 {0,1}

## 开发计划

当前实现为基础版本，未来可优化：
- [ ] 添加RNN层处理时序信息
- [ ] 实现油门连续控制
- [ ] 增加导弹威胁感知
- [ ] 设计课程学习策略
- [ ] 添加TensorBoard支持

## 参考

- 设计文档：`.qoder/quests/ppo-basic-agent-implementation.md`
- 基类接口：`agents/base_agent.py`
- 环境接口：`env_gym/tensor_env.py`
