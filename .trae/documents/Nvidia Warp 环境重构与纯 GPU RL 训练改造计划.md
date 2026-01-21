# Phase 1: 基础设施搭建与 Warp 环境原型
## 1.1 环境与分支准备
1.  在系统终端创建并切换到新分支 `warp-ppo-cleanrl`。
2.  确认 `cleanrl` Conda 环境可用，并安装必要的依赖（如果缺少 `warp-lang`，尝试安装）。

## 1.2 构建 Warp 核心环境 (`env_warp/`)
1.  **创建目录结构**：新建 `env_warp/` 目录。
2.  **实现 `warp_utils.py`**：封装 Warp 初始化、设备选择和 Torch 互操作辅助函数。
3.  **实现 `warp_env.py` (核心)**：
    *   **状态定义**：使用 `wp.struct` 重新定义 `states`，包含 `x, y, vx, vy, rudder, throttle` 等字段，保持与 `TensorEnv` 一致的内存布局以便对比。
    *   **核心 Kernels**：移植 `TensorAerodynamic` 和 `TensorMissileGuidance` 的逻辑到 Warp Kernels：
        *   `physics_step_kernel`: 处理飞机的气动和运动积分。
        *   `missile_guidance_kernel`: 处理导弹的比例导引律。
        *   `event_check_kernel`: 处理距离判定、命中检测、自毁逻辑。
        *   `reset_kernel`: 处理环境重置逻辑。
    *   **环境接口**：实现 `reset()`, `step()`, `get_observations()`。
        *   **关键点**：`step` 函数内部利用 **CUDA Graph** 录制 Warp kernel 的 launch 序列，以最大化减少 CPU 开销。
        *   **数据流**：Obs 生成直接在 Warp 内核中写入到与 Torch 共享显存的 Buffer 中，零拷贝输出。

# Phase 2: PPO Agent 升级 (离散化 & CleanRL 对齐)
## 2.1 新建离散 PPO Agent (`agents/learned/ppo_agent_discrete.py`)
1.  **动作空间定义**：改为 `MultiDiscrete([3, 2])` (方向舵: 左/中/右, 开火: 是/否)，简化控制并符合日志规划。
2.  **网络结构升级**：
    *   Actor 输出改为对应离散动作的 Logits。
    *   引入 **正交初始化 (Orthogonal Initialization)**。
3.  **算法逻辑对齐 CleanRL**：
    *   实现 **Learning Rate Annealing** (线性衰减)。
    *   完善 **Global Gradient Clipping**。
    *   确保 GAE 计算和 Buffer 管理全流程在 GPU 上进行。

# Phase 3: 训练循环重构 (`train_warp.py`)
## 3.1 编写纯 GPU 训练脚本
1.  基于 `train.py` 创建 `train_warp.py`。
2.  **移除 CPU 依赖**：
    *   剔除 PyGame 可视化（或设为可选且仅在 eval 模式开启）。
    *   移除所有 `tensor.item()` 和 `.cpu()` 操作，仅保留必要的 Log 打印。
3.  **集成新组件**：
    *   实例化 `WarpEnv` 和 `DiscretePPOAgent`。
    *   对接数据流：`WarpEnv (Torch Tensor) -> Agent (Torch Tensor) -> WarpEnv (Torch Tensor)`。
4.  **性能监控**：添加 SPS (Steps Per Second) 计数器，实时监控训练吞吐量。

# Phase 4: 验证与调优
## 4.1 冒烟测试
1.  运行 `train_warp.py` 进行短时间的 Run，验证无报错。
2.  检查显存占用和 GPU 利用率 (通过 `nvidia-smi` 或 Nsight)。

## 4.2 性能与收敛性验证
1.  在 GTX 1050 上测试 10,000+ 并行环境下的 SPS，目标 > 500。
2.  观察 Reward 曲线，确保离散动作下的智能体能够学会基本的追踪和躲避策略。
