根据您的反馈，我将更新计划，清理 `spacetime_core.py` 中废弃的历史轨迹追踪代码，并修复导弹 ID 追踪问题，最后将可视化集成到训练循环中。

### 1. 清理时空图核心代码并修复 ID 追踪

由于历史轨迹追踪功能已废弃，我将移除相关代码，并确保预测功能使用稳定的 ID（`slot_idx`）。

* **文件**: `spacetime/spacetime_core.py`

* **清理工作**:

  * 从 `SpacetimeTrail` 类中移除 `self.history`、`record_position` 和 `get_history_points`。

  * 从 `SpacetimeComputer.update` 中移除对 `record_position` 的调用。

  * 更新相关的文档注释，说明历史轨迹功能已被移除。

* **ID 修复**:

  * 修改 `SpacetimeComputer`（在 `update` 和 `_update_predictions` 等方法中），优先使用 `getattr(missile, 'slot_idx', id(missile))` 来标识导弹。这是为了确保在 Tensor 环境下（每帧对象重建）预测缓存能正确关联到同一个导弹。

### 2. 将可视化集成到训练循环

修改训练脚本以支持实时可视化。

* **文件**: `train.py`

* **修改**:

  * **导入**: 引入 `pygame` 和 `Visualizer`。

  * **参数**: 添加 `--render` 命令行参数。

  * **初始化**: 若启用渲染，初始化 Pygame 和 Visualizer。

  * **主循环**:

    * 处理 Pygame 事件（退出、视图切换 `V`、相机控制）。

    * 获取第 0 号环境的渲染状态：`env.get_render_state(0)`。

    * 更新可视化器：`visualizer.update_spacetime(...)`。

    * 绘制界面并刷新显示。

  * **清理**: 训练结束时退出 Pygame。

### 3. 验证

* **代码清理验证**: 确认 `spacetime_core.py` 中不再包含历史轨迹相关的冗余代码。

* **功能验证**: 运行 `python train.py --render --num-envs 32`，检查可视化窗口是否正常显示，且时空图模式下的预测线（Prediction）是否稳定显示（验证 ID 修复是否成功）。

