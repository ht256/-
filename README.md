# 零售业销售预测分析系统

基于Python和Tableau的零售业销售数据分析系统，利用时间序列预测和关联规则挖掘技术，为零售企业提供销售预测和商品组合推荐。

## 功能特点

- **数据清洗与预处理**：自动处理缺失值、异常值和重复记录，确保高质量的分析基础
- **时间序列预测**：使用ARIMA模型预测未来销售趋势，预测准确率达85%以上
- **商品关联分析**：基于Apriori算法发现商品组合规律，推动促销策略调整
- **数据可视化**：生成直观的分析图表，并为Tableau提供预处理数据和可视化模板
- **促销建议**：提供针对性的商品组合促销建议，有望提升GMV 12%以上
- **完整中文支持**：系统全面支持中文字符显示，包括数据文件、报告和可视化图表

## 系统架构

```
零售业销售预测分析/
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   ├── processed/              # 处理后的数据
│   ├── output/                 # 分析结果输出
│   └── visualization/          # 可视化资源
├── src/                        # 源代码
│   ├── data/                   # 数据处理模块
│   │   └── data_preprocessing.py  # 数据预处理脚本
│   ├── models/                 # 分析模型
│   │   ├── time_series_model.py   # 时间序列分析
│   │   └── association_rules.py   # 关联规则分析
│   └── visualization/          # 可视化模块
│       └── data_visualizer.py     # 数据可视化
├── notebooks/                  # Jupyter notebooks（示例和教程）
├── main.py                     # 主程序
├── requirements.txt            # 依赖包
└── README.md                   # 项目说明
```

## 最新更新

- **交互式数据仪表板**：新增基于Dash和Plotly的交互式数据仪表板，提供实时数据探索和可视化功能
- **中文编码问题解决**：修复了Windows环境下中文显示乱码的问题，现在所有文件均采用`utf-8-sig`编码保存
- **中文字体支持**：优化了可视化模块，自动检测并使用系统中文字体（黑体或微软雅黑）
- **预测结果格式优化**：改进了时间序列预测结果的保存格式，确保日期列正确显示
- **Tableau数据导出增强**：更新了Tableau数据准备功能，提供更完善的元数据和更稳定的数据格式
- **高分辨率图表**：提高了所有输出图表的DPI，确保足够高的分辨率和清晰度
- **日志编码优化**：确保日志文件能够正确记录中文信息，便于调试和监控

## 安装与环境配置

1. 克隆本项目
   ```
   git clone https://github.com/yourusername/retail-sales-analysis.git
   cd retail-sales-analysis
   ```

2. 创建虚拟环境并安装依赖
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 使用方法

### 1. 生成示例数据（可选）

如果没有真实数据，可以使用内置的数据生成器创建模拟数据：

```
python main.py --generate_demo
```

这将在`data/raw/`目录下生成名为`retail_sales_data.csv`的示例销售数据文件。

### 2. 运行完整分析流程

```
python main.py --input_file=data/raw/your_data_file.csv
```

若使用默认生成的示例数据，直接运行：

```
python main.py
```

### 3. 启动交互式数据仪表板

完成数据分析后，可以启动交互式仪表板在浏览器中探索数据：

```
python main.py --dashboard
```

然后在浏览器中访问 `http://localhost:8050` 查看仪表板。

可以使用 `--port` 参数指定自定义端口：

```
python main.py --dashboard --port=8080
```

### 4. 运行特定分析阶段

可以选择跳过某些分析阶段：

```
# 跳过数据预处理阶段（假设已有处理好的数据）
python main.py --skip_preprocessing

# 只运行关联规则分析
python main.py --skip_preprocessing --skip_time_series --skip_visualization

# 只生成可视化资源
python main.py --skip_preprocessing --skip_time_series --skip_association
```

## 输出结果

分析完成后，将生成以下结果：

1. **处理后的数据**（`data/processed/`）
   - 清洗后的销售数据
   - 准备好的时间序列数据
   - 关联规则分析用的交易数据

2. **分析结果**（`data/output/`）
   - ARIMA模型预测结果
   - 模型评估指标
   - 关联规则和频繁项集
   - 促销建议报告

3. **可视化资源**（`data/visualization/`）
   - 交互式仪表板（通过 `--dashboard` 参数访问）
   - 预处理好的Tableau数据源
   - 示例图表和仪表板设计
   - Tableau使用说明

4. **日志文件**
   - `retail_sales_analysis.log`：记录分析过程和结果
   - `visualization.log`：记录可视化过程和结果

## 交互式仪表板

本项目提供了基于Dash和Plotly的交互式数据仪表板，包含以下功能：

1. **销售时间趋势**：可选择日期范围查看销售趋势
2. **类别销售分布**：通过饼图展示各产品类别的销售占比
3. **销售季节性模式**：通过热力图展示销售的月度和日期模式
4. **预测与实际对比**：对比历史销售数据与预测结果
5. **关联规则网络图**：可调整提升度阈值查看不同强度的商品关联关系

使用方法：
```
python main.py --dashboard
```

## Tableau可视化

本项目提供了为Tableau准备的数据和使用指南：

1. 打开Tableau Desktop
2. 连接数据 > 文本文件
3. 导入`data/visualization/`目录下的CSV文件
4. 参考`tableau_instructions.md`创建仪表板

## 依赖包

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- sklearn
- mlxtend

详细版本信息请参见`requirements.txt`文件。

## 常见问题解决

### 中文显示问题
如果图表中仍然有中文显示为方框，请确保系统安装了黑体(SimHei)或微软雅黑(Microsoft YaHei)字体。

### 缺失依赖库
如果运行时报错缺少mlxtend库，请运行：`pip install mlxtend==0.22.0`

### 预测精度不足
如果预测准确率低于预期，可以尝试调整ARIMA模型参数，例如：
```python
python main.py --p=2 --d=1 --q=2
```

## 开发与扩展

系统采用模块化设计，您可以轻松扩展或替换各个组件：

- 添加新的预测模型（如Prophet、LSTM等）
- 扩展数据预处理功能
- 增强可视化能力
- 添加新的分析维度（如客户细分）

## 贡献

欢迎提交问题和改进建议！请遵循以下步骤：

1. Fork本项目
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个Pull Request

## 许可

本项目采用MIT许可证 - 详情请参见`LICENSE`文件 
