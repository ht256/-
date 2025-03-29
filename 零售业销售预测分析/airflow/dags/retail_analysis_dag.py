from datetime import datetime, timedelta
import os
import sys

# 将项目根目录添加到Python路径
sys.path.append('/opt/airflow/retail_sales_analysis')

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import Variable
from airflow.utils.email import send_email

# 默认参数
default_args = {
    'owner': 'retail_analytics',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['analytics@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 创建DAG
dag = DAG(
    'retail_sales_analysis',
    default_args=default_args,
    description='零售业销售预测分析工作流',
    schedule_interval='0 1 * * *',  # 每天凌晨1点执行
    catchup=False,
    tags=['retail', 'sales', 'forecast', 'recommendation'],
)

# 导入项目模块的函数，确保可以在Airflow worker中找到这些模块
def import_project_modules():
    # 这里设置Python路径，确保可以导入项目模块
    from src.data.data_preprocessing import DataPreprocessor
    from src.models.time_series_model import TimeSeriesForecaster
    from src.models.association_rules import AssociationAnalyzer
    from src.models.fp_growth_analyzer import FPGrowthAnalyzer
    from src.visualization.data_visualizer import DataVisualizer
    from src.cross_department.collaboration import CollaborationManager
    
    return "项目模块导入成功"

# 检查新数据
check_new_data = FileSensor(
    task_id='check_new_data',
    filepath='/opt/airflow/data/raw/new_sales_data.csv',
    poke_interval=60,  # 每60秒检查一次
    timeout=60 * 60 * 2,  # 最多等待2小时
    mode='reschedule',  # 如果不满足条件，释放worker
    dag=dag,
)

# 数据预处理任务
def preprocess_data(**kwargs):
    from src.data.data_preprocessing import DataPreprocessor
    
    # 获取输入和输出路径
    input_file = '/opt/airflow/data/raw/new_sales_data.csv'
    processed_dir = '/opt/airflow/data/processed/'
    
    # 创建预处理器
    preprocessor = DataPreprocessor(input_file)
    
    # 加载数据
    data = preprocessor.load_data()
    
    # 清洗数据
    clean_data = preprocessor.clean_data(data)
    
    # 处理缺失值
    clean_data = preprocessor.handle_missing_values(clean_data)
    
    # 检测并处理异常值
    clean_data = preprocessor.handle_outliers(clean_data)
    
    # 特征工程
    clean_data = preprocessor.feature_engineering(clean_data)
    
    # 准备时间序列数据
    ts_data = preprocessor.prepare_time_series_data(clean_data)
    
    # 准备关联规则分析数据
    transactions = preprocessor.prepare_association_rules_data(clean_data)
    
    # 保存处理后的数据
    preprocessor.save_processed_data(clean_data, ts_data, transactions, processed_dir)
    
    return {
        'processed_data_path': os.path.join(processed_dir, 'processed_data.csv'),
        'time_series_data_path': os.path.join(processed_dir, 'time_series_Sales.csv'),
        'association_data_path': os.path.join(processed_dir, 'association_transactions.csv')
    }

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

# 时间序列预测任务
def run_time_series_forecast(**kwargs):
    from src.models.time_series_model import TimeSeriesForecaster
    
    # 获取前一个任务的返回值
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='preprocess_data')
    
    time_series_data_path = paths['time_series_data_path']
    output_dir = '/opt/airflow/data/output/'
    
    # 创建时间序列预测器
    forecaster = TimeSeriesForecaster(
        data_path=time_series_data_path,
        date_col="TransactionDate",
        target_col="Sales",
        output_dir=output_dir
    )
    
    # 加载数据
    data = forecaster.load_data()
    
    # 检查平稳性
    is_stationary, p_value = forecaster.check_stationarity()
    
    # 绘制时间序列图表
    forecaster.plot_time_series()
    
    # 寻找最优参数
    best_params = forecaster.find_optimal_params()
    
    # 划分训练集和测试集
    train_data, test_data = forecaster.train_test_split(test_size=0.2)
    
    # 训练模型
    model_results = forecaster.train_model()
    
    # 评估模型
    metrics = forecaster.evaluate_model()
    
    # 预测未来30天销售
    forecast = forecaster.forecast_future(steps=30)
    
    # 生成报告
    forecaster.generate_report()
    
    return {
        'forecast_path': os.path.join(output_dir, 'future_forecast.csv'),
        'accuracy': metrics.get('accuracy', 0)
    }

time_series_task = PythonOperator(
    task_id='time_series_forecast',
    python_callable=run_time_series_forecast,
    provide_context=True,
    dag=dag,
)

# Apriori关联规则分析任务
def run_apriori_analysis(**kwargs):
    from src.models.association_rules import AssociationAnalyzer
    
    # 获取前一个任务的返回值
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='preprocess_data')
    
    association_data_path = paths['association_data_path']
    output_dir = '/opt/airflow/data/output/'
    
    # 创建关联规则分析器
    analyzer = AssociationAnalyzer(
        data_path=association_data_path,
        output_dir=output_dir
    )
    
    # 加载数据
    transactions = analyzer.load_data()
    
    # 分析交易数据
    analyzer.analyze_transactions()
    
    # 寻找频繁项集
    frequent_itemsets = analyzer.find_frequent_itemsets(min_support=0.01)
    
    # 生成关联规则
    rules = analyzer.generate_association_rules(min_threshold=0.5)
    
    # 可视化规则
    analyzer.visualize_rules()
    
    # 导出促销建议
    recommendations = analyzer.export_promotion_recommendations(min_lift=1.5)
    
    return {
        'rules_path': os.path.join(output_dir, 'association_rules.csv'),
        'recommendations_path': os.path.join(output_dir, 'promotion_recommendations.csv')
    }

apriori_task = PythonOperator(
    task_id='apriori_analysis',
    python_callable=run_apriori_analysis,
    provide_context=True,
    dag=dag,
)

# FP-Growth关联规则分析任务
def run_fp_growth_analysis(**kwargs):
    from src.models.fp_growth_analyzer import FPGrowthAnalyzer
    
    # 获取前一个任务的返回值
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='preprocess_data')
    
    association_data_path = paths['association_data_path']
    output_dir = '/opt/airflow/data/output/algorithm_comparison/'
    
    # 创建FP-Growth分析器
    analyzer = FPGrowthAnalyzer(
        data_path=association_data_path,
        output_dir=output_dir
    )
    
    # 加载数据
    transactions = analyzer.load_data()
    
    # 运行性能测试，比较不同支持度
    results_df = analyzer.run_performance_test(
        min_support_values=[0.01, 0.005, 0.003],
        min_confidence=0.3
    )
    
    # 与Apriori结果比较
    apriori_results_path = os.path.join(output_dir, 'apriori_performance_test.csv')
    
    if os.path.exists(apriori_results_path):
        comparison_df = analyzer.compare_with_apriori(apriori_results_path)
        report_path = analyzer.generate_performance_report(comparison_df)
    
    return {
        'fp_growth_results_path': os.path.join(output_dir, 'fp_growth_performance_test.csv'),
        'comparison_report': os.path.join(output_dir, 'algorithm_comparison_report.md')
    }

fp_growth_task = PythonOperator(
    task_id='fp_growth_analysis',
    python_callable=run_fp_growth_analysis,
    provide_context=True,
    dag=dag,
)

# 算法比较任务
def run_algorithm_comparison(**kwargs):
    from src.models.apriori_performance import run_apriori_benchmark
    
    # 获取前一个任务的返回值
    ti = kwargs['ti']
    paths = ti.xcom_pull(task_ids='preprocess_data')
    
    association_data_path = paths['association_data_path']
    output_dir = '/opt/airflow/data/output/algorithm_comparison/'
    
    # 运行Apriori基准测试
    apriori_results = run_apriori_benchmark(
        data_path=association_data_path,
        output_dir=output_dir,
        min_support_values=[0.01, 0.005, 0.003],
        min_confidence=0.3
    )
    
    return {
        'apriori_results_path': os.path.join(output_dir, 'apriori_performance_test.csv')
    }

algorithm_comparison_task = PythonOperator(
    task_id='algorithm_comparison',
    python_callable=run_algorithm_comparison,
    provide_context=True,
    dag=dag,
)

# 数据可视化任务
def run_visualization(**kwargs):
    from src.visualization.data_visualizer import DataVisualizer
    
    # 获取前面任务的返回值
    ti = kwargs['ti']
    preprocess_paths = ti.xcom_pull(task_ids='preprocess_data')
    apriori_paths = ti.xcom_pull(task_ids='apriori_analysis')
    
    processed_data_path = preprocess_paths['processed_data_path']
    time_series_data_path = preprocess_paths['time_series_data_path']
    recommendations_path = apriori_paths['recommendations_path']
    
    visualization_dir = '/opt/airflow/data/visualization/'
    
    # 创建可视化器
    visualizer = DataVisualizer(output_dir=visualization_dir)
    
    # 加载数据
    visualizer.load_processed_data(processed_data_path)
    visualizer.load_time_series_data(time_series_data_path)
    visualizer.load_association_recommendations(recommendations_path)
    
    # 为Tableau准备数据
    tableau_files = visualizer.prepare_for_tableau()
    
    # 创建示例可视化
    visualizer.create_sample_visualizations()
    
    return {
        'tableau_files': tableau_files
    }

visualization_task = PythonOperator(
    task_id='data_visualization',
    python_callable=run_visualization,
    provide_context=True,
    dag=dag,
)

# 跨部门协作任务
def run_collaboration_workflow(**kwargs):
    from src.cross_department.collaboration import CollaborationManager
    import pandas as pd
    
    # 获取前面任务的返回值
    ti = kwargs['ti']
    preprocess_paths = ti.xcom_pull(task_ids='preprocess_data')
    forecast_result = ti.xcom_pull(task_ids='time_series_forecast')
    apriori_paths = ti.xcom_pull(task_ids='apriori_analysis')
    
    processed_data_path = preprocess_paths['processed_data_path']
    forecast_path = forecast_result['forecast_path']
    rules_path = apriori_paths['rules_path']
    
    # 加载处理后的数据
    processed_data = pd.read_csv(processed_data_path, encoding='utf-8-sig')
    forecast_data = pd.read_csv(forecast_path, encoding='utf-8-sig')
    rules_data = pd.read_csv(rules_path, encoding='utf-8-sig')
    
    # 创建模拟历史数据和库存数据（在实际应用中，这些应该来自ERP系统）
    from src.cross_department.collaboration import create_sample_data
    _, historical_data, inventory_data, _ = create_sample_data()
    
    # 创建协作管理器
    collaboration_dir = '/opt/airflow/data/collaboration/'
    manager = CollaborationManager(output_dir=collaboration_dir)
    
    # 运行协作工作流
    results = manager.run_collaboration_workflow(
        forecast_data=forecast_data,
        historical_data=historical_data,
        inventory_data=inventory_data,
        association_rules=rules_data
    )
    
    return {
        'collaboration_results': os.path.join(collaboration_dir, 'collaboration_workflow_results.json'),
        'procurement_recommendations': os.path.join(collaboration_dir, 'procurement_recommendations.csv'),
        'operations_recommendations': os.path.join(collaboration_dir, 'operations_recommendations.csv'),
        'marketing_recommendations': os.path.join(collaboration_dir, 'marketing_recommendations.csv')
    }

collaboration_task = PythonOperator(
    task_id='cross_department_collaboration',
    python_callable=run_collaboration_workflow,
    provide_context=True,
    dag=dag,
)

# API服务启动任务
start_api_service = BashOperator(
    task_id='start_api_service',
    bash_command='cd /opt/airflow/retail_sales_analysis && python src/api/api_server.py &',
    dag=dag,
)

# 发送报告任务
def send_analysis_report(**kwargs):
    # 获取前面任务的返回值
    ti = kwargs['ti']
    forecast_result = ti.xcom_pull(task_ids='time_series_forecast')
    collaboration_results = ti.xcom_pull(task_ids='cross_department_collaboration')
    
    # 构建报告内容
    accuracy = forecast_result.get('accuracy', 0)
    
    subject = f"零售业销售分析报告 - {datetime.now().strftime('%Y-%m-%d')}"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .container {{ width: 80%; margin: 0 auto; }}
            .header {{ background-color: #003366; color: white; padding: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .highlight {{ background-color: #ffffcc; padding: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>零售业销售分析报告</h1>
                <p>生成日期: {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="section">
                <h2>销售预测摘要</h2>
                <p>预测准确率: <span class="highlight">{accuracy:.2f}%</span></p>
                <p>详细预测数据请查看附件。</p>
            </div>
            
            <div class="section">
                <h2>跨部门协作建议</h2>
                <p>本次分析生成了以下跨部门协作建议：</p>
                <ul>
                    <li>采购部门：请查看 procurement_recommendations.csv</li>
                    <li>运营部门：请查看 operations_recommendations.csv</li>
                    <li>市场营销部门：请查看 marketing_recommendations.csv</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>API访问</h2>
                <p>您可以通过以下API获取实时分析：</p>
                <ul>
                    <li>销售预测: <code>http://analytics-server:5000/api/v1/forecast</code></li>
                    <li>商品推荐: <code>http://analytics-server:5000/api/v1/recommendations?product_ids=P001,P002</code></li>
                    <li>销售分析: <code>http://analytics-server:5000/api/v1/analytics?type=seasonal</code></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 准备附件
    attachments = [
        forecast_result.get('forecast_path', ''),
        collaboration_results.get('procurement_recommendations', ''),
        collaboration_results.get('marketing_recommendations', '')
    ]
    
    # 发送邮件
    try:
        recipients = ['analytics@example.com', 'management@example.com']
        send_email(
            to=recipients,
            subject=subject,
            html_content=html_content,
            files=attachments
        )
        return "报告发送成功"
    except Exception as e:
        return f"报告发送失败: {str(e)}"

send_report_task = PythonOperator(
    task_id='send_analysis_report',
    python_callable=send_analysis_report,
    provide_context=True,
    dag=dag,
)

# 定义任务依赖关系
check_new_data >> preprocess_task
preprocess_task >> time_series_task
preprocess_task >> apriori_task
preprocess_task >> algorithm_comparison_task
algorithm_comparison_task >> fp_growth_task

time_series_task >> visualization_task
apriori_task >> visualization_task

visualization_task >> collaboration_task
time_series_task >> collaboration_task
apriori_task >> collaboration_task

collaboration_task >> start_api_service
collaboration_task >> send_report_task 