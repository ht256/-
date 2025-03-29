import os
import sys

def print_directory_structure(startpath):
    """
    打印目录结构
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

def check_python_files(startpath):
    """
    检查Python文件是否存在
    """
    python_files = []
    for root, dirs, files in os.walk(startpath):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"\n找到 {len(python_files)} 个Python文件:")
    for file in python_files:
        print(f" - {file}")
    
    return python_files

def check_required_files():
    """
    检查关键文件是否存在
    """
    required_files = [
        'main.py',
        'README.md',
        'requirements.txt',
        'src/data/data_preprocessing.py',
        'src/models/time_series_model.py',
        'src/models/association_rules.py',
        'src/visualization/data_visualizer.py'
    ]
    
    print("\n检查关键文件:")
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f" {status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_directories():
    """
    检查必要的目录是否存在
    """
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/output',
        'data/visualization',
        'src/data',
        'src/models',
        'src/visualization',
        'notebooks'
    ]
    
    print("\n检查目录结构:")
    all_exist = True
    for directory in required_dirs:
        exists = os.path.isdir(directory)
        status = "✓" if exists else "✗"
        print(f" {status} {directory}")
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    """
    主函数
    """
    print("零售业销售预测分析项目结构检查\n")
    
    # 打印目录结构
    print("项目目录结构:")
    print_directory_structure('.')
    
    # 检查Python文件
    python_files = check_python_files('.')
    
    # 检查关键文件
    files_ok = check_required_files()
    
    # 检查目录结构
    dirs_ok = check_directories()
    
    # 打印总结
    print("\n检查结果摘要:")
    print(f" - 发现Python文件: {len(python_files)} 个")
    print(f" - 关键文件检查: {'通过 ✓' if files_ok else '失败 ✗'}")
    print(f" - 目录结构检查: {'通过 ✓' if dirs_ok else '失败 ✗'}")
    
    if files_ok and dirs_ok:
        print("\n项目结构完整！可以继续开发和测试。")
    else:
        print("\n项目结构不完整。请参考上面的信息修复缺失的文件或目录。")

if __name__ == "__main__":
    main() 