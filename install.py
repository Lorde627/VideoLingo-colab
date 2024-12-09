import os
import platform
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

ascii_logo = """
__     ___     _            _     _                    
\ \   / (_) __| | ___  ___ | |   (_)_ __   __ _  ___  
 \ \ / /| |/ _` |/ _ \/ _ \| |   | | '_ \ / _` |/ _ \ 
  \ V / | | (_| |  __/ (_) | |___| | | | | (_| | (_) |
   \_/  |_|\__,_|\___|\___/|_____|_|_| |_|\__, |\___/ 
                                          |___/        
"""

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_package(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

def check_nvidia_gpu():
    install_package("pynvml")
    import pynvml
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            print(f"Detected NVIDIA GPU(s)")
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU {i}: {name}")
            return True
        else:
            print("No NVIDIA GPU detected")
            return False
    except pynvml.NVMLError:
        print("No NVIDIA GPU detected or NVIDIA drivers not properly installed")
        return False
    finally:
        pynvml.nvmlShutdown()

def check_ffmpeg():
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        console.print(Panel("✅ FFmpeg is already installed", style="green"))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        if is_colab():
            # Install ffmpeg directly on Colab
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
            console.print(Panel("✅ FFmpeg has been installed", style="green"))
            return True
        else:
            # Original ffmpeg installation guidance for local environment
            system = platform.system()
            install_cmd = ""
            
            if system == "Windows":
                install_cmd = "choco install ffmpeg"
                extra_note = "Install Chocolatey first (https://chocolatey.org/)"
            elif system == "Darwin":
                install_cmd = "brew install ffmpeg"
                extra_note = "Install Homebrew first (https://brew.sh/)"
            elif system == "Linux":
                install_cmd = "sudo apt install ffmpeg  # Ubuntu/Debian\nsudo yum install ffmpeg  # CentOS/RHEL"
                extra_note = "Use your distribution's package manager"
            
            console.print(Panel.fit(
                f"❌ FFmpeg not found\n\n"
                f"🛠️ Install using:\n[bold cyan]{install_cmd}[/bold cyan]\n\n"
                f"💡 Note: {extra_note}\n\n"
                f"🔄 After installing FFmpeg, please run this installer again: [bold cyan]python install.py[/bold cyan]",
                style="red"
            ))
            raise SystemExit("FFmpeg is required. Please install it and run the installer again.")

def check_package_version(package_name):
    try:
        import pkg_resources
        version = pkg_resources.get_distribution(package_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return None

def check_requirements_satisfied():
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().splitlines()
        
        # 过滤掉注释和空行
        requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]
        
        for req in requirements:
            # 处理包名和版本号
            if '==' in req:
                package_name = req.split('==')[0]
            elif '>=' in req:
                package_name = req.split('>=')[0]
            else:
                package_name = req
            
            if check_package_version(package_name) is None:
                return False
        return True
    except FileNotFoundError:
        return False

def main():
    # Initialize console at the start of main function
    from rich.console import Console
    from rich.panel import Panel
    from rich.box import DOUBLE
    console = Console()
    
    if is_colab():
        console.print(Panel("🎮 Running in Google Colab environment", style="cyan"))
        
        # 检查是否已经满足所有依赖
        if check_requirements_satisfied():
            console.print(Panel("✅ All required packages are already installed", style="green"))
            check_ffmpeg()
            console.print(Panel.fit("Ready to use!", style="bold green"))
            console.print("To start the application, run:")
            console.print("[bold cyan]streamlit run st.py[/bold cyan]")
            return
    
    install_package("requests", "rich", "ruamel.yaml")
    
    width = max(len(line) for line in ascii_logo.splitlines()) + 4
    welcome_panel = Panel(
        ascii_logo,
        width=width,
        box=DOUBLE,
        title="[bold green]🌏[/bold green]",
        border_style="bright_blue"
    )
    console.print(welcome_panel)
    
    console.print(Panel.fit("🚀 Starting Installation", style="bold magenta"))

    if is_colab():
        console.print(Panel("🎮 Running in Google Colab environment", style="cyan"))
    else:
        # Configure mirrors for non-Colab environment
        from core.pypi_autochoose import main as choose_mirror
        choose_mirror()
        
        # Only check GPU and install PyTorch in non-Colab environment
        has_gpu = platform.system() != 'Darwin' and check_nvidia_gpu()
        if has_gpu:
            console.print(Panel("🎮 NVIDIA GPU detected, installing CUDA version of PyTorch...", style="cyan"))
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchaudio==2.0.0", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            system_name = "🍎 MacOS" if platform.system() == 'Darwin' else "💻 No NVIDIA GPU"
            console.print(Panel(f"{system_name} detected, installing CPU version of PyTorch... However, it would be extremely slow for transcription.", style="cyan"))
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.2", "torchaudio==2.1.2"])

    def install_requirements():
        try:
            # 在 Colab 环境下跳过 torch 相关包的安装
            if is_colab():
                with open('requirements.txt', 'r') as f:
                    requirements = f.read().splitlines()
                # 过滤掉 torch 相关的包
                requirements = [r for r in requirements if not r.startswith('torch')]
                # 创建临时requirements文件
                with open('requirements_temp.txt', 'w') as f:
                    f.write('\n'.join(requirements))
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "-r", 
                    "requirements_temp.txt"
                ], env={**os.environ, "PIP_NO_CACHE_DIR": "0", "PYTHONIOENCODING": "utf-8"})
                os.remove('requirements_temp.txt')
            else:
                subprocess.check_call([
                    sys.executable, 
                    "-m", 
                    "pip", 
                    "install", 
                    "-r", 
                    "requirements.txt"
                ], env={**os.environ, "PIP_NO_CACHE_DIR": "0", "PYTHONIOENCODING": "utf-8"})
        except subprocess.CalledProcessError as e:
            console.print(Panel(f"❌ Failed to install requirements: {str(e)}", style="red"))

    install_requirements()
    check_ffmpeg()
    
    console.print(Panel.fit("Installation completed", style="bold green"))
    console.print("To start the application, run:")
    console.print("[bold cyan]streamlit run st.py[/bold cyan]")
    console.print("[yellow]Note: First startup may take up to 1 minute[/yellow]")
    
    # Add troubleshooting tips
    console.print("\n[yellow]If the application fails to start:[/yellow]")
    console.print("1. [yellow]Check your network connection[/yellow]")
    console.print("2. [yellow]Re-run the installer: [bold]python install.py[/bold][/yellow]")

    # start the application
    subprocess.Popen(["streamlit", "run", "st.py"])

if __name__ == "__main__":
    main()
