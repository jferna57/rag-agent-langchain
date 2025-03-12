import sys
import subprocess
import platform
from typing import Dict
import psutil

from src.data import SystemInfo


def obtener_info_equipo() -> SystemInfo:
    """
    Obtiene información detallada del equipo y la devuelve en un objeto SystemInfo.

    Returns:
        SystemInfo: Un objeto que contiene la información del sistema.

    """
    # Detectar el sistema operativo y arquitectura
    sistema = platform.system()
    arquitectura = platform.machine()

    disk_space_gb: Dict[str, float] = {
        "Total": round(psutil.disk_usage('/').total / (1024**3), 2),
        "Usado": round(psutil.disk_usage('/').used / (1024**3), 2),
        "Libre": round(psutil.disk_usage('/').free / (1024**3), 2)
    }

    info_equipo = SystemInfo(
        operating_system=f"{sistema} {platform.release()}",
        version=platform.version(),
        architecture=arquitectura,
        processor=platform.processor(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        disk_space_gb=disk_space_gb,
        python_version=sys.version,
        gpu=obtener_gpu(),
        gpu_count=obtener_cantidad_gpus(),
        #torch_available=torch_disponible(),
        #cuda_available=torch_cuda_disponible()
    )

    return info_equipo

def obtener_gpu():
    """Detects the presence of the GPU, especially for macOS M1."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Verificar GPU integrada en macOS M1 y versiones posteriores
        try:
            resultado = subprocess.check_output(["system_profiler", "SPDisplaysDataType"])
            resultado = resultado.decode("utf-8")
            if "Apple M1" in resultado or "Apple GPU" in resultado:
                return "Apple GPU (integrada)"
            else:
                return "GPU no disponible"
        except Exception as e:
            return f"Error al obtener información de GPU: {str(e)}"
    else:
        return "GPU no disponible"

def obtener_cantidad_gpus():
    """Devuelve la cantidad de GPUs disponibles en el sistema."""
    if platform.system() == "Darwin":
        # En sistemas Apple, solo tendremos acceso a la GPU integrada
        return 1
    else:
        return 0

def torch_disponible():
    """Verifica si PyTorch está disponible en el sistema."""
    try:
        import torch
        return True
    except ImportError:
        return False

def torch_cuda_disponible():
    """Verifica si CUDA está disponible (para sistemas con PyTorch)."""
    if torch_disponible():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
    return False

if __name__ == "__main__":
    system_info = obtener_info_equipo()
    print(system_info)
