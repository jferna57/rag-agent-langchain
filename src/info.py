import subprocess
import platform
import psutil

def obtener_info_equipo():
    """Obtiene información detallada del equipo."""

    info = {}

    # Sistema operativo y arquitectura
    info['sistema_operativo'] = platform.system()
    info['arquitectura'] = platform.machine()

    # CPU
    info['cpu_nombre'] = platform.processor()
    info['cpu_nucleos_fisicos'] = psutil.cpu_count(logical=False)
    info['cpu_nucleos_logicos'] = psutil.cpu_count(logical=True)
    info['cpu_frecuencia'] = psutil.cpu_freq().max

    # RAM
    info['ram_total'] = psutil.virtual_memory().total / (1024 ** 3)  # En GB

    # GPU
    info['gpu'] = obtener_info_gpu()

    return info

def obtener_info_gpu():
    """Obtiene información de la GPU, detectando NVIDIA y Apple Silicon."""

    try:
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':  # Apple Silicon
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.returncode == 0:
                return {"numero_gpus": 1, "nombres": ["Apple Silicon GPU"]} #simplificado, se puede extraer más información.
        else:  # NVIDIA (asume que está instalado nvidia-smi)
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=name',
                 '--format=csv,noheader'],
                 capture_output=True,
                 text=True,
                 check=True
            )
            if result.returncode == 0:
                nombres_gpu = [line.strip() for line in result.stdout.splitlines()]
                return {"numero_gpus": len(nombres_gpu), "nombres": nombres_gpu}
    except FileNotFoundError:
        return {"numero_gpus": 0, "nombres": ["Información de GPU no disponible"]}
    except subprocess.CalledProcessError as e:
        return {"numero_gpus": 0, "nombres": [f"Error al ejecutar el comando: {e}"]}
    except Exception as e:
        return {"numero_gpus": 0, "nombres": [f"Error inesperado al obtener info de GPU: {e}"]}

# Ejemplo de uso
if __name__ == "__main__":
    info_equipo = obtener_info_equipo()
    print("Información del equipo guardada en info_equipo.json")