import time
import json 

def timed_execution(func, *args, **kwargs):
    """Ejecuta una función y registra el tiempo de ejecución."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def log_times(steps_times):
    """
    Devuelve un resumen de los tiempos para todas las etapas en formato JSON.

    Parámetros:
        steps_times (dict): Diccionario con los nombres de las etapas como clave y el tiempo como valor.

    Retorna:
        str: Resumen de los tiempos en formato JSON.
    """
    tiempos_resumen = {}

    # Agregar los tiempos al diccionario
    for step, time_taken in steps_times.items():
        tiempos_resumen[step] = round(time_taken, 4)  # Redondeamos a 4 decimales

    # Devolvemos el diccionario que se puede almacenar en Firebase
    return tiempos_resumen
