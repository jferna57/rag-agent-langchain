def log_times(steps_times):
    """
    Muestra el resumen de tiempos para todas las etapas.

    Parámetros:
        steps_times (dict): Diccionario con los nombres de las etapas como clave y el tiempo como valor.
    """
    print("\nResumen de tiempos:")
    for step, time_taken in steps_times.items():
        print(f"---> {step}: {time_taken:.4f} segundos")