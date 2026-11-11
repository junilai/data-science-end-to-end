# Imagen del servicio de emergencias. Se construye desde psets/proyecto/,
# porque necesita modelado/ ademas de servicio/. Antes, exporta el modelo con
# python servicio/exportar_modelo.py para que exista servicio/modelo.joblib.
#
#     docker build -f servicio/Dockerfile -t servicio-emergencias .
#     docker run -p 8000:8000 servicio-emergencias
FROM python:3.11-slim
WORKDIR /app
COPY servicio/requirements.txt servicio/requirements.txt
RUN pip install --no-cache-dir -r servicio/requirements.txt
COPY modelado/pipeline_proyecto.py modelado/
COPY servicio/api.py servicio/modelo.joblib servicio/
EXPOSE 8000

CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
