# freedom-etl — образ сервиса сбора котировок.
FROM python:3.11-slim

# PYTHONUNBUFFERED — иначе логи копятся в буфере и в `docker logs` появляются
# рывками; для суточного крона это означает «контейнер молчит часами».
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Зависимости отдельным слоем от кода: правка etl_processor.py не должна
# заставлять пересобирать установку пакетов.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY db/ ./db/
COPY services/ ./services/
COPY main.py ./

# Непривилегированный пользователь: сервис ходит только наружу по HTTPS и в
# PostgreSQL, root ему не нужен ни для чего.
RUN useradd --create-home --uid 10001 etl && chown -R etl:etl /app
USER etl

# Проверка живости для docker-compose: процесс планировщика может «уснуть»
# логически (job снят), и проверка по PID этого не увидит. Здесь достаточно
# импорта конфигурации — она валидирует переменные окружения.
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from config.config import Settings; Settings.from_env()" || exit 1

ENTRYPOINT ["python", "main.py"]
