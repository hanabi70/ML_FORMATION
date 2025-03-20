# Utiliser l'image Python 3.12 slim
FROM python:3.12-slim

# Installer uv via pip
RUN pip install uv

# Définir le répertoire de travail
WORKDIR /app

# Copier uniquement les fichiers nécessaires
COPY app.py .
COPY train.py .
COPY templates/ templates/
COPY pyproject.toml .
COPY uv.lock .

# Installer les dépendances avec uv en utilisant pyproject.toml
RUN uv sync --frozen --no-install-project

# Exposer le port si nécessaire (à ajuster selon votre application)
EXPOSE 8000

# Commande pour exécuter l'application
CMD ["uv", "run", "app.py"] 