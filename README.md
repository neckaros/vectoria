Vector database tool for pro


# Vector DB
## To start everything:

```docker compose up -d```   

Api will be running on port 7979

## To start only Vector DB (usefull to dev on api with hot reload):
```docker compose -f docker-compose.dev.yml up -d```

## Ro stop and delete volumes with all data

```docker compose down -v```   

## Logs

```docker compose logs -f vectoria```   

# API
run with hot reload (run on port 7980)

```
cd api
uv sync
uv run dev
```