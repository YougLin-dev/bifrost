# Enterprise Single-Node Deployment

This directory keeps two deployment modes only:

- `docker-compose.yml`: build the Bifrost image from the current repository checkout
- `docker-compose.cnb.yml`: pull a prebuilt image from CNB Docker registry

## 1. Build and run from the current repo

```bash
cp examples/dockers/enterprise-single-node/.env.example \
   examples/dockers/enterprise-single-node/.env

docker compose \
  -f examples/dockers/enterprise-single-node/docker-compose.yml \
  --env-file examples/dockers/enterprise-single-node/.env \
  up -d --build
```

## 2. Push to CNB Docker registry

Login:

```bash
docker login docker.cnb.cool -u cnb -p ${CNB_TOKEN}
```

### Local command push

Same-name artifact:

```bash
make docker-push-enterprise-cnb \
  CNB_REPO_SLUG_LOWERCASE=your-org/your-repo \
  ENTERPRISE_TAG=latest
```

Non-same-name artifact:

```bash
make docker-push-enterprise-cnb \
  CNB_REPO_SLUG_LOWERCASE=your-org/your-repo \
  CNB_IMAGE_NAME=bifrost-enterprise \
  ENTERPRISE_TAG=latest
```

These commands resolve to:

- same-name: `docker.cnb.cool/${CNB_REPO_SLUG_LOWERCASE}:latest`
- non-same-name: `docker.cnb.cool/${CNB_REPO_SLUG_LOWERCASE}/${CNB_IMAGE_NAME}:latest`

### CNB cloud-native build push

Use [`.cnb.yml.example`](/home/ai/tmp/bifrost/examples/dockers/enterprise-single-node/.cnb.yml.example) as the template. It logs in to CNB, builds with the enterprise Dockerfile, and pushes `latest`.

## 3. Run from a CNB image

Copy the CNB env template:

```bash
cp examples/dockers/enterprise-single-node/.env.cnb.example \
   examples/dockers/enterprise-single-node/.env.cnb
```

Edit `CNB_ARTIFACT_PATH`:

- same-name artifact: `your-org/your-repo`
- non-same-name artifact: `your-org/your-repo/bifrost-enterprise`

Then start the stack:

```bash
docker compose \
  -f examples/dockers/enterprise-single-node/docker-compose.cnb.yml \
  --env-file examples/dockers/enterprise-single-node/.env.cnb \
  up -d
```

`docker-compose.cnb.yml` now expects a full image reference via `BIFROST_IMAGE`.

- same-name artifact: `docker.cnb.cool/${CNB_REPO_SLUG_LOWERCASE}:latest`
- non-same-name artifact: `docker.cnb.cool/${CNB_REPO_SLUG_LOWERCASE}/bifrost-enterprise:latest`

Do not put the registry or tag into a separate path variable anymore. A value like `docker.cnb.cool/fastrouter/bifrost:latest` must be assigned directly to `BIFROST_IMAGE`.

## Notes

- The startup script writes `/app/data/config.json` on each boot, but the file only stores `env.*` references for secrets.
- PostgreSQL 16 is required by the Bifrost log store and is already pinned in the compose files.
- `BIFROST_ENCRYPTION_KEY` should be treated as durable state. Rotating it after data is written requires a planned migration or database reset.
